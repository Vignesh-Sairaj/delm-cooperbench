"""Zero-invasion CLI and suite orchestrator for running DeLM on CooperBench."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import litellm
from cooperbench.eval import evaluate
from cooperbench.runner.coop import execute_coop
from cooperbench.runner.tasks import discover_tasks

import delm_cb.adapter  # noqa: F401 - registers "delm_msa_v2"
from delm_cb.board_runtime import BOARD_WRITE_ACTIONS, PairBoardRuntime
from delm_cb.report import MIDPOINT_ARMS, MidpointArm, export_run, generate_midpoint_comparison

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COOPERBENCH_ROOT = Path("/usr/local/google/home/vigneshsairaj/CooperBench")
DEFAULT_DATASET_DIR = COOPERBENCH_ROOT / "dataset"
DEFAULT_LOGS_DIR = PACKAGE_ROOT / "logs"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"
DEFAULT_SHARDS_DIR = PACKAGE_ROOT / "_shards"
DEFAULT_ENV_FILE = COOPERBENCH_ROOT / ".env"
SHARD_RE = re.compile(r"-shard(\d+)of(\d+)$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def run_benchmark(
    *,
    run_name: str,
    subset: str | None = None,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    model: str = "vertex_ai/gemini-3.8-flash",
    agent_config: str | None = None,
    concurrency: int = 2,
    force: bool = False,
    auto_eval: bool = True,
    eval_concurrency: int = 4,
    key_slot: int | None = None,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    env_file: Path | str | None = None,
    command_str: str | None = None,
) -> dict[str, Any]:
    """Run DeLM coop benchmark pairs on CooperBench without modifying ``~/CooperBench``."""
    _load_cooperbench_env(model_name=model, key_slot=key_slot, env_file=env_file)
    litellm.suppress_debug_info = True

    dataset_path = _resolve_dataset_dir(dataset_dir)
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    results_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    run_dir = logs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_tasks(
        subset=subset,
        repo_filter=repo,
        task_filter=task_id,
        features_filter=features,
        dataset_dir=str(dataset_path),
    )
    if task_id is not None:
        tasks = [t for t in tasks if int(t["task_id"]) == int(task_id)]
    if not tasks:
        raise RuntimeError("No matching CooperBench tasks discovered")

    resolved_agent_config = _resolve_agent_config_path(agent_config)
    _write_run_config(
        run_dir=run_dir,
        run_name=run_name,
        model=model,
        concurrency=concurrency,
        total_tasks=len(tasks),
    )
    _write_launch_metadata(
        run_dir=run_dir,
        run_name=run_name,
        subset=subset,
        concurrency=concurrency,
        key_slot=key_slot,
        command_str=command_str or f"python -m delm_cb.cli run -n {run_name}",
        agent_config_path=resolved_agent_config,
    )

    started = time.monotonic()
    completed = 0
    skipped = 0
    failed = 0
    total_cost = 0.0
    results_list: list[dict[str, Any]] = []
    pair_board_summaries: list[dict[str, Any]] = []
    quiet_agents = len(tasks) > 1

    with _TeeConsole(run_dir / "console.log"):
        print(
            f"delm_cb run={run_name} pairs={len(tasks)} model={model} "
            f"concurrency={concurrency} key_slot={key_slot if key_slot is not None else 'default'}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_task = {
                executor.submit(
                    _run_one_pair,
                    task_info=task_info,
                    run_name=run_name,
                    model=model,
                    force=force,
                    quiet=quiet_agents,
                    agent_config=str(resolved_agent_config) if resolved_agent_config else None,
                    dataset_path=dataset_path,
                    logs_root=logs_root,
                ): task_info
                for task_info in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task_info = future_to_task[future]
                feat_str = ",".join(str(f) for f in sorted(task_info["features"]))
                task_label = f"{task_info['repo']}/{task_info['task_id']}/{feat_str}"
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    results_list.append({"task": task_label, "status": "error", "error": str(exc)})
                    print(f"[error] {task_label}: {exc}")
                    continue

                pair_result = outcome["result"]
                if outcome.get("board_summary"):
                    pair_board_summaries.append(outcome["board_summary"])
                if pair_result is None:
                    failed += 1
                    results_list.append({"task": task_label, "status": "failed", "cost": 0.0})
                    print(f"[failed] {task_label}")
                elif pair_result.get("skipped"):
                    skipped += 1
                    cost = float(pair_result.get("total_cost", 0.0) or 0.0)
                    total_cost += cost
                    results_list.append({"task": task_label, "status": "skip", "cost": cost})
                    print(f"[skip] {task_label}")
                else:
                    completed += 1
                    cost = float(pair_result.get("total_cost", 0.0) or 0.0)
                    total_cost += cost
                    results_list.append({"task": task_label, "status": "done", "cost": cost})
                    print(f"[done] {task_label} cost=${cost:.2f} time={pair_result.get('duration', 0):.0f}s")

        session_time = time.monotonic() - started
        _write_board_summary(run_dir=run_dir, run_name=run_name, pair_summaries=pair_board_summaries)
        summary = _write_run_summary(
            run_dir=run_dir,
            run_name=run_name,
            total_tasks=len(tasks),
            completed=completed,
            skipped=skipped,
            failed=failed,
            total_cost=total_cost,
            total_time_seconds=session_time,
            results_list=results_list,
        )

        if auto_eval:
            evaluate(
                run_name=run_name,
                subset=subset,
                repo=repo,
                task_id=task_id,
                features=features,
                concurrency=eval_concurrency,
                force=force,
                backend="docker",
                dataset_dir=str(dataset_path),
                logs_dir=str(logs_root),
            )
            eval_summary_path = run_dir / "eval_summary.json"
            if eval_summary_path.exists():
                eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
                summary["pass_rate"] = eval_summary.get("pass_rate")
                summary["eval"] = eval_summary
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        export_run(run_name, logs_dir=logs_root, results_dir=results_root)
    return summary


def evaluate_run(
    *,
    run_name: str,
    subset: str | None = None,
    repo: str | None = None,
    task_id: int | None = None,
    features: list[int] | None = None,
    concurrency: int = 4,
    force: bool = False,
    dataset_dir: Path | str | None = None,
    logs_dir: Path | str | None = None,
) -> None:
    """Evaluate a completed DeLM run using CooperBench's Docker grader."""
    dataset_path = _resolve_dataset_dir(dataset_dir)
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    evaluate(
        run_name=run_name,
        subset=subset,
        repo=repo,
        task_id=task_id,
        features=features,
        concurrency=concurrency,
        force=force,
        backend="docker",
        dataset_dir=str(dataset_path),
        logs_dir=str(logs_root),
    )


def split_subset(
    *,
    subset: str,
    num_shards: int,
    dataset_dir: Path | str | None = None,
    shards_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Split a CooperBench subset into ``num_shards`` symlinked shard datasets."""
    dataset_path = _resolve_dataset_dir(dataset_dir)
    shards_root = Path(shards_dir) if shards_dir is not None else DEFAULT_SHARDS_DIR
    subset_path = dataset_path / "subsets" / f"{subset}.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")

    data = json.loads(subset_path.read_text(encoding="utf-8"))
    pairs = [(t["repo"], t["task_id"], p) for t in data["tasks"] for p in t["pairs"]]
    if num_shards < 2 or num_shards > len(pairs):
        raise ValueError(f"num_shards must be between 2 and {len(pairs)}, got {num_shards}")

    shards_info: list[dict[str, Any]] = []
    for k in range(1, num_shards + 1):
        name = f"{subset}-shard{k}of{num_shards}"
        root = shards_root / name
        (root / "subsets").mkdir(parents=True, exist_ok=True)
        for entry in dataset_path.iterdir():
            link = root / entry.name
            if entry.is_dir() and entry.name != "subsets" and not link.exists():
                link.symlink_to(entry.resolve())

        tasks: dict[tuple[str, int], list[Any]] = {}
        for repo_name, task_id, pair in pairs[k - 1 :: num_shards]:
            tasks.setdefault((repo_name, task_id), []).append(pair)
        shard_payload = {
            "name": name,
            "description": f"Shard {k} of {num_shards} of the {subset} subset.",
            "parent": subset,
            "tasks": [{"repo": r, "task_id": t, "pairs": ps} for (r, t), ps in tasks.items()],
        }
        (root / "subsets" / f"{name}.json").write_text(
            json.dumps(shard_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        shards_info.append(
            {
                "shard_name": name,
                "dataset_dir": root,
                "pairs": sum(len(ps) for ps in tasks.values()),
            }
        )
    return shards_info


def merge_shards(
    *,
    shard_runs: list[str],
    logs_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Merge completed shard run directories under ``logs/`` using hard links."""
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    bases = {SHARD_RE.sub("", name) for name in shard_runs}
    if len(bases) != 1 or not all(SHARD_RE.search(name) for name in shard_runs):
        raise ValueError("Pass shard runs of one experiment (names ending in -shard<k>of<n>)")

    combined = bases.pop()
    out = logs_root / combined
    if out.exists():
        if force:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"Merged run directory already exists: {out}")

    dirs = [logs_root / name for name in shard_runs]
    missing = [d.name for d in dirs if not (d / "summary.json").exists()]
    if missing:
        raise RuntimeError(f"Unfinished or missing shard runs: {missing}")

    configs = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in dirs]
    setting = str(configs[0]["setting"])
    seen: dict[Path, str] = {}
    for d in dirs:
        for result_file in (d / setting).glob("*/*/*/result.json"):
            pair_rel = result_file.parent.relative_to(d / setting)
            if pair_rel in seen:
                raise RuntimeError(f"Duplicate pair {pair_rel} in {seen[pair_rel]} and {d.name}")
            seen[pair_rel] = d.name

    out.mkdir(parents=True, exist_ok=True)
    summaries = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in dirs]
    launches = [
        json.loads((d / "launch.json").read_text(encoding="utf-8")) if (d / "launch.json").exists() else {}
        for d in dirs
    ]
    parent_subset = SHARD_RE.sub("", str(launches[0].get("subset") or "")) or None

    for d in dirs:
        if (d / setting).is_dir():
            shutil.copytree(d / setting, out / setting, copy_function=os.link, dirs_exist_ok=True)

    config = {
        **configs[0],
        "run_name": combined,
        "total_tasks": sum(int(c.get("total_tasks", 0)) for c in configs),
        "started_at": min((c.get("started_at") or "") for c in configs),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    merged_results: list[dict[str, Any]] = []
    for s in summaries:
        merged_results.extend(s.get("results") or [])
    summary = {
        "run_name": combined,
        "completed_at": max((s.get("completed_at") or "") for s in summaries),
        "total_tasks": sum(int(s.get("total_tasks", 0)) for s in summaries),
        "completed": sum(int(s.get("completed", 0)) for s in summaries),
        "skipped": sum(int(s.get("skipped", 0)) for s in summaries),
        "failed": sum(int(s.get("failed", 0)) for s in summaries),
        "total_cost": sum(float(s.get("total_cost", 0.0)) for s in summaries),
        "total_time_seconds": max(float(s.get("total_time_seconds", 0.0)) for s in summaries),
        "shards": shard_runs,
        "results": merged_results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    launch = {
        **launches[0],
        "run_name": combined,
        "subset": parent_subset,
        "key_slot": None,
        "shards": [{"run_name": r, "key_slot": lch.get("key_slot")} for r, lch in zip(shard_runs, launches)],
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")

    with (out / "console.log").open("w", encoding="utf-8") as handle:
        for d in dirs:
            console_path = d / "console.log"
            if console_path.exists():
                handle.write(console_path.read_text(encoding="utf-8", errors="replace"))

    merged_board_pairs: list[dict[str, Any]] = []
    for d in dirs:
        bs_path = d / "delm_board_summary.json"
        if bs_path.exists():
            bs = json.loads(bs_path.read_text(encoding="utf-8"))
            merged_board_pairs.extend(bs.get("pairs") or [])
    _write_board_summary(run_dir=out, run_name=combined, pair_summaries=merged_board_pairs)
    return out


def run_midpoint_suite(
    *,
    arms: list[str] | None = None,
    shards: int = 4,
    concurrency_per_shard: int = 2,
    eval_concurrency: int = 4,
    dry_run: bool = False,
    force: bool = False,
    logs_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    shards_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Orchestrate the 4 Stage-4 Midpoint Reference DeLM arms sharded across KEY_SLOT=1..4."""
    selected: list[MidpointArm] = []
    wanted = set(arms) if arms else None
    for arm in MIDPOINT_ARMS:
        if wanted is None or arm.run_name in wanted:
            selected.append(arm)
    if wanted is not None and len(selected) != len(wanted):
        known = {a.run_name for a in MIDPOINT_ARMS}
        unknown = sorted(wanted - known)
        raise ValueError(f"Unknown arm(s): {unknown}. Expected subset of {sorted(known)}")

    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    results_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    shards_root = Path(shards_dir) if shards_dir is not None else DEFAULT_SHARDS_DIR

    plan: list[dict[str, Any]] = []
    for arm in selected:
        shard_infos = split_subset(subset=arm.subset, num_shards=shards, shards_dir=shards_root)
        shard_commands = []
        shard_run_names = []
        for slot, info in enumerate(shard_infos, start=1):
            shard_run_name = f"{arm.run_name}-shard{slot}of{shards}"
            shard_run_names.append(shard_run_name)
            cmd = [
                sys.executable,
                "-m",
                "delm_cb.cli",
                "run",
                "-n",
                shard_run_name,
                "-s",
                info["shard_name"],
                "-m",
                arm.model,
                "--agent-config",
                str(PACKAGE_ROOT / arm.agent_config),
                "-c",
                str(concurrency_per_shard),
                "--key-slot",
                str(slot),
                "--dataset-dir",
                str(info["dataset_dir"]),
                "--logs-dir",
                str(logs_root),
                "--results-dir",
                str(results_root),
                "--no-auto-eval",
            ]
            if force:
                cmd.append("--force")
            shard_commands.append({"key_slot": slot, "shard_run": shard_run_name, "cmd": cmd})

        arm_plan = {
            "arm": arm.run_name,
            "subset": arm.subset,
            "model": arm.model,
            "thinking": arm.thinking,
            "shards": shard_commands,
        }
        plan.append(arm_plan)

        if dry_run:
            continue

        _execute_arm_shards(
            arm=arm,
            shard_commands=shard_commands,
            shard_run_names=shard_run_names,
            eval_concurrency=eval_concurrency,
            force=force,
            logs_root=logs_root,
            results_root=results_root,
        )

    return plan


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``python -m delm_cb.cli``."""
    parser = argparse.ArgumentParser(prog="delm-cb", description="Zero-invasion DeLM on CooperBench CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run DeLM coop benchmark pairs")
    p_run.add_argument("-n", "--name", required=True, help="Run name under logs/")
    p_run.add_argument("-s", "--subset", default=None)
    p_run.add_argument("-r", "--repo", default=None)
    p_run.add_argument("-t", "--task", type=int, default=None)
    p_run.add_argument("-f", "--features", default=None, help="Comma-separated pair e.g. 4,8")
    p_run.add_argument("-m", "--model", default="vertex_ai/gemini-3.8-flash")
    p_run.add_argument("--agent-config", default=str(PACKAGE_ROOT / "configs/reference_medium.yaml"))
    p_run.add_argument("-c", "--concurrency", type=int, default=2)
    p_run.add_argument("--eval-concurrency", type=int, default=4)
    p_run.add_argument("--key-slot", type=int, default=None)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--no-auto-eval", action="store_true")
    p_run.add_argument("--dataset-dir", default=None)
    p_run.add_argument("--logs-dir", default=None)
    p_run.add_argument("--results-dir", default=None)

    p_eval = sub.add_parser("eval", help="Evaluate an existing run with CooperBench grader")
    p_eval.add_argument("-n", "--name", required=True)
    p_eval.add_argument("-s", "--subset", default=None)
    p_eval.add_argument("-r", "--repo", default=None)
    p_eval.add_argument("-t", "--task", type=int, default=None)
    p_eval.add_argument("-f", "--features", default=None)
    p_eval.add_argument("-c", "--concurrency", type=int, default=4)
    p_eval.add_argument("--force", action="store_true")
    p_eval.add_argument("--dataset-dir", default=None)
    p_eval.add_argument("--logs-dir", default=None)

    p_split = sub.add_parser("split", help="Split a subset into shard datasets")
    p_split.add_argument("subset")
    p_split.add_argument("n", type=int)
    p_split.add_argument("--dataset-dir", default=None)
    p_split.add_argument("--shards-dir", default=None)

    p_merge = sub.add_parser("merge", help="Merge shard runs under logs/")
    p_merge.add_argument("shard_runs", nargs="+")
    p_merge.add_argument("--logs-dir", default=None)
    p_merge.add_argument("--force", action="store_true")

    p_report = sub.add_parser("report", help="Export run metrics and Stage-4 comparison table")
    p_report.add_argument("run_name", nargs="?", default=None)
    p_report.add_argument("--logs-dir", default=None)
    p_report.add_argument("--results-dir", default=None)

    p_suite = sub.add_parser("suite", help="Run Stage-4 Midpoint Reference arms sharded across KEY_SLOT=1..4")
    p_suite.add_argument("--arms", nargs="+", default=None)
    p_suite.add_argument("--shards", type=int, default=4)
    p_suite.add_argument("-c", "--concurrency-per-shard", type=int, default=2)
    p_suite.add_argument("--eval-concurrency", type=int, default=4)
    p_suite.add_argument("--dry-run", action="store_true")
    p_suite.add_argument("--force", action="store_true")
    p_suite.add_argument("--logs-dir", default=None)
    p_suite.add_argument("--results-dir", default=None)
    p_suite.add_argument("--shards-dir", default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        features = [int(x) for x in args.features.split(",")] if args.features else None
        run_benchmark(
            run_name=args.name,
            subset=args.subset,
            repo=args.repo,
            task_id=args.task,
            features=features,
            model=args.model,
            agent_config=args.agent_config,
            concurrency=args.concurrency,
            force=args.force,
            auto_eval=not args.no_auto_eval,
            eval_concurrency=args.eval_concurrency,
            key_slot=args.key_slot,
            dataset_dir=args.dataset_dir,
            logs_dir=args.logs_dir,
            results_dir=args.results_dir,
            command_str=" ".join(sys.argv),
        )
        return 0
    if args.command == "eval":
        features = [int(x) for x in args.features.split(",")] if args.features else None
        evaluate_run(
            run_name=args.name,
            subset=args.subset,
            repo=args.repo,
            task_id=args.task,
            features=features,
            concurrency=args.concurrency,
            force=args.force,
            dataset_dir=args.dataset_dir,
            logs_dir=args.logs_dir,
        )
        return 0
    if args.command == "split":
        for item in split_subset(
            subset=args.subset,
            num_shards=args.n,
            dataset_dir=args.dataset_dir,
            shards_dir=args.shards_dir,
        ):
            print(f"{item['shard_name']}: {item['pairs']} pairs -> {item['dataset_dir']}")
        return 0
    if args.command == "merge":
        out = merge_shards(shard_runs=args.shard_runs, logs_dir=args.logs_dir, force=args.force)
        print(f"merged shards into {out}")
        return 0
    if args.command == "report":
        if args.run_name:
            run = export_run(args.run_name, logs_dir=args.logs_dir, results_dir=args.results_dir)
            h = run["headline"]
            print(f"exported {h['pairs']} pairs for {args.run_name} (passed {h['passed']}/{h['pairs']})")
        comp = generate_midpoint_comparison(logs_dir=args.logs_dir, results_dir=args.results_dir)
        print(f"wrote {comp}")
        return 0
    if args.command == "suite":
        plan = run_midpoint_suite(
            arms=args.arms,
            shards=args.shards,
            concurrency_per_shard=args.concurrency_per_shard,
            eval_concurrency=args.eval_concurrency,
            dry_run=args.dry_run,
            force=args.force,
            logs_dir=args.logs_dir,
            results_dir=args.results_dir,
            shards_dir=args.shards_dir,
        )
        if args.dry_run:
            print(json.dumps(plan, indent=2))
        return 0
    return 1


def _load_cooperbench_env(
    *,
    model_name: str,
    key_slot: int | None,
    env_file: Path | str | None = None,
) -> None:
    env_path = Path(env_file) if env_file is not None else DEFAULT_ENV_FILE
    if env_path.exists():
        for key, value in _parse_env_file(env_path).items():
            os.environ.setdefault(key, value)

    os.environ["LITELLM_LOG"] = "ERROR"
    os.environ["MSWEA_SILENT_STARTUP"] = "1"
    os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

    if model_name.startswith("vertex_ai/"):
        os.environ.setdefault("VERTEXAI_LOCATION", "global")
        if key_slot is not None:
            proj_var = f"VERTEXAI_PROJECT_{key_slot}"
            project_val = os.environ.get(proj_var, "").strip()
            if not project_val:
                raise RuntimeError(f"KEY_SLOT={key_slot} with {model_name} requires {proj_var} in {env_path}")
            os.environ["VERTEXAI_PROJECT"] = project_val
        else:
            project_val = (os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
            if not project_val:
                raise RuntimeError(f"{model_name} requires VERTEXAI_PROJECT in {env_path}")
            os.environ["VERTEXAI_PROJECT"] = project_val
    elif model_name.startswith("gemini/") and key_slot is not None:
        key_var = f"GEMINI_API_KEY_{key_slot}"
        key_val = os.environ.get(key_var, "").strip()
        if not key_val:
            raise RuntimeError(f"KEY_SLOT={key_slot} requires {key_var} in {env_path}")
        os.environ["GEMINI_API_KEY"] = key_val


def _parse_env_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.match(key):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        parsed[key] = val
    return parsed


def _resolve_dataset_dir(dataset_dir: Path | str | None) -> Path:
    return Path(dataset_dir).resolve() if dataset_dir is not None else DEFAULT_DATASET_DIR


def _resolve_agent_config_path(agent_config: str | None) -> Path | None:
    if not agent_config:
        return None
    candidate = Path(agent_config)
    if candidate.exists():
        return candidate.resolve()
    rel = PACKAGE_ROOT / agent_config
    if rel.exists():
        return rel.resolve()
    raise FileNotFoundError(f"Agent config file not found: {agent_config}")


def _run_one_pair(
    *,
    task_info: dict[str, Any],
    run_name: str,
    model: str,
    force: bool,
    quiet: bool,
    agent_config: str | None,
    dataset_path: Path,
    logs_root: Path,
) -> dict[str, Any]:
    repo_name = str(task_info["repo"])
    task_id = int(task_info["task_id"])
    features = sorted(int(f) for f in task_info["features"])
    feature_str = "_".join(f"f{f}" for f in features)
    pair_log_dir = logs_root / run_name / "coop" / repo_name / str(task_id) / feature_str

    prev_result = _check_completed_pair(pair_log_dir) if not force else None
    if prev_result is not None:
        return {"result": {"skipped": True, **prev_result}, "board_summary": _read_board_summary(pair_log_dir)}

    with PairBoardRuntime(
        pair_log_dir=pair_log_dir,
        repo_name=repo_name,
        task_id=task_id,
        features=features,
        dataset_dir=dataset_path,
    ):
        result = execute_coop(
            repo_name=repo_name,
            task_id=task_id,
            features=features,
            run_name=run_name,
            agent_name="delm_msa_v2",
            model_name=model,
            force=force,
            quiet=quiet,
            git_enabled=True,
            messaging_enabled=False,
            backend="docker",
            agent_config=agent_config,
            dataset_dir=str(dataset_path),
            logs_dir=str(logs_root),
        )
    board_summary = _backfill_pair_logs(pair_log_dir)
    return {"result": result, "board_summary": board_summary}


def _check_completed_pair(pair_log_dir: Path) -> dict[str, Any] | None:
    result_file = pair_log_dir / "result.json"
    if not result_file.exists():
        return None
    prev_result = json.loads(result_file.read_text(encoding="utf-8"))
    had_error = any(a.get("status") == "Error" for a in (prev_result.get("agents") or {}).values())
    return None if had_error else prev_result


def _read_board_summary(pair_log_dir: Path) -> dict[str, Any] | None:
    events_path = pair_log_dir / "delm_board_events.json"
    if not events_path.exists():
        return None
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    return payload.get("summary")


def _backfill_pair_logs(pair_log_dir: Path) -> dict[str, Any] | None:
    events_path = pair_log_dir / "delm_board_events.json"
    if not events_path.exists():
        return None
    payload = json.loads(events_path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}

    conversation = _extract_board_conversation(payload)
    (pair_log_dir / "conversation.json").write_text(
        json.dumps(conversation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result_path = pair_log_dir / "result.json"
    if result_path.exists():
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
        result_data["messages_sent"] = int(summary.get("board_writes_total", 0))
        result_data["delm_board"] = summary
        result_path.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")

    return summary


def _extract_board_conversation(payload: dict[str, Any]) -> list[dict[str, Any]]:
    preseed_max = int(payload.get("preseed_max_sequence", 0))
    entries: list[dict[str, Any]] = []
    for ev in payload.get("events") or []:
        seq = int(ev.get("sequence", 0))
        kind = ev.get("kind", "")
        ev_payload = ev.get("payload") or {}
        if seq <= preseed_max:
            continue
        if kind == "read" and ev_payload.get("via") == "read":
            continue
        if kind in BOARD_WRITE_ACTIONS or kind in ("body", "wait", "apply_error", "session"):
            entries.append(
                {
                    "sequence": seq,
                    "time_ns": ev.get("time_ns"),
                    "from": ev.get("actor"),
                    "kind": kind,
                    "payload": ev_payload,
                }
            )
    return entries


def _write_run_config(
    *,
    run_dir: Path,
    run_name: str,
    model: str,
    concurrency: int,
    total_tasks: int,
) -> None:
    payload = {
        "run_name": run_name,
        "agent_framework": "delm_msa_v2",
        "model": model,
        "setting": "coop",
        "concurrency": concurrency,
        "total_tasks": total_tasks,
        "started_at": dt.datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_launch_metadata(
    *,
    run_dir: Path,
    run_name: str,
    subset: str | None,
    concurrency: int,
    key_slot: int | None,
    command_str: str,
    agent_config_path: Path | None,
) -> None:
    versions: dict[str, str] = {}
    for pkg in ("cooperbench", "litellm"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "unknown"

    payload = {
        "run_name": run_name,
        "setting": "coop",
        "subset": subset,
        "git_enabled": True,
        "concurrency": concurrency,
        "key_slot": int(key_slot) if key_slot is not None else None,
        "command": command_str,
        "launched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _run_git_stdout("rev-parse", "HEAD"),
        "git_dirty": bool(_run_git_stdout("status", "--porcelain", "--untracked-files=no")),
        "agent_config_path": str(agent_config_path) if agent_config_path else None,
        "agent_config": agent_config_path.read_text(encoding="utf-8") if agent_config_path else None,
        "versions": versions,
        "cpus": os.cpu_count(),
    }
    (run_dir / "launch.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_git_stdout(*args: str) -> str:
    res = subprocess.run(["git", "-C", str(PACKAGE_ROOT), *args], capture_output=True, text=True)
    return res.stdout.strip()


def _write_run_summary(
    *,
    run_dir: Path,
    run_name: str,
    total_tasks: int,
    completed: int,
    skipped: int,
    failed: int,
    total_cost: float,
    total_time_seconds: float,
    results_list: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "run_name": run_name,
        "completed_at": dt.datetime.now().isoformat(),
        "pass_rate": None,
        "total_tasks": total_tasks,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total_cost": round(total_cost, 6),
        "total_time_seconds": round(total_time_seconds, 2),
        "results": results_list,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_board_summary(
    *,
    run_dir: Path,
    run_name: str,
    pair_summaries: list[dict[str, Any]],
) -> None:
    total_writes = sum(int(s.get("board_writes_total", 0)) for s in pair_summaries)
    totals_by_kind: Counter[str] = Counter()
    totals_by_command: Counter[str] = Counter()
    for s in pair_summaries:
        for kind, cnt in (s.get("by_kind") or {}).items():
            totals_by_kind[kind] += int(cnt)
        for cmds in (s.get("commands_by_actor") or {}).values():
            for cmd, cnt in cmds.items():
                totals_by_command[cmd] += int(cnt)
    payload = {
        "run_name": run_name,
        "pairs_with_board": len(pair_summaries),
        "board_writes_total": total_writes,
        "totals_by_kind": dict(totals_by_kind),
        "totals_by_command": dict(totals_by_command),
        "pairs": pair_summaries,
    }
    (run_dir / "delm_board_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _execute_arm_shards(
    *,
    arm: MidpointArm,
    shard_commands: list[dict[str, Any]],
    shard_run_names: list[str],
    eval_concurrency: int,
    force: bool,
    logs_root: Path,
    results_root: Path,
) -> None:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    needed_pp = f"{PACKAGE_ROOT / 'src'}:{PACKAGE_ROOT / 'vendor' / 'delm'}"
    env["PYTHONPATH"] = f"{needed_pp}:{existing_pp}" if existing_pp else needed_pp

    procs: list[tuple[str, subprocess.Popen[str]]] = []
    for entry in shard_commands:
        proc = subprocess.Popen(entry["cmd"], env=env)
        procs.append((entry["shard_run"], proc))

    failures: list[tuple[str, int]] = []
    for shard_run, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append((shard_run, rc))
    if failures:
        raise RuntimeError(f"Shard failure(s) for {arm.run_name}: {failures}")

    merge_shards(shard_runs=shard_run_names, logs_dir=logs_root, force=force)
    evaluate_run(run_name=arm.run_name, concurrency=eval_concurrency, force=force, logs_dir=logs_root)
    export_run(arm.run_name, logs_dir=logs_root, results_dir=results_root)
    generate_midpoint_comparison(logs_dir=logs_root, results_dir=results_root)


class _TeeStream:
    def __init__(self, primary: Any, log_file: Any, lock: threading.Lock) -> None:
        self._primary = primary
        self._log_file = log_file
        self._lock = lock

    def write(self, data: str) -> int:
        with self._lock:
            self._primary.write(data)
            self._log_file.write(data)
            self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._primary.flush()
            self._log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


class _TeeConsole:
    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._stack = ExitStack()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def __enter__(self) -> _TeeConsole:
        lock = threading.Lock()
        handle = self._stack.enter_context(self._log_path.open("a", encoding="utf-8"))
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, handle, lock)
        sys.stderr = _TeeStream(self._orig_stderr, handle, lock)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._stack.close()


if __name__ == "__main__":
    sys.exit(main())
