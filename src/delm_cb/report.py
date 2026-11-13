"""Result export and Stage-4 Midpoint comparison reporting for DeLM on CooperBench."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = PACKAGE_ROOT / "logs"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"

EXPERIMENT_HEADER = "## Experiment"
OBSERVATIONS_HEADER = "## Observations"
HAND_WRITTEN = {
    EXPERIMENT_HEADER: "_(What this run tests, how it differs from other runs, and why we ran it.)_",
    OBSERVATIONS_HEADER: "_(What we saw in the traces.)_",
}

STEP_LIMIT_RE = re.compile(r"^\s*step_limit:\s*(\d+)", re.MULTILINE)
COMPACTION_RE = re.compile(r"^\s*compaction_token_trigger:\s*(\d+)", re.MULTILINE)
THINKING_RE = re.compile(r"^\s*thinkingLevel:\s*([A-Za-z]+)", re.MULTILINE)
RATE_LIMIT_RE = re.compile(r"^Retrying .*RateLimitError", re.IGNORECASE)
SERVER_BUSY_RE = re.compile(
    r"^Retrying .*(ServiceUnavailableError|InternalServerError|BadGatewayError|GatewayTimeoutError)",
    re.IGNORECASE,
)

PAIR_FIELDS = [
    "repo",
    "task_id",
    "feature_a",
    "feature_b",
    "setting",
    "both_passed",
    "clean_merge_passed",
    "feature_a_passed",
    "feature_b_passed",
    "merge_status",
    "merge_strategy",
    "eval_error",
    "n_agents",
    "agents_at_step_limit",
    "agents_errored",
    "steps_total",
    "cost_usd",
    "duration_s",
    "messages_sent",
    "board_writes",
    "notes",
    "applies",
    "apply_conflicts",
    "waits",
    "board_errors",
    "patch_lines_total",
]

AGENT_FIELDS = [
    "repo",
    "task_id",
    "feature_a",
    "feature_b",
    "agent_id",
    "feature_id",
    "role",
    "status",
    "steps",
    "cost_usd",
    "patch_lines",
    "model_calls",
    "reasoning_tokens",
    "compactions",
    "board_writes",
    "notes",
    "applies",
    "apply_conflicts",
    "waits",
    "board_errors",
    "error",
]


@dataclass(frozen=True)
class MidpointArm:
    """Definition of one Stage-4 Midpoint Reference comparison arm."""

    run_name: str
    subset: str
    model: str
    thinking: str
    agent_config: str
    baselines: tuple[tuple[str, str], ...]


MIDPOINT_ARMS: tuple[MidpointArm, ...] = (
    MidpointArm(
        run_name="core-ref-delm",
        subset="core",
        model="vertex_ai/gemini-3.8-flash",
        thinking="medium",
        agent_config="configs/reference_medium.yaml",
        baselines=(
            ("`solo` ref", "solo-msa_v2-gemini-3-8-flash-core-ref"),
            ("`coop --git` ref", "coop-msa_v2-git-gemini-3-8-flash-core-ref"),
            ("`ref-intents`", "coop-msa_v2-git-gemini-3-8-flash-core-ref-intents"),
            ("`ref-delivery-timing`", "coop-msa_v2-git-gemini-3-8-flash-core-ref-delivery-timing"),
            ("`ref-comm-combined`", "coop-msa_v2-git-gemini-3-8-flash-core-ref-comm-combined"),
        ),
    ),
    MidpointArm(
        run_name="flash-ref-delm",
        subset="flash",
        model="vertex_ai/gemini-3.8-flash",
        thinking="medium",
        agent_config="configs/reference_medium.yaml",
        baselines=(
            ("`solo` ref", "solo-msa_v2-gemini-3-8-flash-flash-saturation-medium-128k-vertex"),
            ("`coop --git` ref #1", "coop-msa_v2-git-gemini-3-8-flash-flash-saturation-medium-128k-vertex"),
            ("`coop --git` ref #2", "coop-msa_v2-git-gemini-3-8-flash-flash-ref"),
            ("`ref-intents`", "coop-msa_v2-git-gemini-3-8-flash-flash-ref-intents"),
            ("`ref-delivery-timing`", "coop-msa_v2-git-gemini-3-8-flash-flash-ref-delivery-timing"),
            ("`ref-comm-combined`", "coop-msa_v2-git-gemini-3-8-flash-flash-ref-comm-combined"),
        ),
    ),
    MidpointArm(
        run_name="flash-ref-low-delm",
        subset="flash",
        model="vertex_ai/gemini-3.8-flash",
        thinking="low",
        agent_config="configs/reference_low.yaml",
        baselines=(
            ("`solo` ref-low", "solo-msa_v2-gemini-3-8-flash-flash-ref-low"),
            ("`coop --git` ref-low", "coop-msa_v2-git-gemini-3-8-flash-flash-ref-low"),
            ("`ref-low-intents`", "coop-msa_v2-git-gemini-3-8-flash-flash-ref-low-intents"),
        ),
    ),
    MidpointArm(
        run_name="3-flash-ref-low-delm",
        subset="flash",
        model="vertex_ai/gemini-3-flash-preview",
        thinking="low",
        agent_config="configs/reference_low.yaml",
        baselines=(
            ("`coop --git` 3-flash ref-low", "coop-msa_v2-git-gemini-3-flash-flash-ref-low"),
            ("`3-flash ref-low + delivery-timing`", "coop-msa_v2-git-gemini-3-flash-flash-ref-low-delivery-timing"),
        ),
    ),
)


def resolve_cooperbench_root() -> Path:
    """Resolve the CooperBench repository path from ``COOPERBENCH_ROOT`` or sibling directories."""
    override = os.environ.get("COOPERBENCH_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (PACKAGE_ROOT.parent / "CooperBench", PACKAGE_ROOT.parents[1] / "CooperBench"):
        if candidate.is_dir():
            return candidate
    return PACKAGE_ROOT.parent / "CooperBench"


def relativize_package_paths(value: Any) -> Any:
    """Strip ``PACKAGE_ROOT`` prefixes from command strings and config paths."""
    if not isinstance(value, str) or not value:
        return value
    text = value
    for root in sorted({str(PACKAGE_ROOT), str(PACKAGE_ROOT.resolve())}, key=len, reverse=True):
        text = text.replace(root + "/", "").replace(root, ".")
    return text


def export_run(
    run_name: str,
    *,
    logs_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Export one run from ``logs/<run_name>`` into ``results/<run_name>/``."""
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    results_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    run_dir = logs_root / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config = _load_json(run_dir / "config.json") or {}
    summary = _load_json(run_dir / "summary.json") or {}
    launch = _load_json(run_dir / "launch.json") or {}
    setting = str(config.get("setting") or launch.get("setting") or "coop")

    pairs: list[dict[str, Any]] = []
    agents: list[dict[str, Any]] = []
    traj_paths: list[Path] = []
    for result_file in sorted(run_dir.glob("*/*/*/*/result.json")):
        exported = _export_pair(result_file.parent, setting)
        if exported is not None:
            pairs.append(exported[0])
            agents.extend(exported[1])
            traj_paths.extend(exported[2])
    if not pairs:
        raise RuntimeError(f"No result.json files found under {run_dir}")

    passed = sum(bool(p["both_passed"]) for p in pairs)
    clean_passed = sum(bool(p["clean_merge_passed"]) for p in pairs)
    cost = sum(float(p["cost_usd"]) for p in pairs)
    board_writes_list = [int(p["board_writes"]) for p in pairs]
    load = _compute_load(run_dir, traj_paths)
    run = {
        "run_name": run_name,
        "setting": setting,
        "git_enabled": launch.get("git_enabled"),
        "agent_framework": config.get("agent_framework"),
        "model": config.get("model"),
        "subset": launch.get("subset"),
        "concurrency": launch.get("concurrency", config.get("concurrency")),
        "key_slot": launch.get("key_slot"),
        "command": relativize_package_paths(launch.get("command")),
        "git_commit": launch.get("git_commit"),
        "git_dirty": launch.get("git_dirty"),
        "agent_config_path": relativize_package_paths(launch.get("agent_config_path")),
        "agent_config": launch.get("agent_config"),
        "budgets": _agent_budgets(setting, launch.get("agent_config")),
        "versions": launch.get("versions"),
        "started_at": config.get("started_at"),
        "completed_at": summary.get("completed_at"),
        "raw_logs": f"logs/{run_name}",
        "headline": {
            "pairs": len(pairs),
            "passed": passed,
            "pass_rate": passed / len(pairs),
            "clean_merge_passed": clean_passed,
            "clean_merge_pass_rate": clean_passed / len(pairs),
            "merge_conflicts": sum(p["merge_status"] == "conflicts" for p in pairs),
            "agents_at_step_limit": sum(int(p["agents_at_step_limit"]) for p in pairs),
            "agents_errored": sum(int(p["agents_errored"]) for p in pairs),
            "cost_usd": round(cost, 4),
            "cost_per_pair_usd": round(cost / len(pairs), 4),
            "wall_minutes": round(float(summary.get("total_time_seconds") or 0) / 60, 1),
        },
        "delm": {
            "board_writes_total": sum(board_writes_list),
            "median_board_writes_per_pair": (
                statistics.median(board_writes_list) if board_writes_list else 0
            ),
            "notes": sum(int(p["notes"]) for p in pairs),
            "applies": sum(int(p["applies"]) for p in pairs),
            "apply_conflicts": sum(int(p["apply_conflicts"]) for p in pairs),
            "waits": sum(int(p["waits"]) for p in pairs),
            "board_errors": sum(int(p["board_errors"]) for p in pairs),
            "pairs_with_board_writes": sum(w > 0 for w in board_writes_list),
        },
        "load": load,
    }

    out = results_root / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    _write_csv(out / "pairs.csv", PAIR_FIELDS, pairs)
    _write_csv(out / "agents.csv", AGENT_FIELDS, agents)

    summary_md = out / "summary.md"
    summary_md.write_text(_render_summary(run, pairs, _read_hand_written(summary_md)), encoding="utf-8")
    return run


def generate_midpoint_comparison(
    *,
    logs_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    baseline_results_dir: Path | str | None = None,
) -> Path:
    """Build ``results/delm_midpoint_comparison.md`` comparing DeLM vs CooperBench baselines."""
    logs_root = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    results_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    baselines_root = (
        Path(baseline_results_dir)
        if baseline_results_dir is not None
        else resolve_cooperbench_root() / "results"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    lines = [
        "# DeLM vs. CooperBench Stage-4 Midpoint Reference",
        "",
        "Generated by `delm_cb.report`. Baseline metrics are read from CooperBench `results/`.",
        "",
        "## Headline Comparison",
        "",
        (
            "| Subset | Model (`thinking`) | Arm | Passed | Clean-Merge Passes | Conflicts "
            "| Paired Wins/Losses | Writes / Msgs | Notes / Applies / Conflicts / Waits | Cost | Wall Time |"
        ),
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
    ]

    arm_breakdowns: list[str] = []
    for arm in MIDPOINT_ARMS:
        baseline_data = [
            (label, name, _load_exported_run(baselines_root / name))
            for label, name in arm.baselines
        ]
        delm_data = _ensure_delm_exported(arm.run_name, logs_root=logs_root, results_root=results_root)

        for label, _, base in baseline_data:
            if base is None:
                lines.append(
                    f"| `{arm.subset}` | `{arm.model}` (`{arm.thinking}`) | {label} | (missing) | - | - | - | - | - | - | - |"
                )
                continue
            run_json, base_pairs = base
            lines.append(_format_baseline_row(arm, label, run_json, base_pairs))

        if delm_data is None:
            lines.append(
                f"| `{arm.subset}` | `{arm.model}` (`{arm.thinking}`) | **`{arm.run_name}`** | *(not yet run)* | - | - | - | - | - | - | - |"
            )
        else:
            delm_run, delm_pairs = delm_data
            lines.append(_format_delm_row(arm, delm_run, delm_pairs, baseline_data))
            arm_breakdowns.extend(_format_paired_breakdown(arm, delm_pairs, baseline_data))

    if arm_breakdowns:
        lines.extend(["", "## Paired Outcome Breakdown", *arm_breakdowns])

    out_path = results_root / "delm_midpoint_comparison.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _export_pair(pair_dir: Path, setting: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[Path]] | None:
    result = _load_json(pair_dir / "result.json")
    if result is None:
        return None
    ev = _load_json(pair_dir / "eval.json") or {}
    board_payload = _load_json(pair_dir / "delm_board_events.json") or {}
    board_summary = board_payload.get("summary") or {}
    writes_by_actor = board_summary.get("board_writes_by_actor") or {}
    commands_by_actor = board_summary.get("commands_by_actor") or {}

    features = result.get("features") or []
    fa, fb = (features + [None, None])[:2]
    merge = ev.get("merge") or {}
    merge_status = merge.get("status", "n/a") if setting != "solo" else "n/a"
    both_passed = bool(ev.get("both_passed", False))
    clean_merge_passed = both_passed and merge_status in ("clean", "identical")

    agents = _agents_from_result(result)
    agent_rows: list[dict[str, Any]] = []
    traj_paths: list[Path] = []
    pair_notes = 0
    pair_applies = 0
    pair_apply_conflicts = 0
    pair_waits = 0
    pair_board_errors = 0

    for agent_id, agent_info in agents.items():
        fid = agent_info.get("feature_id")
        traj_path = _resolve_traj_path(pair_dir, agent_id, fid)
        traj_paths.append(traj_path)
        stats = _trajectory_stats(traj_path)
        cmds = commands_by_actor.get(agent_id) or {}
        actor_writes = int(writes_by_actor.get(agent_id, 0))
        actor_notes = int(cmds.get("note", 0))
        actor_applies = int(cmds.get("apply", 0))
        actor_apply_conflicts = int(cmds.get("apply_conflict", 0))
        actor_waits = int(cmds.get("wait", 0))
        actor_errors = _count_actor_board_errors(pair_dir, agent_id)

        pair_notes += actor_notes
        pair_applies += actor_applies
        pair_apply_conflicts += actor_apply_conflicts
        pair_waits += actor_waits
        pair_board_errors += actor_errors

        role = "solo" if setting == "solo" else "peer"
        agent_rows.append(
            {
                "repo": result.get("repo"),
                "task_id": result.get("task_id"),
                "feature_a": fa,
                "feature_b": fb,
                "agent_id": agent_id,
                "feature_id": fid if fid is not None else "",
                "role": role,
                "status": agent_info.get("status"),
                "steps": agent_info.get("steps", 0),
                "cost_usd": round(float(agent_info.get("cost") or 0), 6),
                "patch_lines": agent_info.get("patch_lines", 0),
                **stats,
                "board_writes": actor_writes,
                "notes": actor_notes,
                "applies": actor_applies,
                "apply_conflicts": actor_apply_conflicts,
                "waits": actor_waits,
                "board_errors": actor_errors,
                "error": agent_info.get("error") or "",
            }
        )

    board_writes_total = int(board_summary.get("board_writes_total", result.get("messages_sent", 0)))
    pair_row = {
        "repo": result.get("repo"),
        "task_id": result.get("task_id"),
        "feature_a": fa,
        "feature_b": fb,
        "setting": setting,
        "both_passed": both_passed,
        "clean_merge_passed": clean_merge_passed,
        "feature_a_passed": bool((ev.get("feature1") or {}).get("passed", False)),
        "feature_b_passed": bool((ev.get("feature2") or {}).get("passed", False)),
        "merge_status": merge_status,
        "merge_strategy": merge.get("strategy", "") if setting != "solo" else "",
        "eval_error": ev.get("error") or ("not evaluated" if not ev else ""),
        "n_agents": len(agent_rows),
        "agents_at_step_limit": sum(r["status"] == "LimitsExceeded" for r in agent_rows),
        "agents_errored": sum(r["status"] == "Error" for r in agent_rows),
        "steps_total": sum(int(r["steps"] or 0) for r in agent_rows),
        "cost_usd": round(float(result.get("total_cost") or sum(float(r["cost_usd"]) for r in agent_rows)), 6),
        "duration_s": round(float(result.get("duration_seconds") or 0), 1),
        "messages_sent": result.get("messages_sent", board_writes_total),
        "board_writes": board_writes_total,
        "notes": pair_notes,
        "applies": pair_applies,
        "apply_conflicts": pair_apply_conflicts,
        "waits": pair_waits,
        "board_errors": pair_board_errors,
        "patch_lines_total": sum(int(r["patch_lines"] or 0) for r in agent_rows),
    }
    return pair_row, agent_rows, traj_paths


def _count_actor_board_errors(pair_dir: Path, actor: str) -> int:
    err_path = pair_dir / "delm" / "records" / actor / "context-command-errors.jsonl"
    if not err_path.exists():
        return 0
    return sum(1 for line in err_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())


def _agents_from_result(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "agents" in result:
        return result["agents"]
    if "agent" in result:
        return {"solo": {**result["agent"], "feature_id": None}}
    return {}


def _resolve_traj_path(pair_dir: Path, agent_id: str, feature_id: int | None) -> Path:
    full_name = "solo_full_traj.json" if agent_id == "solo" else f"{agent_id}_full_traj.json"
    full_path = pair_dir / full_name
    if full_path.exists():
        return full_path
    if agent_id == "solo":
        return pair_dir / "solo_traj.json"
    if feature_id is not None and (pair_dir / f"agent{feature_id}_traj.json").exists():
        return pair_dir / f"agent{feature_id}_traj.json"
    return pair_dir / f"{agent_id}_traj.json"


def _trajectory_stats(path: Path) -> dict[str, int | str]:
    traj = _load_json(path)
    if traj is None:
        return {"model_calls": "", "reasoning_tokens": "", "compactions": ""}
    segments = traj.get("segments") or []
    if segments:
        compactions = sum(s.get("kind") == "summarizer" for s in segments)
        solver_msgs = [m for s in segments if s.get("kind") != "summarizer" for m in s.get("messages", [])]
    else:
        solver_msgs = traj.get("messages") or []
        compactions = sum(bool((m.get("extra") or {}).get("summary")) for m in solver_msgs)

    calls = 0
    reasoning = 0
    seen: set[object] = set()
    for msg in solver_msgs:
        extra = msg.get("extra") or {}
        response = extra.get("response")
        if not response or extra.get("summary"):
            continue
        rid = response.get("id") or extra.get("timestamp") or id(msg)
        if rid in seen:
            continue
        seen.add(rid)
        calls += 1
        details = (response.get("usage") or {}).get("completion_tokens_details") or {}
        reasoning += details.get("reasoning_tokens") or 0
    return {"model_calls": calls, "reasoning_tokens": reasoning, "compactions": compactions}


def _compute_load(run_dir: Path, traj_paths: list[Path]) -> dict[str, int]:
    tokens_per_min: dict[int, int] = defaultdict(int)
    calls_per_min: dict[int, int] = defaultdict(int)
    seen: set[object] = set()

    for path in traj_paths:
        traj = _load_json(path)
        if traj is None:
            continue
        segments = traj.get("segments") or []
        messages = [m for s in segments for m in s.get("messages", [])] or traj.get("messages") or []
        for msg in messages:
            extra = msg.get("extra") or {}
            response = extra.get("response")
            ts = extra.get("timestamp")
            if not response or not ts:
                continue
            rid = response.get("id") or (str(path), ts)
            if rid in seen:
                continue
            seen.add(rid)
            bucket = int(ts // 60)
            tokens_per_min[bucket] += (response.get("usage") or {}).get("prompt_tokens") or 0
            calls_per_min[bucket] += 1

    total_retries = 0
    rate_limit_retries = 0
    server_busy_retries = 0
    console_path = run_dir / "console.log"
    if console_path.exists():
        for line in console_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Retrying "):
                total_retries += 1
                if RATE_LIMIT_RE.search(line):
                    rate_limit_retries += 1
                elif SERVER_BUSY_RE.search(line):
                    server_busy_retries += 1

    tok_vals = list(tokens_per_min.values())
    call_vals = list(calls_per_min.values())
    return {
        "peak_input_tokens_per_min": max(tok_vals) if tok_vals else 0,
        "median_input_tokens_per_min": int(statistics.median(tok_vals)) if tok_vals else 0,
        "peak_calls_per_min": max(call_vals) if call_vals else 0,
        "total_retries": total_retries,
        "rate_limit_retries": rate_limit_retries,
        "server_busy_retries": server_busy_retries,
    }


def _agent_budgets(setting: str, agent_config: str | None) -> dict[str, Any]:
    default_name = "solo.yaml" if setting == "solo" else "coop.yaml"
    default_path = (
        resolve_cooperbench_root() / "src/cooperbench/agents/mini_swe_agent_v2/config" / default_name
    )
    defaults = default_path.read_text(encoding="utf-8") if default_path.exists() else ""
    budgets: dict[str, Any] = {}
    for key, pattern in (("step_limit", STEP_LIMIT_RE), ("compaction_token_trigger", COMPACTION_RE)):
        match = pattern.search(agent_config or "") or pattern.search(defaults)
        budgets[key] = int(match.group(1)) if match else None
    thinking = THINKING_RE.search(agent_config or "")
    budgets["thinking_level"] = thinking.group(1).lower() if thinking else None
    return budgets


def _read_hand_written(summary_md: Path) -> dict[str, str]:
    sections: dict[str, str] = {}
    if not summary_md.exists():
        return sections
    current: str | None = None
    for line in summary_md.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line.strip() if line.strip() in HAND_WRITTEN else None
            if current:
                sections[current] = ""
            continue
        if current:
            sections[current] += line + "\n"
    return {k: v.strip() for k, v in sections.items() if v.strip() and v.strip() != HAND_WRITTEN[k]}


def _render_summary(run: dict[str, Any], pairs: list[dict[str, Any]], hand_written: dict[str, str]) -> str:
    h = run["headline"]
    d = run["delm"]
    load = run["load"]
    b = run.get("budgets") or {}
    n_agents = max((int(p["n_agents"]) for p in pairs), default=0)
    steps = f"{b['step_limit']} per agent" if b.get("step_limit") else "unknown"
    trigger = b.get("compaction_token_trigger")
    compaction = f"at {trigger // 1000}K tokens" if trigger else "unknown"
    git_note = " with shared git" if run.get("git_enabled") else ""
    lines = [
        f"# {run['run_name']}",
        "",
        "<!-- Generated by delm_cb.report. Edit only the Experiment and Observations sections. -->",
        "",
        EXPERIMENT_HEADER,
        "",
        hand_written.get(EXPERIMENT_HEADER) or HAND_WRITTEN[EXPERIMENT_HEADER],
        "",
        "## Results",
        "",
        "| | |",
        "| :--- | :--- |",
        f"| Setting | `{run.get('setting')}`{git_note} + DeLM board, {n_agents} agent{'s' if n_agents != 1 else ''} per pair |",
        f"| Model | `{run.get('model')}`, {b.get('thinking_level') or 'LiteLLM default'} thinking |",
        f"| Subset | `{run.get('subset') or 'custom'}` ({h['pairs']} pair{'s' if h['pairs'] != 1 else ''}) |",
        f"| Step limit | {steps} |",
        f"| Compaction | {compaction} |",
        f"| Passed | {h['passed']}/{h['pairs']} ({h['pass_rate']:.0%}) |",
        f"| Clean-merge passes | {h['clean_merge_passed']}/{h['pairs']} ({h['clean_merge_pass_rate']:.0%}) |",
        f"| Agents stopped at the step limit | {h['agents_at_step_limit']} |",
        f"| Merge conflicts | {h['merge_conflicts']} |",
        f"| Agent errors | {h['agents_errored']} |",
        (
            f"| DeLM board activity | {d['board_writes_total']} writes "
            f"(median {d['median_board_writes_per_pair']}/pair), "
            f"{d['notes']} notes, {d['applies']} applies ({d['apply_conflicts']} conflicts), "
            f"{d['waits']} waits, {d['board_errors']} CLI errors |"
        ),
        f"| Cost | `${h['cost_usd']:.2f}` (`${h['cost_per_pair_usd']:.2f}` per pair) |",
        f"| Wall time | {h['wall_minutes']:.0f} min |",
        "",
        "## Pairs",
        "",
        "| Pair | Passed | Clean | Feature A | Feature B | Merge | Writes | Cost |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :---: | ---: |",
    ]
    for p in pairs:
        lines.append(
            f"| `{p['repo']}/{p['task_id']}` ({p['feature_a']},{p['feature_b']}) "
            f"| {_yes_no(p['both_passed'])} | {_yes_no(p['clean_merge_passed'])} "
            f"| {_yes_no(p['feature_a_passed'])} | {_yes_no(p['feature_b_passed'])} "
            f"| {p['merge_status']} | {p['board_writes']} | `${float(p['cost_usd']):.2f}` |"
        )
    key_slot = run.get("key_slot")
    lines += [
        "",
        OBSERVATIONS_HEADER,
        "",
        hand_written.get(OBSERVATIONS_HEADER) or HAND_WRITTEN[OBSERVATIONS_HEADER],
        "",
        "## Run Details",
        "",
        "| | |",
        "| :--- | :--- |",
        f"| Agent config | `{run.get('agent_config_path') or 'unknown'}` |",
        f"| Concurrency | `-c {run.get('concurrency')}` (pairs run at once) |",
        f"| API key slot | {key_slot if key_slot is not None else 'default'} |",
        f"| Commit | `{(run.get('git_commit') or 'unknown')[:7]}` |",
        (
            f"| Peak input load | {load['peak_input_tokens_per_min']:,} tok/min "
            f"({load['peak_calls_per_min']} calls/min, median {load['median_input_tokens_per_min']:,}) |"
        ),
        (
            f"| Model retries | {load['rate_limit_retries']} rate-limit (429), "
            f"{load['server_busy_retries']} server-busy (5xx), {load['total_retries']} total |"
        ),
        f"| Raw logs | `logs/{run['run_name']}/` |",
        "",
    ]
    return "\n".join(lines)


def _ensure_delm_exported(
    run_name: str,
    *,
    logs_root: Path,
    results_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    exported = _load_exported_run(results_root / run_name)
    if exported is not None:
        return exported
    run_dir = logs_root / run_name
    if run_dir.is_dir() and any(run_dir.glob("*/*/*/*/result.json")):
        export_run(run_name, logs_dir=logs_root, results_dir=results_root)
        return _load_exported_run(results_root / run_name)
    return None


def _load_exported_run(run_results_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    run_json = _load_json(run_results_dir / "run.json")
    pairs_csv = run_results_dir / "pairs.csv"
    if run_json is None or not pairs_csv.exists():
        return None
    with pairs_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return run_json, rows


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _yes_no(value: Any) -> str:
    return "yes" if value else "no"


def _is_clean_merge_pass(row: dict[str, Any]) -> bool:
    return _is_true(row.get("both_passed")) and row.get("merge_status") in ("clean", "identical")


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["repo"]),
        int(row["task_id"]),
        int(row["feature_a"]),
        int(row["feature_b"]),
    )


def _short_baseline_label(label: str) -> str:
    clean = label.replace("`", "").strip()
    if clean.startswith("solo"):
        return "`solo`"
    if clean.startswith("coop --git"):
        if "#1" in clean:
            return "`coop` #1"
        if "#2" in clean:
            return "`coop` #2"
        return "`coop` ref"
    if "+" in clean:
        return f"`{clean.split('+', 1)[1].strip()}`"
    return f"`{clean}`"


def _format_baseline_row(
    arm: MidpointArm,
    label: str,
    run_json: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> str:
    h = run_json["headline"]
    n = max(int(h["pairs"]), 1)
    passed = int(h["passed"])
    is_solo = run_json.get("setting") == "solo"
    clean_passed = sum(_is_clean_merge_pass(p) for p in pairs)
    clean_cell = "n/a" if is_solo else f"{clean_passed}/{n} ({clean_passed / n:.0%})"
    conflicts_cell = "—" if is_solo else str(int(h["merge_conflicts"]))
    msgs = [int(float(p.get("messages_sent") or 0)) for p in pairs]
    med_msgs = int(statistics.median(msgs)) if msgs else 0
    msgs_cell = "—" if is_solo else f"{sum(msgs):,} (med {med_msgs})"
    cost = float(h["cost_usd"])
    wall = float(h["wall_minutes"])
    return (
        f"| `{arm.subset}` | `{arm.model}` (`{arm.thinking}`) | {label} "
        f"| {passed}/{n} ({passed / n:.0%}) | {clean_cell} "
        f"| {conflicts_cell} | — | {msgs_cell} | — | `${cost:.2f}` | {wall:.0f}m |"
    )


def _format_delm_row(
    arm: MidpointArm,
    delm_run: dict[str, Any],
    delm_pairs: list[dict[str, Any]],
    baseline_data: list[tuple[str, str, tuple[dict[str, Any], list[dict[str, Any]]] | None]],
) -> str:
    h = delm_run["headline"]
    d = delm_run.get("delm") or {}
    n = max(int(h["pairs"]), 1)
    passed = int(h["passed"])
    clean_passed = int(h.get("clean_merge_passed", sum(_is_clean_merge_pass(p) for p in delm_pairs)))
    conflicts = int(h["merge_conflicts"])
    writes = int(d.get("board_writes_total", 0))
    med_writes = d.get("median_board_writes_per_pair", 0)
    win_loss_parts: list[str] = []
    for label, _, base in baseline_data:
        if base is None:
            continue
        base_run, base_pairs = base
        wins, losses, clean_wins, clean_losses = _paired_delta(delm_pairs, base_pairs)
        short = _short_baseline_label(label)
        if base_run.get("setting") == "solo":
            win_loss_parts.append(f"+{wins}/-{losses} vs {short}")
        else:
            win_loss_parts.append(f"+{wins}/-{losses} (clean +{clean_wins}/-{clean_losses}) vs {short}")
    win_loss = "; ".join(win_loss_parts) if win_loss_parts else "—"
    activity = (
        f"{int(d.get('notes', 0))} / {int(d.get('applies', 0))} / "
        f"{int(d.get('apply_conflicts', 0))} / {int(d.get('waits', 0))}"
    )
    cost = float(h["cost_usd"])
    wall = float(h["wall_minutes"])
    return (
        f"| `{arm.subset}` | `{arm.model}` (`{arm.thinking}`) | **`{arm.run_name}`** "
        f"| **{passed}/{n} ({passed / n:.0%})** | **{clean_passed}/{n} ({clean_passed / n:.0%})** "
        f"| {conflicts} | {win_loss} | {writes:,} (med {med_writes}) | {activity} "
        f"| `${cost:.2f}` | {wall:.0f}m |"
    )


def _paired_delta(
    delm_pairs: list[dict[str, Any]],
    base_pairs: list[dict[str, Any]],
) -> tuple[int, int, int, int]:
    delm_map = {_pair_key(p): p for p in delm_pairs}
    base_map = {_pair_key(p): p for p in base_pairs}
    wins = losses = clean_wins = clean_losses = 0
    for key, dp in delm_map.items():
        bp = base_map.get(key)
        if bp is None:
            continue
        dp_pass = _is_true(dp.get("both_passed"))
        bp_pass = _is_true(bp.get("both_passed"))
        if dp_pass and not bp_pass:
            wins += 1
        elif bp_pass and not dp_pass:
            losses += 1
        dp_clean = _is_clean_merge_pass(dp)
        bp_clean = _is_clean_merge_pass(bp)
        if dp_clean and not bp_clean:
            clean_wins += 1
        elif bp_clean and not dp_clean:
            clean_losses += 1
    return wins, losses, clean_wins, clean_losses


def _format_paired_breakdown(
    arm: MidpointArm,
    delm_pairs: list[dict[str, Any]],
    baseline_data: list[tuple[str, str, tuple[dict[str, Any], list[dict[str, Any]]] | None]],
) -> list[str]:
    coop_baselines = [
        (label, base[1])
        for label, _, base in baseline_data
        if base is not None and base[0].get("setting") != "solo"
    ]
    if not coop_baselines:
        return []
    label, base_pairs = coop_baselines[0]
    base_map = {_pair_key(p): p for p in base_pairs}
    delm_map = {_pair_key(p): p for p in delm_pairs}
    wins: list[str] = []
    losses: list[str] = []
    for key, dp in sorted(delm_map.items()):
        bp = base_map.get(key)
        if bp is None:
            continue
        dp_clean = _is_clean_merge_pass(dp)
        bp_clean = _is_clean_merge_pass(bp)
        desc = f"`{key[0]}/{key[1]}` ({key[2]},{key[3]})"
        if dp_clean and not bp_clean:
            wins.append(desc)
        elif bp_clean and not dp_clean:
            losses.append(desc)
    return [
        "",
        f"### `{arm.run_name}` vs. {label} (Clean-Merge)",
        f"- **Wins ({len(wins)}):** {', '.join(wins) if wins else 'none'}",
        f"- **Losses ({len(losses)}):** {', '.join(losses) if losses else 'none'}",
    ]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Export DeLM-on-CooperBench results.")
    parser.add_argument("run_name", nargs="?", help="Optional run name under logs/ to export.")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--baseline-results-dir", default=None)
    args = parser.parse_args()

    if args.run_name:
        run = export_run(args.run_name, logs_dir=args.logs_dir, results_dir=args.results_dir)
        h = run["headline"]
        print(
            f"exported {h['pairs']} pairs to {Path(args.results_dir) / args.run_name}/ "
            f"(passed {h['passed']}/{h['pairs']}, clean {h['clean_merge_passed']}/{h['pairs']}, cost ${h['cost_usd']:.2f})"
        )
    out = generate_midpoint_comparison(
        logs_dir=args.logs_dir,
        results_dir=args.results_dir,
        baseline_results_dir=args.baseline_results_dir,
    )
    print(f"updated {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
