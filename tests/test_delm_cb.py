"""Unit and Docker integration tests for the zero-invasion DeLM CooperBench harness."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jinja2
import pytest

from delm_cb.adapter import DelmDefaultAgent, VENDOR_BOARD_DIR
from delm_cb.board_runtime import PairBoardRuntime
from delm_cb.cli import merge_shards, run_midpoint_suite, split_subset
from delm_cb.prompts import DELM_INSTANCE_TEMPLATE, DELM_SYSTEM_TEMPLATE
from delm_cb.report import (
    PACKAGE_ROOT,
    export_run,
    generate_midpoint_comparison,
    resolve_cooperbench_root,
)
from delm_cb.showcase import (
    compile_delm_pair_replay,
    convert_monotonic_to_epoch,
    export_cooperbench_diff_patch,
    patch_replay_html,
)

COOPERBENCH_ROOT = resolve_cooperbench_root()
COOPERBENCH_DATASET = COOPERBENCH_ROOT / "dataset"
COOPERBENCH_RESULTS = COOPERBENCH_ROOT / "results"


def test_prepopulated_board_and_read_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First read by each agent gets full view with pre-claimed items; second read is compact."""

    class _FakeOriginalFiles:
        def __init__(self, image: str, workdir: str = "/workspace/repo") -> None:
            del image, workdir

        def fetch(self, path: str) -> bytes | None:
            del path
            return None

        def remove(self) -> None:
            return None

    class _NoopHTTPServer:
        def __init__(self, addr: tuple[str, int], handler: Any) -> None:
            del addr, handler
            self.server_address = ("0.0.0.0", 45678)

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        def server_close(self) -> None:
            return None

    monkeypatch.setattr("delm_cb.board_runtime.OriginalFiles", _FakeOriginalFiles)
    monkeypatch.setattr("delm_cb.board_runtime._BoardHTTPServer", _NoopHTTPServer)

    pair_log_dir = tmp_path / "logs" / "r1" / "coop" / "openai_tiktoken_task" / "0" / "f4_f8"
    with PairBoardRuntime(
        pair_log_dir=pair_log_dir,
        repo_name="openai_tiktoken_task",
        task_id=0,
        features=[8, 4],
        dataset_dir=COOPERBENCH_DATASET,
    ) as runtime:
        assert runtime.sorted_features == [4, 8]
        assert runtime.board is not None
        assert set(runtime.item_ids) == {"agent1", "agent2"}

        first_a1 = runtime.board.read("agent1")
        assert first_a1["full"] is True
        items = {it["id"]: it for it in first_a1["queue"]}
        assert items[runtime.item_ids["agent1"]]["owner"] == "agent1"
        assert items[runtime.item_ids["agent1"]]["subject"].startswith("Feature 4:")
        assert items[runtime.item_ids["agent2"]]["owner"] == "agent2"
        assert items[runtime.item_ids["agent2"]]["subject"].startswith("Feature 8:")

        first_a2 = runtime.board.read("agent2")
        assert first_a2["full"] is True

        second_a1 = runtime.board.read("agent1")
        assert second_a1["full"] is False

        restored_a1 = runtime.board.read("agent1", full=True)
        assert restored_a1["full"] is True

    exported = json.loads((pair_log_dir / "delm_board_events.json").read_text(encoding="utf-8"))
    assert exported["summary"]["board_writes_total"] == 0


def test_prompts_render_cleanly_with_strict_undefined() -> None:
    """Jinja2 StrictUndefined renders DeLM system and instance templates with no messaging references."""
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    variables = {
        "agent_id": "agent1",
        "agents": ["agent1", "agent2"],
        "git_enabled": True,
        "messaging_enabled": False,
        "delm_item1_id": "q-1",
        "delm_item2_id": "q-2",
        "delm_f1": 4,
        "delm_f2": 8,
        "task": "Implement feature 4.",
        "system": "Linux",
        "release": "6.0",
        "version": "1",
        "machine": "x86_64",
    }
    sys_text = env.from_string(DELM_SYSTEM_TEMPLATE).render(**variables)
    inst_text = env.from_string(DELM_INSTANCE_TEMPLATE).render(**variables)
    combined = sys_text + "\n" + inst_text

    assert "send_message" not in combined
    assert "sh /tools/context.sh" in combined
    assert "gh pr create" in combined
    assert "q-1" in combined
    assert "q-2" in combined


def test_nudge_unsubmitted_triggers_without_comm() -> None:
    """DelmDefaultAgent._nudge_unsubmitted nudges when repo is dirty and no PR tag exists."""

    class _FakeModel:
        def __init__(self) -> None:
            self.cost = 0.0
            self.n_calls = 0

        def serialize(self) -> dict[str, Any]:
            return {"info": {"model_stats": {"instance_cost": 0.0, "api_calls": 0}}}

        def format_message(self, *, role: str, content: str) -> dict[str, str]:
            return {"role": role, "content": content}

    class _FakeEnv:
        def __init__(self, has_pr: bool) -> None:
            self.has_pr = has_pr

        def serialize(self) -> dict[str, Any]:
            return {}

        def execute(self, action: dict[str, str]) -> dict[str, Any]:
            cmd = action["command"]
            if "git ls-remote" in cmd:
                return {"output": "refs/tags/pr/agent1\n" if self.has_pr else ""}
            if "git status --porcelain" in cmd:
                return {"output": " M src/app.py\n"}
            if "git rev-list" in cmd:
                return {"output": "0\n"}
            return {"output": ""}

    agent = DelmDefaultAgent(
        model=_FakeModel(),
        env=_FakeEnv(has_pr=False),
        comm=None,
        agent_id="agent1",
        delm_agent_id="agent1",
        system_template="sys",
        instance_template="{{task}}",
    )
    assert agent._nudge_unsubmitted() is True
    assert len(agent.messages) == 1
    assert "git push origin HEAD:agent1" in agent.messages[0]["content"]
    assert "gh pr create" in agent.messages[0]["content"]

    agent_with_pr = DelmDefaultAgent(
        model=_FakeModel(),
        env=_FakeEnv(has_pr=True),
        comm=None,
        agent_id="agent1",
        delm_agent_id="agent1",
        system_template="sys",
        instance_template="{{task}}",
    )
    assert agent_with_pr._nudge_unsubmitted() is False


@pytest.mark.parametrize(
    ("repo_name", "task_id", "features", "existing_file"),
    [
        ("openai_tiktoken_task", 0, [4, 8], "tiktoken/core.py"),
        ("go_chi_task", 27, [3, 4], "mux.go"),
    ],
)
def test_docker_container_board_end_to_end(
    tmp_path: Path,
    repo_name: str,
    task_id: int,
    features: list[int],
    existing_file: str,
) -> None:
    """Exercise full context.sh workflow across two live Docker containers."""
    pair_log_dir = tmp_path / "logs" / "it" / "coop" / repo_name / str(task_id) / "f_pair"
    image = f"akhatua/cooperbench-{repo_name.removesuffix('_task').replace('_', '-')}:task{task_id}"
    c1_name = f"delm-test-a1-{uuid.uuid4().hex[:8]}"
    c2_name = f"delm-test-a2-{uuid.uuid4().hex[:8]}"

    with PairBoardRuntime(
        pair_log_dir=pair_log_dir,
        repo_name=repo_name,
        task_id=task_id,
        features=features,
        dataset_dir=COOPERBENCH_DATASET,
    ) as runtime:
        for actor in ("agent1", "agent2"):
            (runtime.records_dir(actor) / "prompt.txt").write_text("test prompt", encoding="utf-8")

        try:
            _start_sleep_container(
                name=c1_name,
                image=image,
                runtime=runtime,
                actor="agent1",
            )
            _start_sleep_container(
                name=c2_name,
                image=image,
                runtime=runtime,
                actor="agent2",
            )

            # 1. Read full on first call, compact on second, full on --all.
            r1 = json.loads(_docker_exec(c1_name, "sh /tools/context.sh read").stdout)
            assert r1["full"] is True
            r1_compact = json.loads(_docker_exec(c1_name, "sh /tools/context.sh read").stdout)
            assert r1_compact["full"] is False
            r1_all = json.loads(_docker_exec(c1_name, "sh /tools/context.sh read --all").stdout)
            assert r1_all["full"] is True

            # 2. Status update.
            st = json.loads(_docker_exec(c1_name, 'sh /tools/context.sh status --text "starting work"').stdout)
            assert "view" in st

            # 3. Modify existing file + create new file (verifies Docker 29 OriginalFiles for both).
            _docker_exec(c1_name, f"printf '\n// delm integration marker\n' >> /workspace/repo/{existing_file}")
            _docker_exec(c1_name, "printf 'brand new file content\n' > /workspace/repo/delm_new_file.txt")
            note_res = json.loads(
                _docker_exec(
                    c1_name,
                    (
                        "sh /tools/context.sh note --type FACT "
                        "--text 'Added marker and new helper file' "
                        "--scope integration-test --basis local-edit "
                        f"--attach {existing_file} delm_new_file.txt"
                    ),
                ).stdout
            )
            pub_id = note_res["id"]
            assert pub_id

            # 4. agent2 expands and cleanly applies agent1's publication.
            expand_out = _docker_exec(c2_name, f"sh /tools/context.sh expand --id {pub_id}").stdout
            assert "delm_new_file.txt" in expand_out
            assert existing_file in expand_out

            apply_res = json.loads(_docker_exec(c2_name, f"sh /tools/context.sh apply --id {pub_id}").stdout)
            assert apply_res["applied"] is True
            assert "delm_new_file.txt" in apply_res["written"]
            assert (
                _docker_exec(c2_name, "cat /workspace/repo/delm_new_file.txt").stdout.strip()
                == "brand new file content"
            )

            # 5. Conflicting local edit in agent2 prevents apply overwrite.
            _docker_exec(c2_name, "printf 'agent2 local edit\n' > /workspace/repo/delm_new_file.txt")
            _docker_exec(c1_name, "printf 'agent1 conflicting edit\n' > /workspace/repo/delm_new_file.txt")
            note2_res = json.loads(
                _docker_exec(
                    c1_name,
                    (
                        "sh /tools/context.sh note --type FACT "
                        "--text 'Updated new helper file' "
                        "--scope integration-test --basis local-edit "
                        "--attach delm_new_file.txt"
                    ),
                ).stdout
            )
            conflict_proc = _docker_exec(
                c2_name,
                f"sh /tools/context.sh apply --id {note2_res['id']}",
                check=False,
            )
            assert conflict_proc.returncode != 0
            conflict_json = json.loads(conflict_proc.stdout)
            assert conflict_json["applied"] is False
            assert "delm_new_file.txt" in conflict_json["conflicts"]
            assert (
                _docker_exec(c2_name, "cat /workspace/repo/delm_new_file.txt").stdout.strip()
                == "agent2 local edit"
            )

            # 6. done --attach succeeds.
            done_res = json.loads(
                _docker_exec(
                    c1_name,
                    (
                        f"sh /tools/context.sh done --item {runtime.item_ids['agent1']} "
                        "--summary 'Finished feature 1' --attach delm_new_file.txt"
                    ),
                ).stdout
            )
            assert done_res["item"] == runtime.item_ids["agent1"]

            # 7. read --wait wakes early on peer publication AND on peer session ended.
            _docker_exec(c1_name, "sh /tools/context.sh read")
            t_wait = _DelayedThread(
                delay=0.6,
                fn=lambda: _docker_exec(
                    c2_name,
                    "sh /tools/context.sh note --type FACT --text 'ready' --scope s --basis b",
                ),
            )
            t_wait.start()
            t0 = time.monotonic()
            wait_proc = _docker_exec(c1_name, "sh /tools/context.sh read --wait 5")
            elapsed_update = time.monotonic() - t0
            t_wait.join()
            assert "Update from agent2" in wait_proc.stdout
            assert elapsed_update < 4.5

            _docker_exec(c2_name, "sh /tools/context.sh read")
            t_end = _DelayedThread(delay=0.6, fn=lambda: runtime.mark_session("agent1", "ended"))
            t_end.start()
            t1 = time.monotonic()
            end_proc = _docker_exec(c2_name, "sh /tools/context.sh read --wait 5")
            elapsed_end = time.monotonic() - t1
            t_end.join()
            assert "Update from agent1" in end_proc.stdout
            assert elapsed_end < 4.5

            for actor in ("agent1", "agent2"):
                err_file = runtime.records_dir(actor) / "context-command-errors.jsonl"
                assert err_file.read_text(encoding="utf-8").strip() == ""
        finally:
            subprocess.run(["docker", "rm", "-f", c1_name, c2_name], capture_output=True)

    exported = json.loads((pair_log_dir / "delm_board_events.json").read_text(encoding="utf-8"))
    summary = exported["summary"]
    assert summary["board_writes_total"] >= 6
    assert summary["commands_by_actor"]["agent1"]["note"] == 2
    assert summary["commands_by_actor"]["agent2"]["apply"] >= 2
    assert summary["commands_by_actor"]["agent2"]["apply_conflict"] == 1
    for body in exported["bodies"]:
        for file_meta in body["files"]:
            assert "content" not in file_meta
            assert "original" not in file_meta

    leaked = subprocess.run(
        ["docker", "ps", "-a", "--filter", "label=delm.workspace.research=true", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert leaked == ""


def test_split_merge_suite_dry_run_and_report(tmp_path: Path) -> None:
    """Verify sharding split/merge, suite --dry-run, and Stage-4 comparison reporting."""
    shards_dir = tmp_path / "_shards"
    logs_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"

    shards = split_subset(subset="core", num_shards=2, dataset_dir=COOPERBENCH_DATASET, shards_dir=shards_dir)
    assert len(shards) == 2
    assert sum(s["pairs"] for s in shards) == 10

    for k in (1, 2):
        s_dir = logs_dir / f"core-ref-delm-shard{k}of2"
        pair_dir = s_dir / "coop" / "dottxt_ai_outlines_task" / str(k) / "f1_f3"
        pair_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "config.json").write_text(
            json.dumps(
                {
                    "run_name": s_dir.name,
                    "setting": "coop",
                    "agent_framework": "delm_msa_v2",
                    "model": "vertex_ai/gemini-3.8-flash",
                    "concurrency": 2,
                    "total_tasks": 1,
                    "started_at": "2026-11-12T00:00:00",
                }
            ),
            encoding="utf-8",
        )
        (s_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_name": s_dir.name,
                    "completed_at": "2026-11-12T00:10:00",
                    "total_tasks": 1,
                    "completed": 1,
                    "skipped": 0,
                    "failed": 0,
                    "total_cost": 1.25,
                    "total_time_seconds": 120.0,
                    "results": [],
                }
            ),
            encoding="utf-8",
        )
        (s_dir / "launch.json").write_text(
            json.dumps({"subset": f"core-shard{k}of2", "key_slot": k, "git_enabled": True}),
            encoding="utf-8",
        )
        (pair_dir / "result.json").write_text(
            json.dumps(
                {
                    "repo": "dottxt_ai_outlines_task" if k == 1 else "openai_tiktoken_task",
                    "task_id": 1655 if k == 1 else 0,
                    "features": [1, 3] if k == 1 else [4, 8],
                    "agents": {
                        "agent1": {"feature_id": 1 if k == 1 else 4, "status": "Submitted", "cost": 0.6, "steps": 20, "patch_lines": 10},
                        "agent2": {"feature_id": 3 if k == 1 else 8, "status": "Submitted", "cost": 0.65, "steps": 25, "patch_lines": 12},
                    },
                    "total_cost": 1.25,
                    "duration_seconds": 90.0,
                    "messages_sent": 4,
                }
            ),
            encoding="utf-8",
        )
        (pair_dir / "eval.json").write_text(
            json.dumps(
                {
                    "both_passed": True,
                    "feature1": {"passed": True},
                    "feature2": {"passed": True},
                    "merge": {"status": "clean", "strategy": "naive"},
                }
            ),
            encoding="utf-8",
        )
        (pair_dir / "delm_board_events.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "board_writes_total": 4,
                        "board_writes_by_actor": {"agent1": 2, "agent2": 2},
                        "commands_by_actor": {"agent1": {"note": 1, "status": 1}, "agent2": {"apply": 1, "done": 1}},
                    }
                }
            ),
            encoding="utf-8",
        )

    merged = merge_shards(
        shard_runs=["core-ref-delm-shard1of2", "core-ref-delm-shard2of2"],
        logs_dir=logs_dir,
    )
    assert merged.name == "core-ref-delm"
    exported = export_run("core-ref-delm", logs_dir=logs_dir, results_dir=results_dir)
    assert exported["headline"]["pairs"] == 2
    assert exported["headline"]["clean_merge_passed"] == 2
    assert exported["delm"]["board_writes_total"] == 8

    comp = generate_midpoint_comparison(
        logs_dir=logs_dir,
        results_dir=results_dir,
        baseline_results_dir=COOPERBENCH_RESULTS,
    )
    comp_text = comp.read_text(encoding="utf-8")
    assert "core-ref-delm" in comp_text
    assert "3-flash-ref-low-delm" in comp_text
    assert "not yet run" in comp_text

    plan = run_midpoint_suite(
        dry_run=True,
        shards=4,
        concurrency_per_shard=2,
        logs_dir=logs_dir,
        results_dir=results_dir,
        shards_dir=shards_dir,
    )
    assert [entry["arm"] for entry in plan] == [
        "core-ref-delm",
        "flash-ref-delm",
        "flash-ref-low-delm",
        "3-flash-ref-low-delm",
    ]
    assert all(len(entry["shards"]) == 4 for entry in plan)


def test_resolve_cooperbench_root_and_export_relativization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify COOPERBENCH_ROOT override and PACKAGE_ROOT path relativization in export_run."""
    custom_cb = tmp_path / "CustomCooperBench"
    custom_cb.mkdir()
    monkeypatch.setenv("COOPERBENCH_ROOT", str(custom_cb))
    assert resolve_cooperbench_root() == custom_cb.resolve()

    logs_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    run_dir = logs_dir / "rel-test"
    pair_dir = run_dir / "coop" / "openai_tiktoken_task" / "0" / "f4_f8"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"run_name": "rel-test", "setting": "coop", "model": "vertex_ai/gemini-3.8-flash"}),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"total_time_seconds": 60.0}),
        encoding="utf-8",
    )
    (run_dir / "launch.json").write_text(
        json.dumps(
            {
                "subset": "core",
                "git_enabled": True,
                "command": (
                    f"{PACKAGE_ROOT}/src/delm_cb/cli.py run -n rel-test "
                    f"--agent-config {PACKAGE_ROOT}/configs/reference_medium.yaml"
                ),
                "agent_config_path": f"{PACKAGE_ROOT}/configs/reference_medium.yaml",
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "result.json").write_text(
        json.dumps(
            {
                "repo": "openai_tiktoken_task",
                "task_id": 0,
                "features": [4, 8],
                "agents": {
                    "agent1": {"feature_id": 4, "status": "Submitted", "cost": 0.5, "steps": 10, "patch_lines": 5},
                    "agent2": {"feature_id": 8, "status": "Submitted", "cost": 0.5, "steps": 10, "patch_lines": 5},
                },
                "total_cost": 1.0,
                "duration_seconds": 30.0,
                "messages_sent": 2,
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "eval.json").write_text(
        json.dumps(
            {
                "both_passed": True,
                "feature1": {"passed": True},
                "feature2": {"passed": True},
                "merge": {"status": "clean", "strategy": "naive"},
            }
        ),
        encoding="utf-8",
    )

    exported = export_run("rel-test", logs_dir=logs_dir, results_dir=results_dir)
    assert exported["agent_config_path"] == "configs/reference_medium.yaml"
    assert (
        exported["command"]
        == "src/delm_cb/cli.py run -n rel-test --agent-config configs/reference_medium.yaml"
    )
    assert str(PACKAGE_ROOT) not in (results_dir / "rel-test" / "summary.md").read_text(encoding="utf-8")


def test_showcase_alignment_collapsing_delivery_and_html(tmp_path: Path) -> None:
    """Verify monotonic-to-epoch conversion, wait collapsing, peer read badges, and HTML patching."""
    base_wall_ns = 1_794_478_700_000_000_000
    base_mono_ns = 1_000_000_000_000
    events = [
        {
            "sequence": 1,
            "time_ns": base_mono_ns,
            "actor": "system",
            "kind": "item",
            "payload": {"id": "id1", "subject": "Feature 4: Alpha", "detail": "Spec 4"},
        },
        {
            "sequence": 2,
            "time_ns": base_mono_ns + 100_000_000,
            "actor": "system",
            "kind": "item",
            "payload": {"id": "id2", "subject": "Feature 8: Beta", "detail": "Spec 8"},
        },
        {"sequence": 3, "time_ns": base_mono_ns + 200_000_000, "actor": "agent1", "kind": "claim", "payload": {"item": "id1"}},
        {"sequence": 4, "time_ns": base_mono_ns + 300_000_000, "actor": "agent1", "kind": "status", "payload": {"text": "Feature 4: Alpha", "auto": True}},
        {"sequence": 5, "time_ns": base_mono_ns + 400_000_000, "actor": "agent2", "kind": "claim", "payload": {"item": "id2"}},
        {"sequence": 6, "time_ns": base_mono_ns + 500_000_000, "actor": "agent2", "kind": "status", "payload": {"text": "Feature 8: Beta", "auto": True}},
        {
            "sequence": 7,
            "time_ns": base_mono_ns + 1_000_000_000,
            "actor": "system",
            "kind": "session",
            "payload": {"actor": "agent1", "state": "running", "recorded_at_ns": base_wall_ns + 1_000_000_000},
        },
        {
            "sequence": 8,
            "time_ns": base_mono_ns + 5_000_000_000,
            "actor": "agent1",
            "kind": "note",
            "payload": {"id": "n1", "type": "FACT", "text": "Shared signature", "scope": "core.py", "basis": "inspected"},
        },
        {
            "sequence": 9,
            "time_ns": base_mono_ns + 10_000_000_000,
            "actor": "agent2",
            "kind": "wait",
            "payload": {"requested": 25.0, "waited": 25.0, "woke_by": "timeout", "woken": []},
        },
        {
            "sequence": 10,
            "time_ns": base_mono_ns + 35_000_000_000,
            "actor": "agent2",
            "kind": "wait",
            "payload": {"requested": 25.0, "waited": 25.0, "woke_by": "timeout", "woken": []},
        },
        {
            "sequence": 11,
            "time_ns": base_mono_ns + 40_000_000_000,
            "actor": "agent2",
            "kind": "read",
            "payload": {"via": "read"},
        },
        {
            "sequence": 12,
            "time_ns": base_mono_ns + 42_000_000_000,
            "actor": "agent2",
            "kind": "wait",
            "payload": {"requested": 25.0, "waited": 0.2, "woke_by": "update", "woken": [{"actor": "agent1", "kind": "note"}]},
        },
        {
            "sequence": 13,
            "time_ns": base_mono_ns + 45_000_000_000,
            "actor": "agent1",
            "kind": "done",
            "payload": {"item": "id1", "summary": "Completed Feature 4"},
        },
    ]

    seq_map = convert_monotonic_to_epoch(events, [], [])
    assert seq_map[8] == pytest.approx((base_wall_ns + 5_000_000_000) / 1e9)

    step_like = [SimpleNamespace(timestamp=100.0), SimpleNamespace(timestamp=200.0)]
    fallback_map = convert_monotonic_to_epoch(
        [{"sequence": 1, "time_ns": 10, "kind": "note"}, {"sequence": 2, "time_ns": 20, "kind": "done"}],
        step_like,
        [],
    )
    assert fallback_map == {1: pytest.approx(100.0), 2: pytest.approx(200.0)}

    run_dir = tmp_path / "logs" / "core-ref-delm"
    pair_dir = run_dir / "coop" / "openai_tiktoken_task" / "0" / "f4_f8"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "core-ref-delm").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"setting": "coop", "model": "vertex_ai/gemini-3.8-flash"}), encoding="utf-8")
    (run_dir / "launch.json").write_text(json.dumps({"setting": "coop", "git_enabled": True}), encoding="utf-8")
    (pair_dir / "result.json").write_text(json.dumps({"repo": "openai_tiktoken_task", "task_id": 0, "features": [4, 8]}), encoding="utf-8")
    (pair_dir / "eval.json").write_text(json.dumps({"both_passed": True, "merge": {"status": "clean"}}), encoding="utf-8")
    (pair_dir / "delm_board_events.json").write_text(
        json.dumps({"preseed_max_sequence": 6, "events": events, "bodies": []}),
        encoding="utf-8",
    )

    t0 = base_wall_ns / 1e9
    (pair_dir / "agent1_traj.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "extra": {
                            "timestamp": t0 + 4.0,
                            "actions": [{"command": "sh /tools/context.sh note --type FACT"}],
                            "response": {"usage": {}},
                        },
                    },
                    {"role": "tool", "content": "ok", "extra": {"returncode": 0}},
                    {
                        "role": "assistant",
                        "content": "",
                        "extra": {
                            "timestamp": t0 + 44.0,
                            "actions": [{"command": "sh /tools/context.sh done --item id1"}],
                            "response": {"usage": {}},
                        },
                    },
                    {"role": "tool", "content": "done", "extra": {"returncode": 0}},
                ]
            }
        ),
        encoding="utf-8",
    )
    (pair_dir / "agent2_traj.json").write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "extra": {
                            "timestamp": t0 + 39.0,
                            "actions": [{"command": "sh /tools/context.sh read"}],
                            "response": {"usage": {}},
                        },
                    },
                    {"role": "tool", "content": "ok", "extra": {"returncode": 0}},
                ]
            }
        ),
        encoding="utf-8",
    )

    replay = compile_delm_pair_replay(
        "core-ref-delm",
        "openai_tiktoken_task",
        0,
        (4, 8),
        logs_dir=tmp_path / "logs",
        results_dir=tmp_path / "results",
        dataset_dir=COOPERBENCH_DATASET,
    )
    assert replay["config_flags"]["delm_board"] is True
    assert replay["steps"]["agent1"][0]["action_phase"] == "message"
    assert replay["waits"] == 3
    assert [m["intent"] for m in replay["messages"]] == ["FACT", "DONE"]
    assert all(not m["wait"] for m in replay["messages"])
    assert len(replay["wait_events"]) == 2
    assert replay["wait_events"][0]["count"] == 2
    assert replay["wait_events"][0]["waited_s"] == pytest.approx(50.0)
    assert replay["wait_events"][0]["woke_by"] == "timeout"
    assert replay["wait_events"][1]["woke_by"] == "update"
    assert replay["wait_events"][1]["waited_s"] == pytest.approx(0.2)
    assert replay["messages"][0]["delivered_step"] == 1
    assert replay["messages"][1]["delivery_status"] == "undelivered_peer_exited"

    assert len(replay["board_snapshots"]) >= 3
    last_snap = replay["board_snapshots"][-1]
    assert last_snap["queue_items"][0]["label"] == "Item 1"
    assert last_snap["queue_items"][0]["state"] == "done"
    assert last_snap["queue_items"][0]["summary"] == "Completed Feature 4"
    assert last_snap["queue_items"][0]["detail"] == "Spec 4"
    raw_view = json.loads(last_snap["raw_text"])
    assert raw_view["reader"] == "viewer"
    assert raw_view["full"] is True
    assert len(raw_view["notes"]) == 1

    assert replay["board_snapshots"][0]["sessions"][0]["state"] == "starting"
    assert replay["board_snapshots"][1]["sessions"][0]["state"] == "running"

    first_frame = replay["timeline"][0]
    assert [t["status"] for t in first_frame["center"]["threads"]] == ["open", "open"]
    assert first_frame["board_snapshot_idx"] == 1
    assert any(f["kind"] == "board_wait" for f in replay["timeline"])
    last_frame = replay["timeline"][-1]
    assert last_frame["board_snapshot_idx"] == len(replay["board_snapshots"]) - 1
    assert [t["status"] for t in last_frame["center"]["threads"]] == ["resolved", "open"]
    badges = {m["id"]: m["live_badge"] for m in last_frame["center"]["messages"]}
    assert badges["b1"] == "👁 Read by agent2 @ Step 1"
    assert badges["b2"] == "⏭ Unread before agent2 exited"

    template_html = (COOPERBENCH_ROOT / "tools" / "replay_viewer" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    patched = patch_replay_html(template_html)
    assert "<title>DeLM × CooperBench — Replay Studio</title>" in patched
    assert "DeLM Task Queue & Claims" in patched
    assert ".intent-FACT" in patched
    assert "centerTabBar" in patched
    assert "Shared Context (" in patched
    assert "Task Queue (" in patched
    assert "Raw Board View" in patched
    assert "Posted" in patched
    assert "jumpWaitPollPill" in patched


def test_export_cooperbench_diff_patch(tmp_path: Path) -> None:
    """Verify CooperBench-to-DeLM unified diff patch covers prompts, adapter, and UI."""
    patch_path = export_cooperbench_diff_patch(
        tmp_path / "cooperbench_to_delm.patch",
        cooperbench_root=COOPERBENCH_ROOT,
    )
    patch_text = patch_path.read_text(encoding="utf-8")

    expected_headers = (
        "--- a/cooperbench/prompts/coop_git_rendered.md",
        "+++ b/delm_cb/prompts/delm_coop_git_rendered.md",
        "--- a/cooperbench/src/cooperbench/agents/mini_swe_agent_v2/adapter.py",
        "+++ b/src/delm_cb/adapter.py",
        "--- a/cooperbench/tools/replay_viewer/static/index.html",
        "+++ b/demos/delm_showcase_template.html",
    )
    for header in expected_headers:
        assert header in patch_text

    assert "vendor/delm/board/ is a 0-diff verbatim copy of yuzhenmao/DeLM @ 6661761" in patch_text
    assert "## Shared Context Board (DeLM)" in patch_text
    assert '@register("delm_msa_v2")' in patch_text
    assert "centerTabBar" in patch_text


def _start_sleep_container(
    *,
    name: str,
    image: str,
    runtime: PairBoardRuntime,
    actor: str,
) -> None:
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-w",
        "/workspace/repo",
        "--entrypoint",
        "/bin/bash",
        "--network",
        "cooperbench",
        "--add-host=host.docker.internal:host-gateway",
        "-v",
        f"{VENDOR_BOARD_DIR}:/tools:ro",
        "-v",
        f"{runtime.records_dir(actor)}:/records:rw",
        "-e",
        f"CONTEXT_URL={runtime.context_url}",
        "-e",
        f"CONTEXT_TOKEN={runtime.token_for(actor)}",
        "-e",
        "CONTEXT_PYTHON=python3",
        "-e",
        "CONTEXT_LIBRARY_PATH=",
        "-e",
        "TASK_WORKSPACE=/workspace/repo",
        image,
        "-c",
        "sleep 300",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _docker_exec(name: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "exec", "-w", "/workspace/repo", name, "sh", "-c", command],
        check=check,
        capture_output=True,
        text=True,
    )


class _DelayedThread(threading.Thread):
    def __init__(self, *, delay: float, fn: Any) -> None:
        super().__init__(daemon=True)
        self._delay = delay
        self._fn = fn

    def run(self) -> None:
        time.sleep(self._delay)
        self._fn()
