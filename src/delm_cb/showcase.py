"""Self-contained 3-pane HTML trajectory replay showcase compiler for DeLM and CooperBench."""

from __future__ import annotations

import copy
import csv
import difflib
import getpass
import importlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jinja2
import yaml

from delm_cb.prompts import DELM_INSTANCE_TEMPLATE, DELM_SYSTEM_TEMPLATE
from delm_cb.report import (
    DEFAULT_LOGS_DIR,
    DEFAULT_RESULTS_DIR,
    PACKAGE_ROOT,
    relativize_package_paths,
    resolve_cooperbench_root,
)

DEFAULT_SHOWCASE_HTML = PACKAGE_ROOT / "demos" / "delm_showcase.html"
DEFAULT_PROMPT_PATCH = PACKAGE_ROOT / "demos" / "prompt.patch"
DEFAULT_ADAPTER_PATCH = PACKAGE_ROOT / "demos" / "adapter.patch"
MAX_SHOWCASE_OUTPUT_CHARS = 4_000
MAX_SHOWCASE_THOUGHT_CHARS = 3_000

_PROMPT_PATCH_HEADER = "\n".join(
    [
        "# Rendered coop --git prompt diff: CooperBench coop.yaml vs. delm_cb.prompts.",
        "# Rendered with Jinja2 StrictUndefined under identical workspace settings (messaging_enabled=True vs. False).",
    ]
)
_ADAPTER_PATCH_HEADER = "\n".join(
    [
        "# Agent runner diff: CooperBench mini_swe_agent_v2/adapter.py vs. src/delm_cb/adapter.py.",
        "# vendor/delm/board/ is a 0-diff verbatim copy of yuzhenmao/DeLM @ 6661761 board/.",
    ]
)

_BOARD_VIEW_KINDS = frozenset({"read", "status", "note", "done", "claim", "reserve", "publish", "apply"})
_BOARD_STATE_KINDS = frozenset({"session", "status", "item", "claim", "reserve", "done", "note", "body"})
_ENV_SECRET_LINE_RE = re.compile(r"^\s*(?:export\s+)?(?:VERTEXAI_PROJECT|GEMINI_API_KEY)\w*\s*=\s*([^\s#]+)")

_SHOWCASE_CSS_ADDITIONS = """\
    .intent-FACT { background-color: #39c5cf2a; color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
    .intent-FAIL { background-color: #f851492a; color: var(--accent-red); border: 1px solid var(--accent-red); }
    .intent-STATUS { background-color: #8b949e2a; color: var(--text-muted); border: 1px solid var(--border-color); }
    .intent-DONE { background-color: #3fb9502a; color: var(--accent-green); border: 1px solid var(--accent-green); }
    .intent-PROPOSAL { background-color: #bc8cff2a; color: var(--accent-purple); border: 1px solid var(--accent-purple); }
    .intent-APPLY { background-color: #bc8cff2a; color: var(--accent-purple); border: 1px solid var(--accent-purple); }
    .intent-WAIT { background-color: #d299222a; color: var(--accent-amber); border: 1px solid var(--accent-amber); }
    .divider-content.wait { border-color: var(--accent-amber); color: var(--accent-amber); background-color: #d2992218; }
    .divider-content code { font-family: var(--font-mono); font-size: 10.5px; background: var(--bg-base); padding: 1px 4px; border-radius: 3px; }
    .session-strip { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; flex-shrink: 0; }
    .session-card { background-color: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 6px; padding: 8px 10px; display: flex; flex-direction: column; gap: 4px; }
    .session-card-head { display: flex; align-items: center; justify-content: space-between; font-size: 11px; }
    .session-state-pill { font-family: var(--font-mono); font-size: 9.5px; font-weight: 600; text-transform: uppercase; padding: 1px 6px; border-radius: 8px; border: 1px solid var(--border-color); }
    .session-state-running { color: var(--accent-green); border-color: var(--accent-green); background: #3fb95022; }
    .session-state-ended { color: var(--text-muted); border-color: var(--border-color); background: #8b949e22; }
    .session-state-starting { color: var(--accent-amber); border-color: var(--accent-amber); background: #d2992222; }
    .session-status-text { font-size: 11.5px; color: var(--text-main); white-space: pre-wrap; word-break: break-word; }
    .queue-card { background-color: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }
    .queue-card-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 11px; }
    .queue-card-subject { font-weight: 600; font-size: 12.5px; color: var(--text-heading); }
    .queue-detail-toggle { background-color: var(--bg-base); border: 1px solid var(--border-color); border-radius: 4px; padding: 6px 8px; font-size: 11px; }
    .queue-detail-toggle summary { cursor: pointer; font-weight: 600; color: var(--text-muted); user-select: none; }
    .queue-detail-body { margin-top: 6px; white-space: pre-wrap; word-break: break-word; font-family: var(--font-mono); font-size: 11px; color: var(--text-main); max-height: 220px; overflow-y: auto; }
    .done-summary-box { border: 1px solid var(--accent-green); background-color: #3fb95014; border-radius: 6px; padding: 8px; display: flex; flex-direction: column; gap: 4px; font-size: 11.5px; }
    .done-summary-title { font-weight: 700; font-size: 10.5px; text-transform: uppercase; color: var(--accent-green); }"""


@dataclass(frozen=True)
class ShowcasePairSpec:
    """One task pair and its matched DeLM + CooperBench runs in the showcase."""

    repo: str
    task_id: int
    features: tuple[int, int]
    delm_runs: tuple[str, ...]
    cooperbench_runs: tuple[str, ...]


SHOWCASE_PAIRS: tuple[ShowcasePairSpec, ...] = (
    ShowcasePairSpec(
        repo="llama_index_task",
        task_id=17244,
        features=(5, 6),
        delm_runs=("core-ref-delm",),
        cooperbench_runs=(
            "coop-msa_v2-git-gemini-3-8-flash-core-ref-intents",
            "coop-msa_v2-git-gemini-3-8-flash-core-ref-delivery-timing",
            "coop-msa_v2-git-gemini-3-8-flash-core-ref-comm-combined",
        ),
    ),
    ShowcasePairSpec(
        repo="pallets_click_task",
        task_id=2068,
        features=(6, 7),
        delm_runs=("flash-ref-delm",),
        cooperbench_runs=(
            "coop-msa_v2-git-gemini-3-8-flash-flash-ref-intents",
            "coop-msa_v2-git-gemini-3-8-flash-flash-ref-delivery-timing",
            "coop-msa_v2-git-gemini-3-8-flash-flash-ref-comm-combined",
        ),
    ),
    ShowcasePairSpec(
        repo="pallets_click_task",
        task_id=2068,
        features=(1, 6),
        delm_runs=("flash-ref-delm", "flash-ref-low-delm"),
        cooperbench_runs=(
            "coop-msa_v2-git-gemini-3-8-flash-flash-ref-intents",
            "coop-msa_v2-git-gemini-3-8-flash-flash-ref-low-intents",
        ),
    ),
    ShowcasePairSpec(
        repo="openai_tiktoken_task",
        task_id=0,
        features=(6, 8),
        delm_runs=("3-flash-ref-low-delm",),
        cooperbench_runs=(
            "coop-msa_v2-git-gemini-3-flash-flash-ref-low",
            "coop-msa_v2-git-gemini-3-flash-flash-ref-low-delivery-timing",
        ),
    ),
)


def compile_delm_pair_replay(
    run_name: str,
    repo: str,
    task_id: int | str,
    features: list[int] | tuple[int, int] | str,
    *,
    logs_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    dataset_dir: Path | str | None = None,
    compiler: Any | None = None,
) -> dict[str, Any]:
    """Compile a 3-pane replay payload for one DeLM pair run."""
    comp = compiler or _load_cooperbench_compiler()
    cb_root = resolve_cooperbench_root()
    ld = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
    rd = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    dd = Path(dataset_dir) if dataset_dir is not None else cb_root / "dataset"

    fa, fb = comp._parse_features_arg(features)
    tid = int(task_id)
    pair_dir = comp._find_pair_log_dir(ld / run_name, repo, tid, fa, fb)
    if not pair_dir.exists():
        raise FileNotFoundError(f"DeLM pair directory not found: {pair_dir}")

    result_data = _load_json(pair_dir / "result.json") or {}
    eval_data = _load_json(pair_dir / "eval.json") or {}
    board_data = _load_json(pair_dir / "delm_board_events.json") or {}
    run_dir = ld / run_name
    run_json = _load_json(rd / run_name / "run.json") or {}
    launch = _load_json(run_dir / "launch.json") or {}
    config = _load_json(run_dir / "config.json") or {}

    setting = str(config.get("setting") or launch.get("setting") or run_json.get("setting") or "coop")
    model = comp._extract_model(run_name, run_json, config, launch)
    budgets = comp._extract_budgets(setting, run_json if run_json.get("budgets") else launch)
    config_flags = comp._extract_config_flags(run_json or launch, run_dir)
    config_flags["delm_board"] = True

    raw1 = _load_json(comp._resolve_traj_file(pair_dir, "agent1", fa)) or {}
    raw2 = _load_json(comp._resolve_traj_file(pair_dir, "agent2", fb)) or {}
    turns1, compactions1 = comp._stitch_trajectory(raw1, "agent1")
    turns2, compactions2 = comp._stitch_trajectory(raw2, "agent2")
    raw_steps1, nudges1 = comp._extract_agent_steps(turns1, "agent1")
    raw_steps2, nudges2 = comp._extract_agent_steps(turns2, "agent2")

    steps1 = [_reclassify_delm_step(s) for s in raw_steps1]
    steps2 = [_reclassify_delm_step(s) for s in raw_steps2]

    messages, wait_events, queue_events, board_snapshots, snapshot_times, board_waits = _compile_delm_board(
        board_data=board_data,
        steps1=steps1,
        steps2=steps2,
        compiler=comp,
    )
    timeline = _build_delm_timeline(
        steps1=steps1,
        steps2=steps2,
        messages=messages,
        wait_events=wait_events,
        queue_events=queue_events,
        snapshot_times=snapshot_times,
        compactions1=compactions1,
        compactions2=compactions2,
        nudges1=nudges1,
        nudges2=nudges2,
    )

    spec_a = comp._load_feature_spec(dd, repo, tid, fa)
    spec_b = comp._load_feature_spec(dd, repo, tid, fb)
    patch_a = comp._load_patch(pair_dir, "agent1", fa)
    patch_b = comp._load_patch(pair_dir, "agent2", fb)

    return {
        "run_name": run_name,
        "setting": setting,
        "git_enabled": True,
        "model": model,
        "budgets": budgets,
        "repo": repo,
        "task_id": tid,
        "features": [fa, fb],
        "config_flags": config_flags,
        "waits": board_waits,
        "result": result_data,
        "eval": eval_data,
        "specs": {"feature_a": spec_a, "feature_b": spec_b},
        "patches": {"agent1": patch_a, "agent2": patch_b},
        "messages": [asdict(m) for m in messages],
        "wait_events": wait_events,
        "board_snapshots": board_snapshots,
        "compactions": {
            "agent1": [asdict(c) for c in compactions1],
            "agent2": [asdict(c) for c in compactions2],
        },
        "nudges": {
            "agent1": [asdict(n) for n in nudges1],
            "agent2": [asdict(n) for n in nudges2],
        },
        "steps": {
            "agent1": [comp._serialize_step(s) for s in steps1],
            "agent2": [comp._serialize_step(s) for s in steps2],
        },
        "timeline": timeline,
    }


def export_delm_showcase(
    output_path: Path | str = DEFAULT_SHOWCASE_HTML,
    *,
    pair_specs: tuple[ShowcasePairSpec, ...] = SHOWCASE_PAIRS,
    delm_logs_dir: Path | str | None = None,
    delm_results_dir: Path | str | None = None,
    cooperbench_root: Path | str | None = None,
    max_output_chars: int = MAX_SHOWCASE_OUTPUT_CHARS,
    max_thought_chars: int = MAX_SHOWCASE_THOUGHT_CHARS,
) -> Path:
    """Compile and export the self-contained 15-replay DeLM vs. CooperBench HTML showcase."""
    cb_root = Path(cooperbench_root) if cooperbench_root is not None else resolve_cooperbench_root()
    compiler = _load_cooperbench_compiler(cb_root)
    d_logs = Path(delm_logs_dir) if delm_logs_dir is not None else DEFAULT_LOGS_DIR
    d_results = Path(delm_results_dir) if delm_results_dir is not None else DEFAULT_RESULTS_DIR
    cb_logs = cb_root / "logs"
    cb_results = cb_root / "results"
    cb_dataset = cb_root / "dataset"

    delm_pairs_cache = _load_pairs_lookup(
        pair_specs, logs_dir=d_logs, results_dir=d_results, compiler=compiler, is_delm=True
    )
    cb_pairs_cache = _load_pairs_lookup(
        pair_specs, logs_dir=cb_logs, results_dir=cb_results, compiler=compiler, is_delm=False
    )
    delm_runs_meta = _load_runs_meta(d_logs, d_results, compiler=compiler, is_delm=True)
    cb_runs_meta = _load_runs_meta(cb_logs, cb_results, compiler=compiler, is_delm=False)

    pairs_index: list[dict[str, Any]] = []
    replays: dict[str, Any] = {}
    used_runs: dict[str, dict[str, Any]] = {}

    for spec in pair_specs:
        fa, fb = spec.features
        pair_runs: dict[str, Any] = {}

        for run_name in spec.delm_runs:
            replay = compile_delm_pair_replay(
                run_name=run_name,
                repo=spec.repo,
                task_id=spec.task_id,
                features=(fa, fb),
                logs_dir=d_logs,
                results_dir=d_results,
                dataset_dir=cb_dataset,
                compiler=compiler,
            )
            key = f"{run_name}:{spec.repo}:{spec.task_id}:f{fa}_f{fb}"
            replays[key] = _compact_replay(
                replay, max_output_chars=max_output_chars, max_thought_chars=max_thought_chars
            )
            pair_runs[run_name] = _build_pair_run_summary(
                run_name=run_name,
                replay=replay,
                pair_row=delm_pairs_cache.get((run_name, spec.repo, spec.task_id, fa, fb)),
            )
            if run_name in delm_runs_meta:
                used_runs[run_name] = delm_runs_meta[run_name]

        for run_name in spec.cooperbench_runs:
            replay = compiler.compile_pair_replay(
                run_name=run_name,
                repo=spec.repo,
                task_id=spec.task_id,
                features=(fa, fb),
                logs_dir=cb_logs,
                results_dir=cb_results,
                dataset_dir=cb_dataset,
            )
            key = f"{run_name}:{spec.repo}:{spec.task_id}:f{fa}_f{fb}"
            replays[key] = _compact_replay(
                replay, max_output_chars=max_output_chars, max_thought_chars=max_thought_chars
            )
            pair_runs[run_name] = _build_pair_run_summary(
                run_name=run_name,
                replay=replay,
                pair_row=cb_pairs_cache.get((run_name, spec.repo, spec.task_id, fa, fb)),
            )
            if run_name in cb_runs_meta:
                used_runs[run_name] = cb_runs_meta[run_name]

        pairs_index.append(
            {
                "repo": spec.repo,
                "task_id": spec.task_id,
                "feature_a": fa,
                "feature_b": fb,
                "pair_id": f"{spec.repo}:{spec.task_id}:f{fa}_f{fb}",
                "runs": pair_runs,
            }
        )

    secret_tokens = _collect_sensitive_tokens(cb_root)
    bundle = _sanitize_payload(
        {
            "runs": list(used_runs.values()),
            "pairs": pairs_index,
            "replays": replays,
        },
        cb_root=cb_root,
        secret_tokens=secret_tokens,
    )

    template_path = cb_root / "tools" / "replay_viewer" / "static" / "index.html"
    patched_html = patch_replay_html(template_path.read_text(encoding="utf-8"))
    bundle_json = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    safe_bundle_json = bundle_json.replace("</script>", "<\\/script>").replace("<!--", "<\\!--")
    script_tag = f"<script>window.__PRELOADED_REPLAY_BUNDLE__ = {safe_bundle_json};</script>"
    bundled_html = patched_html.replace("</head>", f"  {script_tag}\n</head>")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundled_html, encoding="utf-8")
    return out_path


def export_cooperbench_diff_patches(
    *,
    prompt_patch_path: Path | str = DEFAULT_PROMPT_PATCH,
    adapter_patch_path: Path | str = DEFAULT_ADAPTER_PATCH,
    cooperbench_root: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write separate unified diffs for rendered prompts and the agent runner adapter against CooperBench."""
    cb_root = Path(cooperbench_root) if cooperbench_root is not None else resolve_cooperbench_root()

    coop_yaml_path = cb_root / "src" / "cooperbench" / "agents" / "mini_swe_agent_v2" / "config" / "coop.yaml"
    coop_cfg = yaml.safe_load(coop_yaml_path.read_text(encoding="utf-8")) or {}
    agent_cfg = coop_cfg.get("agent") or {}

    base_vars: dict[str, Any] = {
        "task": "[... task description omitted ...]",
        "agent_id": "agent1",
        "agents": ["agent1", "agent2"],
        "git_enabled": True,
        "system": "Linux",
        "release": "6.x",
        "version": "#1 SMP",
        "machine": "x86_64",
        "delm_item1_id": "<item1_id>",
        "delm_item2_id": "<item2_id>",
        "delm_f1": 1,
        "delm_f2": 2,
    }
    cb_prompt = _render_prompt_markdown(
        str(agent_cfg["system_template"]),
        str(agent_cfg["instance_template"]),
        {**base_vars, "messaging_enabled": True},
    )
    delm_prompt = _render_prompt_markdown(
        DELM_SYSTEM_TEMPLATE,
        DELM_INSTANCE_TEMPLATE,
        {**base_vars, "messaging_enabled": False},
    )

    cb_adapter = (cb_root / "src" / "cooperbench" / "agents" / "mini_swe_agent_v2" / "adapter.py").read_text(
        encoding="utf-8"
    )
    delm_adapter = (PACKAGE_ROOT / "src" / "delm_cb" / "adapter.py").read_text(encoding="utf-8")

    prompt_diff = _unified_diff_section(
        cb_prompt,
        delm_prompt,
        fromfile="a/cooperbench/prompts/coop_git_rendered.md",
        tofile="b/delm_cb/prompts/delm_coop_git_rendered.md",
    )
    adapter_diff = _unified_diff_section(
        cb_adapter,
        delm_adapter,
        fromfile="a/cooperbench/src/cooperbench/agents/mini_swe_agent_v2/adapter.py",
        tofile="b/src/delm_cb/adapter.py",
    )

    secret_tokens = _collect_sensitive_tokens(cb_root)
    prompt_out = _write_sanitized_patch(
        Path(prompt_patch_path),
        header=_PROMPT_PATCH_HEADER,
        diff_body=prompt_diff,
        cb_root=cb_root,
        secret_tokens=secret_tokens,
    )
    adapter_out = _write_sanitized_patch(
        Path(adapter_patch_path),
        header=_ADAPTER_PATCH_HEADER,
        diff_body=adapter_diff,
        cb_root=cb_root,
        secret_tokens=secret_tokens,
    )
    return prompt_out, adapter_out


def patch_replay_html(template_html: str) -> str:
    """Apply DeLM Replay Studio CSS, center tabs, and wait-divider JS enhancements."""
    replacements = (
        (
            "<title>CooperBench Studio — Multi-Agent Replay Viewer</title>",
            "<title>DeLM × CooperBench — Replay Studio</title>",
        ),
        (
            '<span class="brand-badge">CooperBench</span>\n        <span>Replay Studio</span>',
            '<span class="brand-badge">DeLM × CooperBench</span>\n        <span>Replay Studio</span>',
        ),
        (
            ".intent-FYI { background-color: #8b949e2a; color: var(--text-muted); border: 1px solid var(--border-color); }",
            ".intent-FYI { background-color: #8b949e2a; color: var(--text-muted); border: 1px solid var(--border-color); }\n"
            + _SHOWCASE_CSS_ADDITIONS,
        ),
        (
            """    <!-- CENTER PANE: Coordination, Thread Board & Chat Stream -->
    <section class="pane" id="paneCenter">
      <!-- Sticky Active Thread Board -->
      <div class="threads-sticky">
        <div class="threads-header">
          <span>Active Thread Board & Obligations</span>
          <span id="openThreadsCountBadge" class="tag highlight">0 Open</span>
        </div>
        <div id="threadsListContainer" style="display:flex; flex-direction:column; gap:4px;">
          <div style="color:var(--text-muted); font-size:11px; font-style:italic;">No active conversation threads.</div>
        </div>
      </div>

      <!-- Chat Stream -->
      <div class="chat-stream" id="chatStream">
        <div style="color:var(--text-muted); text-align:center; padding:20px; font-style:italic;">
          Scrub the timeline below to watch coordination unfold.
        </div>
      </div>
    </section>""",
            """    <!-- CENTER PANE: Coordination, Thread Board & Chat Stream -->
    <section class="pane" id="paneCenter">
      <div class="tab-bar" id="centerTabBar" style="display:none;">
        <button class="tab-btn active" data-center-tab="stream" onclick="switchCenterTab('stream')">Shared Context (<span id="centerWriteCount">0</span> writes)</button>
        <button class="tab-btn" data-center-tab="queue" onclick="switchCenterTab('queue')">Task Queue (<span id="centerQueueCount">0</span> items)</button>
        <button class="tab-btn" data-center-tab="raw" onclick="switchCenterTab('raw')">Raw Board View</button>
      </div>

      <!-- Sticky Active Thread Board -->
      <div class="threads-sticky" id="threadsStickyBox">
        <div class="threads-header">
          <span id="threadsHeaderTitle">Active Thread Board & Obligations</span>
          <span id="openThreadsCountBadge" class="tag highlight">0 Open</span>
        </div>
        <div id="threadsListContainer" style="display:flex; flex-direction:column; gap:4px;">
          <div style="color:var(--text-muted); font-size:11px; font-style:italic;">No active conversation threads.</div>
        </div>
      </div>

      <!-- Chat Stream -->
      <div class="chat-stream" id="chatStream">
        <div style="color:var(--text-muted); text-align:center; padding:20px; font-style:italic;">
          Scrub the timeline below to watch coordination unfold.
        </div>
      </div>

      <!-- DeLM Full Queue Tab -->
      <div class="pane-body" id="centerQueueView" style="display:none;"></div>

      <!-- DeLM Raw Board View Tab -->
      <div class="pane-body" id="centerRawView" style="display:none;">
        <pre id="centerRawPre" class="step-out" style="max-height:100%; height:100%; margin:0;"></pre>
      </div>
    </section>""",
        ),
        (
            """      <div class="jump-moments">
        <span>Jump to:</span>
        <span class="jump-pill" onclick="jumpTo('first_msg')">💬 First Message</span>
        <span class="jump-pill" onclick="jumpTo('held')">⏸ Held Buffer</span>
        <span class="jump-pill" onclick="jumpTo('flush')">⚡ Flush Boundary</span>
        <span class="jump-pill" onclick="jumpTo('compaction')">📑 Compaction</span>
        <span class="jump-pill" onclick="jumpTo('submit')">🏁 Submission</span>
      </div>""",
            """      <div class="jump-moments">
        <span>Jump to:</span>
        <span id="jumpFirstMsgPill" class="jump-pill" onclick="jumpTo('first_msg')">💬 First Message</span>
        <span id="jumpWaitPollPill" class="jump-pill" style="display:none;" onclick="jumpTo('board_wait')">⏳ Wait Poll</span>
        <span id="jumpHeldPill" class="jump-pill" onclick="jumpTo('held')">⏸ Held Buffer</span>
        <span id="jumpFlushPill" class="jump-pill" onclick="jumpTo('flush')">⚡ Flush Boundary</span>
        <span class="jump-pill" onclick="jumpTo('compaction')">📑 Compaction</span>
        <span class="jump-pill" onclick="jumpTo('submit')">🏁 Submission</span>
      </div>""",
        ),
        (
            "    let playSpeed = 1.0;",
            "    let playSpeed = 1.0;\n    let centerTab = \"stream\";\n    const expandedQueueDetails = new Set();",
        ),
        (
            """    function formatInterventionLabel(rName, rData) {\n      const flags = rData.config_flags || {};""",
            """    function formatInterventionLabel(rName, rData) {\n      const flags = rData.config_flags || {};\n      if (rName.includes("delm") || flags.delm_board) {\n        return "DeLM Board";\n      }""",
        ),
        (
            """    function interventionSortOrder(label) {\n      if (label.startsWith("Baseline")) return 1;""",
            """    function interventionSortOrder(label) {\n      if (label === "DeLM Board") return 0;\n      if (label.startsWith("Baseline")) return 1;""",
        ),
        (
            '        subSpan.textContent = `(${mergeStatus}, ${rData.messages_sent} msgs)`;',
            """        const isDelm = rData.config_flags && rData.config_flags.delm_board;
        if (isDelm) {
          const waitText = rData.waits !== undefined ? `, ${rData.waits} waits` : "";
          subSpan.textContent = `(${mergeStatus}, ${rData.messages_sent} writes${waitText})`;
        } else {
          subSpan.textContent = `(${mergeStatus}, ${rData.messages_sent} msgs)`;
        }""",
        ),
        (
            """        const bestRun =
          runKeys.find(r => r.includes("ref-comm-combined")) ||
          runKeys.find(r => r.includes("delivery-timing")) ||
          runKeys[0];""",
            """        const bestRun =
          runKeys.find(r => r.includes("flash-ref-delm")) ||
          runKeys.find(r => r.includes("core-ref-delm")) ||
          runKeys.find(r => r.includes("delm")) ||
          runKeys.find(r => r.includes("ref-comm-combined")) ||
          runKeys.find(r => r.includes("delivery-timing")) ||
          runKeys[0];""",
        ),
        (
            '      document.getElementById("badgeSetting").textContent = formatSettingLabel(settingVal, gitEnabled);',
            """      const isDelm = Boolean(replayData.config_flags && replayData.config_flags.delm_board);
      document.getElementById("badgeSetting").textContent = isDelm
        ? `${formatSettingLabel(settingVal, gitEnabled)} + DeLM`
        : formatSettingLabel(settingVal, gitEnabled);
      document.getElementById("jumpFirstMsgPill").textContent = isDelm ? "💬 First Board Write" : "💬 First Message";
      document.getElementById("jumpWaitPollPill").style.display = isDelm ? "inline-block" : "none";
      document.getElementById("jumpHeldPill").style.display = isDelm ? "none" : "inline-block";
      document.getElementById("jumpFlushPill").style.display = isDelm ? "none" : "inline-block";

      const centerTabBar = document.getElementById("centerTabBar");
      if (isDelm) {
        centerTabBar.style.display = "flex";
        document.getElementById("centerWriteCount").textContent = (replayData.messages || []).length;
        const firstSnap = (replayData.board_snapshots || [])[0];
        document.getElementById("centerQueueCount").textContent = firstSnap ? (firstSnap.queue_items || []).length : 0;
      } else {
        centerTabBar.style.display = "none";
        centerTab = "stream";
      }
      applyCenterTabVisibility(isDelm);""",
        ),
        (
            '        else if (frame.kind === "nudge") markerClass = "marker-nudge";',
            '        else if (frame.kind === "board_wait") markerClass = "marker-held";\n        else if (frame.kind === "nudge") markerClass = "marker-nudge";',
        ),
        (
            """      // Active Threads
      const threadsContainer = document.getElementById("threadsListContainer");
      const openBadge = document.getElementById("openThreadsCountBadge");
      const threads = center.threads || [];
      const openThreads = threads.filter(t => t.status === "open");

      openBadge.textContent = `${openThreads.length} Open`;
      openBadge.className = `tag ${openThreads.length > 0 ? 'highlight' : 'success'}`;

      if (threads.length === 0) {
        threadsContainer.innerHTML = `<div style="color:var(--text-muted); font-size:11px; font-style:italic;">No active conversation threads.</div>`;
      } else {
        threadsContainer.innerHTML = "";
        threads.forEach(t => {
          const row = document.createElement("div");
          row.className = "thread-row";
          row.innerHTML = `
            <span class="thread-id">${t.thread_id}</span>
            <span class="thread-status-badge thread-status-${t.status}">${t.status}</span>
            <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
              <b>${t.last_sender}:</b> "${t.last_summary}"
            </span>
          `;
          threadsContainer.appendChild(row);
        });
      }

      // Chat Stream
      const chatStream = document.getElementById("chatStream");
      chatStream.innerHTML = "";

      const msgs = (center.messages || []).map(m => (m.content !== undefined ? m : { ...(messagesById[m.id] || {}), ...m }));
      if (msgs.length === 0) {
        chatStream.innerHTML = `<div style="color:var(--text-muted); text-align:center; padding:20px; font-style:italic;">No messages sent yet at this point in the timeline.</div>`;
      } else {
        msgs.forEach(m => {
          const wrapper = document.createElement("div");
          wrapper.className = `bubble-wrapper from-${m.sender}`;

          const intentClass = m.intent ? `intent-${m.intent}` : '';
          const intentTag = m.intent ? `<span class="intent-badge ${intentClass}">${m.intent}</span>` : '';
          const replyTag = m.reply_to ? `<span class="tag" style="font-size:9.5px;">↩ ${m.reply_to}</span>` : '';
          const waitTag = m.wait ? `<span class="tag danger" style="font-size:9.5px;">⏳ --wait</span>` : '';

          let statusClass = "delivered";
          if (m.live_status === "held") statusClass = "held";
          else if (m.live_status === "in_transit") statusClass = "in_transit";
          else if (m.live_status === "filtered_fyi" || m.live_status === "undelivered_peer_exited") statusClass = "filtered";

          let liveBadge = m.live_badge;
          if (!liveBadge) {
            if (m.live_status === "filtered_fyi") liveBadge = "🔇 Filtered FYI";
            else if (m.live_status === "delivered") liveBadge = `✓ Delivered @ Step ${m.delivered_step}`;
            else if (m.live_status === "held") liveBadge = `⏸ Held in ${m.recipient}'s buffer (${m.held_steps} steps)`;
            else if (m.live_status === "undelivered_peer_exited") liveBadge = "⏭ Peer exited before poll";
            else liveBadge = `⏳ In transit to ${m.recipient} (next poll)`;
          }

          wrapper.innerHTML = `
            <div class="bubble-meta">
              <b>${m.sender}</b>
              <span style="font-family:var(--font-mono);">${m.id}</span>
              ${intentTag}
              ${replyTag}
              ${waitTag}
            </div>
            <div class="chat-bubble">
              <div class="bubble-body">${escapeHtml(m.content)}</div>
              <div class="bubble-footer">
                <span style="color:var(--text-muted);">Sent @ Step ${m.sent_step}</span>
                <span class="status-badge ${statusClass}">${liveBadge}</span>
              </div>
            </div>
          `;
          chatStream.appendChild(wrapper);
        });
      }""",
            """      const isDelm = Boolean(replayData && replayData.config_flags && replayData.config_flags.delm_board);
      const headerTitle = document.getElementById("threadsHeaderTitle");
      if (headerTitle) {
        headerTitle.textContent = isDelm ? "DeLM Task Queue & Claims" : "Active Thread Board & Obligations";
      }

      // Active Threads / Queue Summary
      const threadsContainer = document.getElementById("threadsListContainer");
      const openBadge = document.getElementById("openThreadsCountBadge");
      const threads = center.threads || [];
      const openThreads = threads.filter(t => t.status === "open");
      const doneThreads = threads.filter(t => t.status !== "open");

      if (isDelm) {
        openBadge.textContent = `${openThreads.length} Claimed · ${doneThreads.length} Done`;
        openBadge.className = `tag ${openThreads.length > 0 ? 'highlight' : 'success'}`;
      } else {
        openBadge.textContent = `${openThreads.length} Open`;
        openBadge.className = `tag ${openThreads.length > 0 ? 'highlight' : 'success'}`;
      }

      if (threads.length === 0) {
        threadsContainer.innerHTML = `<div style="color:var(--text-muted); font-size:11px; font-style:italic;">${isDelm ? "No queue items." : "No active conversation threads."}</div>`;
      } else {
        threadsContainer.innerHTML = "";
        threads.forEach(t => {
          const statusLabel = isDelm ? (t.status === "open" ? "claimed" : "done") : t.status;
          const row = document.createElement("div");
          row.className = "thread-row";
          row.innerHTML = `
            <span class="thread-id">${escapeHtml(t.thread_id)}</span>
            <span class="thread-status-badge thread-status-${t.status}">${escapeHtml(statusLabel)}</span>
            <span style="flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
              <b>${escapeHtml(t.last_sender)}:</b> "${escapeHtml(t.last_summary)}"
            </span>
          `;
          threadsContainer.appendChild(row);
        });
      }

      if (isDelm) {
        renderDelmExtraTabs(frame);
      }

      // Chat / Shared Context Stream
      const chatStream = document.getElementById("chatStream");
      chatStream.innerHTML = "";

      const msgs = (center.messages || []).map(m => (m.content !== undefined ? m : { ...(messagesById[m.id] || {}), ...m }));
      const streamItems = msgs.map((m, i) => ({ kind: "msg", ts: m.sent_at || 0, ord: i, msg: m }));
      if (isDelm && Array.isArray(replayData.wait_events)) {
        const frameTs = frame.timestamp || 0;
        replayData.wait_events.forEach((w, i) => {
          if ((w.timestamp || 0) <= frameTs) {
            streamItems.push({ kind: "wait", ts: w.timestamp || 0, ord: msgs.length + i, wait: w });
          }
        });
        streamItems.sort((a, b) => (a.ts - b.ts) || (a.ord - b.ord));
      }

      if (streamItems.length === 0) {
        chatStream.innerHTML = `<div style="color:var(--text-muted); text-align:center; padding:20px; font-style:italic;">${isDelm ? "No board activity yet at this point in the timeline." : "No messages sent yet at this point in the timeline."}</div>`;
      } else {
        streamItems.forEach(item => {
          if (item.kind === "wait") {
            const w = item.wait;
            const div = document.createElement("div");
            div.className = "event-divider";
            let textHtml = "";
            if (w.woke_by === "timeout") {
              const sec = Math.round(w.waited_s || 0);
              textHtml = w.count > 1
                ? `<span>⏳</span><b>${escapeHtml(w.actor)}</b> polled <code>context.sh read --wait</code> ×${w.count} @ Step ${w.step} (${sec}s total, timeout — no peer update)`
                : `<span>⏳</span><b>${escapeHtml(w.actor)}</b> polled <code>context.sh read --wait</code> @ Step ${w.step} (${sec}s, timeout — no peer update)`;
            } else {
              const sec = Number(w.waited_s || 0).toFixed(1);
              textHtml = `<span>⏳</span><b>${escapeHtml(w.actor)}</b> <code>context.sh read --wait</code> woke after ${sec}s (${escapeHtml(w.woke_by)}) @ Step ${w.step}`;
            }
            div.innerHTML = `<div class="divider-content wait">${textHtml}</div>`;
            chatStream.appendChild(div);
            return;
          }

          const m = item.msg;
          const wrapper = document.createElement("div");
          wrapper.className = `bubble-wrapper from-${m.sender}`;

          const intentClass = m.intent ? `intent-${m.intent}` : '';
          const intentTag = m.intent ? `<span class="intent-badge ${intentClass}">${m.intent}</span>` : '';
          const replyTag = m.reply_to ? `<span class="tag" style="font-size:9.5px;">↩ ${m.reply_to}</span>` : '';
          const waitTag = (!isDelm && m.wait) ? `<span class="tag danger" style="font-size:9.5px;">⏳ --wait</span>` : '';

          let statusClass = "delivered";
          if (m.live_status === "held") statusClass = "held";
          else if (m.live_status === "in_transit") statusClass = "in_transit";
          else if (m.live_status === "filtered_fyi" || m.live_status === "undelivered_peer_exited") statusClass = "filtered";

          let liveBadge = m.live_badge;
          if (!liveBadge) {
            if (m.live_status === "filtered_fyi") liveBadge = "🔇 Filtered FYI";
            else if (m.live_status === "delivered") liveBadge = `✓ Delivered @ Step ${m.delivered_step}`;
            else if (m.live_status === "held") liveBadge = `⏸ Held in ${m.recipient}'s buffer (${m.held_steps} steps)`;
            else if (m.live_status === "undelivered_peer_exited") liveBadge = "⏭ Peer exited before poll";
            else liveBadge = `⏳ In transit to ${m.recipient} (next poll)`;
          }

          const stepPrefix = isDelm ? "Posted" : "Sent";
          wrapper.innerHTML = `
            <div class="bubble-meta">
              <b>${escapeHtml(m.sender)}</b>
              <span style="font-family:var(--font-mono);">${escapeHtml(m.id)}</span>
              ${intentTag}
              ${replyTag}
              ${waitTag}
            </div>
            <div class="chat-bubble">
              <div class="bubble-body">${escapeHtml(m.content)}</div>
              <div class="bubble-footer">
                <span style="color:var(--text-muted);">${stepPrefix} @ Step ${m.sent_step}</span>
                <span class="status-badge ${statusClass}">${escapeHtml(liveBadge)}</span>
              </div>
            </div>
          `;
          chatStream.appendChild(wrapper);
        });
      }""",
        ),
        (
            """      if (momentType === "first_msg") {
        targetIdx = replayData.timeline.findIndex(f => f.kind === "msg_sent");
      } else if (momentType === "held") {""",
            """      if (momentType === "first_msg") {
        targetIdx = replayData.timeline.findIndex(f => f.kind === "msg_sent");
      } else if (momentType === "board_wait") {
        targetIdx = replayData.timeline.findIndex(f => f.kind === "board_wait");
      } else if (momentType === "held") {""",
        ),
        (
            "    function switchTab(agent, tab) {",
            """    function switchCenterTab(tab) {
      centerTab = tab;
      const isDelm = Boolean(replayData && replayData.config_flags && replayData.config_flags.delm_board);
      applyCenterTabVisibility(isDelm);
    }

    function applyCenterTabVisibility(isDelm) {
      const activeTab = isDelm ? centerTab : "stream";
      document.querySelectorAll("#centerTabBar .tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.centerTab === activeTab);
      });
      document.getElementById("threadsStickyBox").style.display = activeTab === "stream" ? "flex" : "none";
      document.getElementById("chatStream").style.display = activeTab === "stream" ? "flex" : "none";
      document.getElementById("centerQueueView").style.display = (isDelm && activeTab === "queue") ? "flex" : "none";
      document.getElementById("centerRawView").style.display = (isDelm && activeTab === "raw") ? "flex" : "none";
    }

    function renderDelmExtraTabs(frame) {
      const snapshots = (replayData && replayData.board_snapshots) || [];
      const snapIdx = frame.board_snapshot_idx !== undefined ? frame.board_snapshot_idx : 0;
      const snap = snapshots[snapIdx] || { sessions: [], statuses: [], queue_items: [], raw_text: "" };

      const queueCountEl = document.getElementById("centerQueueCount");
      if (queueCountEl) {
        queueCountEl.textContent = (snap.queue_items || []).length;
      }

      const rawPre = document.getElementById("centerRawPre");
      if (rawPre) {
        rawPre.textContent = snap.raw_text || "";
      }

      const queueView = document.getElementById("centerQueueView");
      if (!queueView) return;
      queueView.innerHTML = "";

      const statusByActor = {};
      (snap.statuses || []).forEach(s => { statusByActor[s.actor] = s.text; });

      const strip = document.createElement("div");
      strip.className = "session-strip";
      (snap.sessions || []).forEach(ses => {
        const card = document.createElement("div");
        card.className = "session-card";
        const pillCls = ses.actor === "agent1" ? "agent1-pill" : "agent2-pill";
        const stCls = `session-state-${ses.state || "starting"}`;
        const statusText = statusByActor[ses.actor] || "(no status set)";
        card.innerHTML = `
          <div class="session-card-head">
            <span class="agent-pill ${pillCls}">${escapeHtml(ses.actor)}</span>
            <span class="session-state-pill ${stCls}">${escapeHtml(ses.state || "starting")}</span>
          </div>
          <div class="session-status-text">${escapeHtml(statusText)}</div>
        `;
        strip.appendChild(card);
      });
      queueView.appendChild(strip);

      (snap.queue_items || []).forEach(item => {
        const card = document.createElement("div");
        card.className = "queue-card";
        const isDone = item.state === "done";
        const ownerCls = item.owner === "agent1" ? "agent1-pill" : (item.owner === "agent2" ? "agent2-pill" : "");
        const badgeCls = isDone ? "thread-status-resolved" : "thread-status-open";
        const badgeText = isDone ? "done" : (item.state === "claimed" ? "claimed" : "open");
        const isOpen = expandedQueueDetails.has(item.raw_id);
        const filesHtml = (item.files && item.files.length > 0)
          ? `<div style="margin-top:4px; font-family:var(--font-mono); font-size:10.5px;">Files: ${item.files.map(f => escapeHtml(f)).join(", ")}</div>`
          : "";
        const doneHtml = isDone
          ? `<div class="done-summary-box"><div class="done-summary-title">Completion Summary (done)</div><div style="white-space:pre-wrap; word-break:break-word;">${escapeHtml(item.summary || "")}</div>${filesHtml}</div>`
          : "";

        card.innerHTML = `
          <div class="queue-card-head">
            <div style="display:flex; align-items:center; gap:6px;">
              <span class="thread-id">${escapeHtml(item.label)}</span>
              <span class="tag">${escapeHtml(item.id)}</span>
              <span class="agent-pill ${ownerCls}">${escapeHtml(item.owner)}</span>
            </div>
            <span class="thread-status-badge ${badgeCls}">${escapeHtml(badgeText)}</span>
          </div>
          <div class="queue-card-subject">${escapeHtml(item.subject)}</div>
          <details class="queue-detail-toggle" ${isOpen ? "open" : ""}>
            <summary>Feature Spec / Detail</summary>
            <div class="queue-detail-body">${escapeHtml(item.detail || "")}</div>
          </details>
          ${doneHtml}
        `;
        const detailsEl = card.querySelector("details");
        if (detailsEl) {
          detailsEl.addEventListener("toggle", () => {
            if (detailsEl.open) expandedQueueDetails.add(item.raw_id);
            else expandedQueueDetails.delete(item.raw_id);
          });
        }
        queueView.appendChild(card);
      });
    }

    function switchTab(agent, tab) {""",
        ),
    )

    html = template_html
    for old, new in replacements:
        if old not in html:
            raise ValueError(f"Expected replay template anchor not found: {old[:80]!r}")
        html = html.replace(old, new, 1)
    return html


def convert_monotonic_to_epoch(
    events: list[dict[str, Any]],
    steps1: list[Any],
    steps2: list[Any],
) -> dict[int, float]:
    """Convert DeLM ``time.monotonic_ns()`` event timestamps into wall-clock epoch seconds."""
    if not events:
        return {}

    for ev in events:
        if ev.get("kind") != "session":
            continue
        payload = ev.get("payload") or {}
        if payload.get("recorded_at_ns") is not None and ev.get("time_ns") is not None:
            ref_mono_ns = int(ev["time_ns"])
            ref_wall_ns = int(payload["recorded_at_ns"])
            return {
                int(item["sequence"]): (int(item["time_ns"]) - ref_mono_ns + ref_wall_ns) / 1e9
                for item in events
            }

    step_times = [float(s.timestamp) for s in (*steps1, *steps2) if float(s.timestamp) > 0.0]
    mono_times = [int(ev["time_ns"]) for ev in events]
    min_mono = min(mono_times)
    max_mono = max(mono_times)
    if not step_times or max_mono == min_mono:
        base_ts = min(step_times) if step_times else 0.0
        return {
            int(ev["sequence"]): base_ts + (int(ev["time_ns"]) - min_mono) / 1e9
            for ev in events
        }

    min_step = min(step_times)
    span_step = max(step_times) - min_step
    span_mono = max_mono - min_mono
    return {
        int(ev["sequence"]): min_step + ((int(ev["time_ns"]) - min_mono) / span_mono) * span_step
        for ev in events
    }


def _load_cooperbench_compiler(cooperbench_root: Path | None = None) -> Any:
    cb_root = cooperbench_root if cooperbench_root is not None else resolve_cooperbench_root()
    cb_root_str = str(cb_root)
    if cb_root_str not in sys.path:
        sys.path.insert(0, cb_root_str)
    return importlib.import_module("tools.replay_viewer.compiler")


def _reclassify_delm_step(step: Any) -> Any:
    if step.command and "context.sh" in step.command:
        return replace(step, action_phase="message")
    return step


def _compile_delm_board(
    *,
    board_data: dict[str, Any],
    steps1: list[Any],
    steps2: list[Any],
    compiler: Any,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[float], int]:
    events = sorted(board_data.get("events") or [], key=lambda e: int(e.get("sequence", 0)))
    bodies = board_data.get("bodies") or []
    preseed_max = int(board_data.get("preseed_max_sequence", 0))
    seq_to_epoch = convert_monotonic_to_epoch(events, steps1, steps2)

    body_files: dict[str, list[str]] = {
        str(b["id"]): [str(f["path"]) for f in (b.get("files") or [])]
        for b in bodies
        if b.get("id")
    }
    item_labels: dict[str, str] = {}
    queue_events: list[dict[str, Any]] = []
    for ev in events:
        seq = int(ev.get("sequence", 0))
        kind = str(ev.get("kind") or "")
        payload = ev.get("payload") or {}
        if kind == "item" and payload.get("id"):
            raw_id = str(payload["id"])
            item_labels[raw_id] = f"Item {len(item_labels) + 1}"
        if kind in ("item", "claim", "reserve", "done"):
            queue_events.append(
                {
                    "sequence": seq,
                    "preseed": seq <= preseed_max,
                    "ts": seq_to_epoch.get(seq, 0.0),
                    "actor": str(ev.get("actor") or ""),
                    "kind": kind,
                    "payload": payload,
                    "label": item_labels.get(str(payload.get("item") or payload.get("id") or ""), ""),
                }
            )

    steps_by_actor = {"agent1": steps1, "agent2": steps2}
    peer_view_events = [
        (ev, seq_to_epoch[int(ev["sequence"])])
        for ev in events
        if int(ev.get("sequence", 0)) > preseed_max and str(ev.get("kind") or "") in _BOARD_VIEW_KINDS
    ]

    write_records, wait_records = _collapse_post_preseed_events(
        events=events,
        preseed_max=preseed_max,
        seq_to_epoch=seq_to_epoch,
        item_labels=item_labels,
        body_files=body_files,
    )

    messages: list[Any] = []
    for idx, rec in enumerate(write_records, start=1):
        sender = rec["actor"]
        recipient = "agent2" if sender == "agent1" else "agent1"
        sent_at = rec["sent_at"]

        sent_step = _latest_step_number(steps_by_actor.get(sender, []), sent_at)
        r_step_at_send, r_action_at_send, r_rc_at_send = _receiver_state_at(
            steps_by_actor.get(recipient, []),
            sent_at,
        )

        deliv_at: float | None = None
        deliv_step: int | None = None
        deliv_status = "undelivered_peer_exited"
        for pev, pev_ts in peer_view_events:
            if pev.get("actor") == recipient and int(pev["time_ns"]) > rec["write_time_ns"]:
                deliv_at = pev_ts
                deliv_step = _latest_step_number(steps_by_actor.get(recipient, []), deliv_at)
                deliv_status = "immediate"
                break

        messages.append(
            compiler.MessageEvent(
                id=f"b{idx}",
                sender=sender,
                recipient=recipient,
                content=rec["content"],
                intent=rec["intent"],
                reply_to=None,
                resolve=(rec["kind"] == "done"),
                wait=False,
                sent_step=sent_step,
                sent_at=sent_at,
                receiver_step_at_send=r_step_at_send,
                receiver_action_at_send=r_action_at_send,
                receiver_rc_at_send=r_rc_at_send,
                delivered_step=deliv_step,
                delivered_at=deliv_at,
                held_steps=0,
                hold_reason="",
                flush_reason="",
                delivery_status=deliv_status,
            )
        )

    wait_events: list[dict[str, Any]] = []
    total_waits = 0
    for idx, wrec in enumerate(wait_records, start=1):
        actor = wrec["actor"]
        step_num = _latest_step_number(steps_by_actor.get(actor, []), wrec["sent_at"])
        total_waits += int(wrec["count"])
        wait_events.append(
            {
                "id": f"w{idx}",
                "actor": actor,
                "step": step_num,
                "timestamp": wrec["sent_at"],
                "count": int(wrec["count"]),
                "waited_s": round(float(wrec["waited_s"]), 1),
                "woke_by": wrec["woke_by"],
            }
        )

    board_snapshots, snapshot_times = _build_board_snapshots(
        events=events,
        bodies=bodies,
        preseed_max=preseed_max,
        seq_to_epoch=seq_to_epoch,
    )
    return messages, wait_events, queue_events, board_snapshots, snapshot_times, total_waits


def _collapse_post_preseed_events(
    *,
    events: list[dict[str, Any]],
    preseed_max: int,
    seq_to_epoch: dict[int, float],
    item_labels: dict[str, str],
    body_files: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    writes: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    pending_waits: dict[str, dict[str, Any]] = {}

    def flush_actor_wait(actor: str) -> None:
        buf = pending_waits.pop(actor, None)
        if buf is None:
            return
        waits.append(
            {
                "actor": actor,
                "sent_at": buf["sent_at"],
                "count": buf["count"],
                "waited_s": buf["total_waited"],
                "woke_by": "timeout",
            }
        )

    def flush_all_waits() -> None:
        for actor_key in list(pending_waits.keys()):
            flush_actor_wait(actor_key)

    for ev in events:
        seq = int(ev.get("sequence", 0))
        if seq <= preseed_max:
            continue
        actor = str(ev.get("actor") or "")
        if actor not in ("agent1", "agent2"):
            continue
        kind = str(ev.get("kind") or "")
        payload = ev.get("payload") or {}
        ts = seq_to_epoch.get(seq, 0.0)
        time_ns = int(ev.get("time_ns") or 0)

        if kind in ("read", "session", "body"):
            continue

        if kind == "wait":
            woke_by = str(payload.get("woke_by") or "timeout")
            waited = float(payload.get("waited") or 0.0)
            if woke_by == "timeout":
                buf = pending_waits.get(actor)
                if buf is None:
                    pending_waits[actor] = {
                        "count": 1,
                        "total_waited": waited,
                        "sent_at": ts,
                    }
                else:
                    buf["count"] += 1
                    buf["total_waited"] += waited
                continue

            flush_actor_wait(actor)
            waits.append(
                {
                    "actor": actor,
                    "sent_at": ts,
                    "count": 1,
                    "waited_s": waited,
                    "woke_by": woke_by,
                }
            )
            continue

        if kind == "status" and payload.get("auto") is True:
            continue

        flush_all_waits()
        formatted = _format_board_entry(kind=kind, payload=payload, item_labels=item_labels, body_files=body_files)
        if formatted is None:
            continue
        intent, content = formatted
        writes.append(
            {
                "actor": actor,
                "kind": kind,
                "intent": intent,
                "content": content,
                "sent_at": ts,
                "write_time_ns": time_ns,
            }
        )

    flush_all_waits()
    writes.sort(key=lambda r: r["sent_at"])
    waits.sort(key=lambda r: r["sent_at"])
    return writes, waits


def _format_board_entry(
    *,
    kind: str,
    payload: dict[str, Any],
    item_labels: dict[str, str],
    body_files: dict[str, list[str]],
) -> tuple[str, str] | None:
    if kind == "note":
        note_type = str(payload.get("type") or "FACT").upper()
        intent = "FAIL" if note_type == "FAIL" else "FACT"
        scope = str(payload.get("scope") or "").strip()
        text = str(payload.get("text") or "").strip()
        basis = str(payload.get("basis") or "").strip()
        parts: list[str] = []
        if scope:
            parts.append(f"[Scope: {scope}]")
        if text:
            parts.append(text)
        if basis:
            parts.append(f"(Basis: {basis})")
        files = body_files.get(str(payload.get("id") or ""), [])
        if files:
            parts.append(f"[Attached: {', '.join(files)}]")
        return intent, "\n".join(parts)

    if kind == "status":
        return "STATUS", str(payload.get("text") or "").strip()

    if kind == "done":
        pub_id = str(payload.get("publication_id") or payload.get("id") or "")
        item_label = item_labels.get(str(payload.get("item") or ""), "Item")
        summary = str(payload.get("summary") or "").strip()
        files = body_files.get(pub_id, [])
        content = f"[{item_label}] {summary}".strip()
        if files:
            content = f"{content}\n[Attached: {', '.join(files)}]"
        return "DONE", content

    if kind == "publish":
        items = payload.get("items") or []
        subjects = [str(it.get("subject") or "").strip() for it in items if isinstance(it, dict)]
        body = "\n".join(f"- {s}" for s in subjects if s) or "Published open queue item(s)"
        return "PROPOSAL", f"Published {len(items)} queue item(s):\n{body}"

    if kind == "claim":
        item_label = item_labels.get(str(payload.get("item") or ""), str(payload.get("item") or "")[:8])
        return "PROPOSAL", f"Claimed {item_label}"

    if kind == "reserve":
        subject = str(payload.get("subject") or "").strip()
        detail = str(payload.get("detail") or "").strip()
        return "PROPOSAL", f"Reserved: {subject}\n{detail}".strip()

    if kind == "apply":
        pub_id = str(payload.get("id") or "")[:8]
        decisions = payload.get("decisions") or {}
        if payload.get("applied"):
            summary = ", ".join(f"{p} ({d})" for p, d in decisions.items()) or "clean apply"
            return "APPLY", f"Applied attachment {pub_id}: {summary}"
        conflicts = ", ".join(sorted((payload.get("conflicts") or {}).keys())) or "local conflict"
        return "APPLY", f"Apply rejected for {pub_id} (conflict on {conflicts})"

    if kind == "apply_error":
        return "APPLY", f"Apply failed for {str(payload.get('id') or '')[:8]}: {payload.get('error', '')}"

    return None


def _build_board_snapshots(
    *,
    events: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
    preseed_max: int,
    seq_to_epoch: dict[int, float],
    actors: tuple[str, ...] = ("agent1", "agent2"),
) -> tuple[list[dict[str, Any]], list[float]]:
    """Build deduplicated board snapshots keyed by non-read state transitions."""
    body_files_meta: dict[str, list[dict[str, Any]]] = {
        str(b["id"]): list(b.get("files") or [])
        for b in bodies
        if b.get("id")
    }
    sessions = dict.fromkeys(actors, "starting")
    statuses: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    notes: list[dict[str, Any]] = []
    attached: dict[str, list[dict[str, Any]]] = dict(body_files_meta)
    last_seq = 0

    def apply_state_event(ev: dict[str, Any]) -> None:
        nonlocal last_seq
        last_seq = max(last_seq, int(ev.get("sequence", 0)))
        actor = str(ev.get("actor") or "")
        kind = str(ev.get("kind") or "")
        payload = ev.get("payload") or {}

        if kind == "session":
            target_actor = str(payload.get("actor") or "")
            if target_actor in sessions:
                sessions[target_actor] = str(payload.get("state") or "starting")
        elif kind == "status":
            if actor in actors:
                statuses[actor] = str(payload.get("text") or "")
        elif kind == "item":
            raw_id = str(payload.get("id") or "")
            if raw_id and raw_id not in items:
                ordered_ids.append(raw_id)
                items[raw_id] = {**payload, "state": "open"}
        elif kind == "claim":
            raw_id = str(payload.get("item") or "")
            if raw_id in items:
                items[raw_id].update(state="claimed", owner=actor)
        elif kind == "reserve":
            raw_id = str(payload.get("item") or payload.get("id") or "")
            if raw_id and raw_id not in items:
                ordered_ids.append(raw_id)
                items[raw_id] = {
                    "id": raw_id,
                    "subject": payload.get("subject", ""),
                    "detail": payload.get("detail", ""),
                    "state": "claimed",
                    "owner": actor,
                }
        elif kind == "done":
            raw_id = str(payload.get("item") or "")
            if raw_id in items:
                items[raw_id].update(
                    state="done",
                    summary=payload.get("summary", ""),
                    owner=actor,
                    publication_id=payload.get("publication_id", raw_id),
                )
        elif kind == "note":
            notes.append({**payload, "actor": actor})
        elif kind == "body":
            pub_id = str(payload.get("id") or "")
            if pub_id and isinstance(payload.get("files"), list):
                attached[pub_id] = list(payload["files"])

    def capture_snapshot() -> dict[str, Any]:
        sessions_list = [{"actor": a, "state": sessions[a]} for a in actors]
        statuses_list = [{"actor": a, "text": statuses[a]} for a in actors if a in statuses]

        queue_items: list[dict[str, Any]] = []
        raw_queue: list[dict[str, Any]] = []
        for idx, item_id in enumerate(ordered_ids, start=1):
            item = items[item_id]
            pub_id = str(item.get("publication_id") or item_id)
            item_files = attached.get(pub_id, [])
            queue_items.append(
                {
                    "label": f"Item {idx}",
                    "id": item_id[:8],
                    "raw_id": item_id,
                    "owner": str(item.get("owner") or "unclaimed"),
                    "state": str(item.get("state") or "open"),
                    "subject": str(item.get("subject") or ""),
                    "detail": str(item.get("detail") or ""),
                    "summary": str(item.get("summary") or ""),
                    "files": [str(f.get("path") or "") for f in item_files if isinstance(f, dict)],
                }
            )
            raw_entry = dict(item)
            if pub_id in attached:
                raw_entry["files"] = attached[pub_id]
            raw_queue.append(raw_entry)

        raw_notes: list[dict[str, Any]] = []
        for note in notes:
            n_entry = dict(note)
            note_id = str(n_entry.get("id") or "")
            if note_id in attached:
                n_entry["files"] = attached[note_id]
            raw_notes.append(n_entry)

        view_dict = {
            "reader": "viewer",
            "agents": list(actors),
            "sessions": sessions_list,
            "status": statuses_list,
            "queue": raw_queue,
            "notes": raw_notes,
            "sequence": last_seq,
            "full": True,
        }
        return {
            "sequence": last_seq,
            "sessions": sessions_list,
            "statuses": statuses_list,
            "queue_items": queue_items,
            "raw_text": json.dumps(view_dict, ensure_ascii=False, indent=2),
        }

    for ev in events:
        if int(ev.get("sequence", 0)) <= preseed_max:
            apply_state_event(ev)

    preseed_ts = min(
        (seq_to_epoch[int(ev["sequence"])] for ev in events if int(ev.get("sequence", 0)) <= preseed_max),
        default=0.0,
    )
    snapshots = [capture_snapshot()]
    snapshot_times = [preseed_ts]

    for ev in events:
        seq = int(ev.get("sequence", 0))
        if seq <= preseed_max:
            continue
        if str(ev.get("kind") or "") not in _BOARD_STATE_KINDS:
            continue
        apply_state_event(ev)
        snapshots.append(capture_snapshot())
        snapshot_times.append(seq_to_epoch.get(seq, 0.0))

    return snapshots, snapshot_times


def _build_delm_timeline(
    *,
    steps1: list[Any],
    steps2: list[Any],
    messages: list[Any],
    wait_events: list[dict[str, Any]],
    queue_events: list[dict[str, Any]],
    snapshot_times: list[float],
    compactions1: list[Any],
    compactions2: list[Any],
    nudges1: list[Any],
    nudges2: list[Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for s in steps1:
        events.append({"ts": s.timestamp, "kind": "step", "agent": "agent1", "step_num": s.step_number, "data": s})
    for s in steps2:
        events.append({"ts": s.timestamp, "kind": "step", "agent": "agent2", "step_num": s.step_number, "data": s})
    for w in wait_events:
        events.append({"ts": w["timestamp"], "kind": "board_wait", "agent": w["actor"], "data": w})
    for m in messages:
        events.append({"ts": m.sent_at, "kind": "msg_sent", "agent": m.sender, "msg_id": m.id, "data": m})
        if m.delivered_at is not None:
            events.append({"ts": m.delivered_at, "kind": "msg_delivered", "agent": m.recipient, "msg_id": m.id, "data": m})
    for c in compactions1:
        events.append({"ts": c.timestamp, "kind": "compaction", "agent": "agent1", "data": c})
    for c in compactions2:
        events.append({"ts": c.timestamp, "kind": "compaction", "agent": "agent2", "data": c})
    for n in nudges1:
        events.append({"ts": n.timestamp, "kind": "nudge", "agent": "agent1", "data": n})
    for n in nudges2:
        events.append({"ts": n.timestamp, "kind": "nudge", "agent": "agent2", "data": n})

    events.sort(key=lambda e: (e["ts"], 0 if e["kind"] == "step" else 1))

    preseed_queue = [qe for qe in queue_events if qe["preseed"]]
    dynamic_queue = [qe for qe in queue_events if not qe["preseed"]]

    frames: list[dict[str, Any]] = []
    for idx, ev in enumerate(events):
        curr_ts = ev["ts"]
        queue_rows = _compute_queue_snapshot(preseed_queue, dynamic_queue, curr_ts)
        snap_idx = _latest_snapshot_index(snapshot_times, curr_ts)

        a1_step = _latest_step_number(steps1, curr_ts)
        a2_step = _latest_step_number(steps2, curr_ts)
        a1_comp = sum(1 for c in compactions1 if c.timestamp <= curr_ts)
        a2_comp = sum(1 for c in compactions2 if c.timestamp <= curr_ts)

        center_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.sent_at > curr_ts:
                continue
            if m.delivered_at is not None and curr_ts >= m.delivered_at:
                live_status = "delivered"
                live_badge = f"👁 Read by {m.recipient} @ Step {m.delivered_step}"
            elif m.delivery_status == "undelivered_peer_exited" and _peer_has_exited(
                steps1 if m.recipient == "agent1" else steps2,
                curr_ts,
            ):
                live_status = "undelivered_peer_exited"
                live_badge = f"⏭ Unread before {m.recipient} exited"
            else:
                live_status = "in_transit"
                live_badge = f"📌 On board (unread by {m.recipient})"
            center_msgs.append({"id": m.id, "live_status": live_status, "live_badge": live_badge})

        frames.append(
            {
                "frame_index": idx,
                "timestamp": curr_ts,
                "kind": ev["kind"],
                "agent": ev.get("agent"),
                "delivery_status": ev["data"].delivery_status if ev["kind"] == "msg_delivered" else None,
                "description": _format_delm_description(ev),
                "board_snapshot_idx": snap_idx,
                "agent1": {"step": a1_step, "compactions": a1_comp, "held_ids": []},
                "agent2": {"step": a2_step, "compactions": a2_comp, "held_ids": []},
                "center": {
                    "messages": center_msgs,
                    "threads": queue_rows,
                    "active_nudge": (
                        {"kind": ev["data"].kind, "message": ev["data"].message[:120]}
                        if ev["kind"] == "nudge"
                        else None
                    ),
                    "active_compaction": (
                        {"agent_id": ev["data"].agent_id, "summary": ev["data"].summary[:120]}
                        if ev["kind"] == "compaction"
                        else None
                    ),
                },
            }
        )

    return frames


def _latest_snapshot_index(snapshot_times: list[float], curr_ts: float) -> int:
    idx = 0
    for k, ts in enumerate(snapshot_times):
        if ts <= curr_ts:
            idx = k
        else:
            break
    return idx


def _compute_queue_snapshot(
    preseed_queue: list[dict[str, Any]],
    dynamic_queue: list[dict[str, Any]],
    curr_ts: float,
) -> list[dict[str, str]]:
    items: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []

    def apply_qe(qe: dict[str, Any]) -> None:
        kind = qe["kind"]
        payload = qe["payload"]
        actor = qe["actor"]
        if kind == "item":
            raw_id = str(payload.get("id") or "")
            if raw_id and raw_id not in items:
                label = f"Item {len(ordered_ids) + 1}"
                subject = str(payload.get("subject") or label)
                ordered_ids.append(raw_id)
                items[raw_id] = {
                    "thread_id": label,
                    "status": "open",
                    "last_sender": actor or "system",
                    "last_summary": _truncate_line(subject, 80),
                }
        elif kind == "claim":
            raw_id = str(payload.get("item") or "")
            if raw_id in items:
                items[raw_id]["last_sender"] = actor
        elif kind == "reserve":
            raw_id = str(payload.get("item") or payload.get("id") or "")
            if raw_id and raw_id not in items:
                label = f"Item {len(ordered_ids) + 1}"
                ordered_ids.append(raw_id)
                items[raw_id] = {
                    "thread_id": label,
                    "status": "open",
                    "last_sender": actor,
                    "last_summary": _truncate_line(str(payload.get("subject") or label), 80),
                }
        elif kind == "done":
            raw_id = str(payload.get("item") or "")
            if raw_id in items:
                items[raw_id]["status"] = "resolved"
                items[raw_id]["last_sender"] = actor
                items[raw_id]["last_summary"] = _truncate_line(str(payload.get("summary") or "done"), 80)

    for qe in preseed_queue:
        apply_qe(qe)
    for qe in dynamic_queue:
        if qe["ts"] <= curr_ts:
            apply_qe(qe)

    return [items[item_id] for item_id in ordered_ids]


def _format_delm_description(ev: dict[str, Any]) -> str:
    kind = ev["kind"]
    agent = ev.get("agent", "")
    data = ev.get("data")
    if kind == "step":
        cmd_s = data.command[:40] if data.command else "think"
        return f"{agent} Step {data.step_number}: {cmd_s}"
    if kind == "board_wait":
        woke_by = data.get("woke_by", "timeout")
        waited_s = float(data.get("waited_s") or 0.0)
        count = int(data.get("count") or 1)
        if woke_by == "timeout":
            mult = f" ×{count}" if count > 1 else ""
            return f"{agent} polled context.sh read --wait{mult} ({waited_s:.0f}s)"
        return f"{agent} read --wait woke ({woke_by}, {waited_s:.1f}s)"
    if kind == "msg_sent":
        return f"{agent} posted {data.id} ({data.intent or 'board'})"
    if kind == "msg_delivered":
        return f"{agent} read {data.id} ({data.intent or 'board'})"
    if kind == "compaction":
        return f"{agent} context compaction ({data.summary[:30]}...)"
    if kind == "nudge":
        return f"{agent} harness nudge: {data.kind}"
    return f"{agent} {kind}"


def _latest_step_number(steps: list[Any], ts: float) -> int:
    step_num = 0
    for s in steps:
        if float(s.timestamp) <= ts:
            step_num = int(s.step_number)
        else:
            break
    return step_num


def _receiver_state_at(steps: list[Any], ts: float) -> tuple[int, str, int]:
    step_num = 0
    action = "none"
    rc = 0
    for s in steps:
        if float(s.timestamp) <= ts:
            step_num = int(s.step_number)
            action = str(s.action_phase)
            rc = int(s.returncode) if s.returncode is not None else 0
        else:
            break
    return step_num, action, rc


def _peer_has_exited(peer_steps: list[Any], curr_ts: float) -> bool:
    if not peer_steps:
        return True
    return curr_ts >= float(peer_steps[-1].timestamp)


def _truncate_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."


def _load_runs_meta(
    logs_dir: Path,
    results_dir: Path,
    *,
    compiler: Any,
    is_delm: bool,
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    if not is_delm:
        for r in compiler.list_runs(logs_dir=logs_dir, results_dir=results_dir):
            by_name[r["run_name"]] = r
        return by_name

    for res_dir in sorted(results_dir.iterdir()) if results_dir.is_dir() else []:
        run_json = _load_json(res_dir / "run.json")
        if not run_json:
            continue
        rname = str(run_json.get("run_name") or res_dir.name)
        flags = compiler._extract_config_flags(run_json, logs_dir / rname)
        flags["delm_board"] = True
        by_name[rname] = {
            "run_name": rname,
            "setting": run_json.get("setting", "coop"),
            "model": run_json.get("model", ""),
            "subset": run_json.get("subset", ""),
            "git_enabled": bool(run_json.get("git_enabled", True)),
            "budgets": run_json.get("budgets") or {},
            "headline": run_json.get("headline") or {},
            "config_flags": flags,
            "has_results_dir": True,
        }
    return by_name


def _load_pairs_lookup(
    pair_specs: tuple[ShowcasePairSpec, ...],
    *,
    logs_dir: Path,
    results_dir: Path,
    compiler: Any,
    is_delm: bool,
) -> dict[tuple[str, str, int, int, int], dict[str, Any]]:
    wanted_runs = {
        run_name
        for spec in pair_specs
        for run_name in (spec.delm_runs if is_delm else spec.cooperbench_runs)
    }
    lookup: dict[tuple[str, str, int, int, int], dict[str, Any]] = {}
    for run_name in sorted(wanted_runs):
        for row in compiler._load_run_pairs(run_name, logs_dir, results_dir):
            key = (run_name, row["repo"], int(row["task_id"]), int(row["feature_a"]), int(row["feature_b"]))
            lookup[key] = row
        csv_path = results_dir / run_name / "pairs.csv"
        if is_delm and csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    key = (
                        run_name,
                        row["repo"],
                        int(row["task_id"]),
                        int(row["feature_a"]),
                        int(row["feature_b"]),
                    )
                    if key in lookup:
                        lookup[key]["waits"] = int(row.get("waits") or 0)
    return lookup


def _build_pair_run_summary(
    *,
    run_name: str,
    replay: dict[str, Any],
    pair_row: dict[str, Any] | None,
) -> dict[str, Any]:
    res = replay.get("result") or {}
    ev = replay.get("eval") or {}
    merge = ev.get("merge") or {}
    both_passed = bool(pair_row["both_passed"]) if pair_row else bool(ev.get("both_passed", False))
    merge_status = (pair_row or {}).get("merge_status") or merge.get("status") or "unknown"
    summary = {
        "run_name": run_name,
        "setting": replay.get("setting", "coop"),
        "git_enabled": bool(replay.get("git_enabled", True)),
        "model": replay.get("model", ""),
        "budgets": replay.get("budgets") or {},
        "both_passed": both_passed,
        "merge_status": merge_status,
        "messages_sent": int((pair_row or {}).get("messages_sent", res.get("messages_sent", 0))),
        "steps_total": int((pair_row or {}).get("steps_total", res.get("total_steps", 0))),
        "cost_usd": float((pair_row or {}).get("cost_usd", res.get("total_cost", 0.0))),
        "config_flags": replay.get("config_flags") or {},
    }
    if "waits" in replay:
        summary["waits"] = int(replay["waits"])
    elif pair_row and "waits" in pair_row:
        summary["waits"] = int(pair_row["waits"])
    return summary


def _compact_replay(
    replay: dict[str, Any],
    *,
    max_output_chars: int,
    max_thought_chars: int,
) -> dict[str, Any]:
    out = copy.deepcopy(replay)
    raw_eval = out.get("eval")
    if isinstance(raw_eval, dict):
        merge_info = raw_eval.get("merge") if isinstance(raw_eval.get("merge"), dict) else {}
        out["eval"] = {
            "both_passed": bool(raw_eval.get("both_passed", False)),
            "merge": {
                "status": merge_info.get("status", "unknown"),
                "strategy": merge_info.get("strategy", ""),
            },
        }
    for agent_id in ("agent1", "agent2"):
        for step in (out.get("steps") or {}).get(agent_id, []):
            output_text = step.get("output")
            if isinstance(output_text, str) and len(output_text) > max_output_chars:
                step["output"] = (
                    f"{output_text[:max_output_chars]}\n... [truncated {len(output_text) - max_output_chars} chars]"
                )
            thought_text = step.get("thought")
            if isinstance(thought_text, str) and len(thought_text) > max_thought_chars:
                step["thought"] = (
                    f"{thought_text[:max_thought_chars]}\n... [truncated {len(thought_text) - max_thought_chars} chars]"
                )
    return out


def _render_prompt_markdown(
    system_template: str,
    instance_template: str,
    template_vars: dict[str, Any],
) -> str:
    rendered_sys = jinja2.Template(system_template, undefined=jinja2.StrictUndefined).render(**template_vars).strip()
    rendered_inst = (
        jinja2.Template(instance_template, undefined=jinja2.StrictUndefined).render(**template_vars).strip()
    )
    return f"# System Prompt\n\n{rendered_sys}\n\n# Instance Prompt\n\n{rendered_inst}\n"


def _unified_diff_section(
    before_text: str,
    after_text: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    before_lines = (before_text.rstrip("\n") + "\n").splitlines(keepends=True)
    after_lines = (after_text.rstrip("\n") + "\n").splitlines(keepends=True)
    return "".join(difflib.unified_diff(before_lines, after_lines, fromfile=fromfile, tofile=tofile, n=3))


def _write_sanitized_patch(
    path: Path,
    *,
    header: str,
    diff_body: str,
    cb_root: Path,
    secret_tokens: list[str],
) -> Path:
    raw_patch = f"{header}\n\n{diff_body}"
    sanitized_patch = _sanitize_payload(raw_patch, cb_root=cb_root, secret_tokens=secret_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitized_patch, encoding="utf-8")
    return path


def _collect_sensitive_tokens(cooperbench_root: Path) -> list[str]:
    tokens: list[str] = []
    env_file = cooperbench_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _ENV_SECRET_LINE_RE.match(line)
            if match:
                val = match.group(1).strip().strip("'\"")
                if len(val) > 3:
                    tokens.append(val)
    for candidate in (
        str(Path.home()),
        os.environ.get("USER", "").strip(),
        getpass.getuser().strip(),
        Path.home().name.strip(),
    ):
        if len(candidate) > 3:
            tokens.append(candidate)
    return sorted(set(tokens), key=len, reverse=True)


def _sanitize_payload(value: Any, *, cb_root: Path, secret_tokens: list[str]) -> Any:
    if isinstance(value, str):
        text = relativize_package_paths(value)
        for root in sorted({str(cb_root), str(cb_root.resolve())}, key=len, reverse=True):
            text = text.replace(root + "/", "CooperBench/").replace(root, "CooperBench")
        for token in secret_tokens:
            text = text.replace(token, "[redacted]")
        return text
    if isinstance(value, list):
        return [_sanitize_payload(item, cb_root=cb_root, secret_tokens=secret_tokens) for item in value]
    if isinstance(value, dict):
        return {
            k: _sanitize_payload(v, cb_root=cb_root, secret_tokens=secret_tokens)
            for k, v in value.items()
        }
    return value


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
