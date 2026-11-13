# delm-cooperbench

Zero-invasion evaluation of **DeLM: Decentralized Multi-Agent Systems with Shared Context** (`arXiv:2606.10662`, [`yuzhenmao/DeLM`](https://github.com/yuzhenmao/DeLM) @ `6661761`) as a pull-based coordination strategy on **CooperBench** (`arXiv:2601.13295`, [`cooperbench/CooperBench`](https://github.com/cooperbench/CooperBench) @ `63b9d44` with Stage-4 `500`-step / `128K`-compaction reference configs).

## Overview & Architecture

This package runs DeLM's unmodified shared-context board (`vendor/delm/board/`) alongside CooperBench's Docker agent runtime and 3-way `git merge` evaluator without modifying the `CooperBench` repository:

- **Plugin registration (`@register("delm_msa_v2")`)**: `src/delm_cb/adapter.py` subclasses CooperBench's `DefaultAgent` (`DelmDefaultAgent`, preserving local unpushed/unsubmitted PR nudges without Redis) and registers `DelmMiniSweAgentV2Runner` at runtime.
- **Channel isolation (`--git` + `--no-messaging`)**: Runs execute with CooperBench's shared `cooperbench-git` remote (`--git`) and Redis push messaging disabled (`messaging_enabled=False`) so `sh /tools/context.sh` is the sole inter-agent coordination channel. `ProtectedBoard.apply` stays enabled alongside 3-way `git merge` grading.
- **Host-side pair board & pre-seeded queue (`src/delm_cb/board_runtime.py`)**: Each task pair gets an isolated `ProtectedBoard` SQLite database and HTTP server (`0.0.0.0:0`, exposed inside containers via `--add-host=host.docker.internal:host-gateway`, per-actor 32-hex `CONTEXT_TOKEN`, and read-only `/tools` + per-agent writable `/records` mounts). `OriginalFiles` reads base-image files lazily via `docker cp` (handling Docker 29 `"No such container:path"` / `"Could not find the file"` stderr for newly created files). At `t=0`, both features are pre-seeded and pre-claimed on the queue (`Item 1` -> `agent1`, `Item 2` -> `agent2`) **without** appending `read` events, so each agent's first `sh /tools/context.sh read` returns `full=True`.
- **Stage-4 parity (`configs/`, `src/delm_cb/cli.py`, `src/delm_cb/report.py`)**: Mirrors CooperBench's frozen Stage-4 reference settings (`500` steps/agent, `128K` compaction trigger, `step_counter: true`, `$40` per-agent cost limit, `10h` container timeout, `4`-shard `KEY_SLOT=1..4` orchestration at `-c 2`).

## Repository Layout

- `src/delm_cb/`: Board server lifecycle (`board_runtime.py`), DeLM-adapted prompt templates (`prompts.py`), CooperBench adapter (`adapter.py`), CLI and 4-shard suite runner (`cli.py`), and CSV/Markdown comparison exporter (`report.py`).
- `vendor/delm/board/`: Unmodified DeLM shared-context board (`shared_context.py`, `protected_board.py`, `context_client.py`, `context.sh`, `__init__.py`), plus `vendor/delm/LICENSE` and `vendor/delm/UPSTREAM.md`.
- `configs/`: Stage-4 `reference_medium.yaml` and `reference_low.yaml`.
- `scripts/run_midpoint_suite.py`: Convenience entrypoint for running all 4 Stage-4 DeLM arms.
- `tests/test_delm_cb.py`: Unit tests and live two-container Docker integration tests (`openai_tiktoken_task/0` and `go_chi_task/27`).
- `results/`: Exported per-arm `run.json`, `pairs.csv`, `agents.csv`, `summary.md`, and `results/delm_midpoint_comparison.md`.
- `examples/`: Three self-contained curated pair trace walkthroughs (`examples/README.md`).

## Setup & Usage

Set `COOPERBENCH_ROOT` if your `CooperBench` checkout is not in a sibling directory (`../CooperBench` or `../../CooperBench`), and reuse CooperBench's Python environment (which provides `cooperbench`, `litellm`, `pytest`, and `ruff`):

```bash
export COOPERBENCH_ROOT=/path/to/CooperBench
export PYTHONPATH=src:vendor/delm
```

### Run Tests

```bash
# Fast unit and reporting tests (no Docker required)
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/pytest" -q tests/test_delm_cb.py -k "not docker"

# Full suite including live two-container Docker board integration tests (Python + Go images)
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/pytest" -v tests/test_delm_cb.py
```

### Run a Single Pair, the 4-Arm Suite, and Reports

```bash
# Single pair
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/python" -m delm_cb.cli run \
  -n smoketest-delm -r openai_tiktoken_task -t 0 -f 4,8 \
  -m vertex_ai/gemini-3-flash-preview --agent-config configs/reference_low.yaml --key-slot 1

# Dry-run or execute the 4-arm Stage-4 suite (4 shards x -c 2)
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/python" scripts/run_midpoint_suite.py --dry-run
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/python" scripts/run_midpoint_suite.py

# Re-export metrics and regenerate results/delm_midpoint_comparison.md
PYTHONPATH=src:vendor/delm "$COOPERBENCH_ROOT/.venv/bin/python" -m delm_cb.cli report core-ref-delm
```

## Results (`160` Evaluated Pairs)

All 4 Stage-4 Midpoint arms completed with `0` agent errors and `0` board CLI errors (`results/delm_midpoint_comparison.md`):

| Subset | Model (`thinking`) | Baseline (`coop --git`) | DeLM Arm | DeLM Passed (Clean-Merge) | Conflicts | Writes vs. Baseline Msgs | Cost (Wall Time) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | ---: |
| `core` (`10`) | `gemini-3.8-flash` (`medium`) | `5/10` (`50%`, `$21.62`, `110m`) | `core-ref-delm` | **6/10 (60%)** | `0` | `99` vs. `307` (`-67.8%`) | `$18.57` (`50m`) |
| `flash` (`50`) | `gemini-3.8-flash` (`medium`) | `35/50` `#1`, `33/50` `#2` (`68%` mean, `$102.06`) | `flash-ref-delm` | **34/50 (68%)** | `0` | `439` vs. `1,396` mean (`-68.6%`) | `$89.58` (`181m`) |
| `flash` (`50`) | `gemini-3.8-flash` (`low`) | `34/50` (`68%`, `0` conflicts, `$48.96`, `170m`) | `flash-ref-low-delm` | `31/50` (`30/50` = `60%` clean) | `5` | `385` vs. `1,266` (`-69.6%`) | `$82.13` (`207m`) |
| `flash` (`50`) | `gemini-3-flash-preview` (`low`) | `17/50` (`16/50` = `32%` clean, `17` conflicts, `$23.92`) | `3-flash-ref-low-delm` | `16/50` (`11/50` = `22%` clean) | `29` | `233` vs. `895` (`-74.0%`) | `$17.10` (`32m`) |

### Key Empirical Findings

1. **Higher coordination efficiency at `medium` thinking**: With `gemini-3.8-flash` (`medium`), DeLM matches or outperforms push-based `coop --git` baselines (`6/10` on `core`, `+10pp` over `coop --git` ref and `ref-intents`; `34/50` clean-merge on `flash`, matching the `34.0/50` `coop --git` mean and `ref-delivery-timing`, `1` clean pair behind `ref-intents` at `35/50`) while cutting board/message writes by `68–69%`, compactions by `86–91%`, reasoning tokens by `54.6%` on `flash`, and total cost by `12.2–14.1%`.
2. **Pull polling and note sparsity degrade weaker / `low`-thinking runs**: At `low` thinking, `gemini-3.8-flash` nearly doubled blocking `read --wait` calls (`1,026` vs. `523` on `medium`, driving steps up from `9,003` to `12,627` and cost from `$48.96` to `$82.13`), while `gemini-3-flash-preview` (`low`) posted only `23` notes across `50` pairs (vs. `216` on `3.8-flash` `medium`), causing `29` git merge conflicts. Across all `160` pairs, `context.sh apply` was invoked only `2` times total.
3. **Single-line co-edit limits persist across push and pull protocols**: On the `39` non-Click `flash` pairs, `flash-ref-delm` (`medium`) scored `28/39` (`71.8%`), beating `solo` (`27/39`) and every `coop` arm (`25–27/39`). On `pallets_click_task` (`11` pairs), `flash-ref-delm` dropped to `6/11` — reproducing the same `ed.process_priority` post-construction workaround via DeLM `[FACT]` note (`sequence 30` in `examples/flash_pallets_click_2068_f6_f7/`) that passes `git merge` cleanly while breaking `Editor.__init__` mock assertions.

See [`results/delm_midpoint_comparison.md`](results/delm_midpoint_comparison.md) for the full 4-arm comparison and paired win/loss breakdown, and [`examples/README.md`](examples/README.md) for annotated trace walkthroughs.
