# delm-cooperbench

Zero-invasion evaluation of **DeLM** (`arXiv:2606.10662`, [`yuzhenmao/DeLM`](https://github.com/yuzhenmao/DeLM) @ `6661761`) as a pull-based shared-context coordination strategy on **CooperBench** (`arXiv:2601.13295`, [`cooperbench/CooperBench`](https://github.com/cooperbench/CooperBench) @ `63b9d44`) for Stanford CS329Z.

**Interactive Replay Studio (4 showcase pairs, 17 replays):** [Live GitHub Pages Demo](https://vignesh-sairaj.github.io/delm-cooperbench/demos/delm_showcase.html) | [`demos/delm_showcase.html`](demos/delm_showcase.html)

## Applying DeLM on CooperBench

- **Thin adapter (`src/delm_cb/`)**: Wraps DeLM's unmodified Python+stdlib board (`vendor/delm/board/`) with CooperBench's Docker runner via `@register("delm_msa_v2")`, leaving `CooperBench` completely unmodified.
- **Task Queue**: Pre-initialized and pre-claimed at `t=0` with both feature specifications (`Item 1` -> `agent1`, `Item 2` -> `agent2`), leaving the first `context.sh read` full.
- **Shared Context vs. Push Messaging**: Replaces CooperBench's push-based Redis queue (which pastes incoming peer messages directly into the next turn's input) with DeLM's pull-based `sh /tools/context.sh` board—agents choose when to read diffs since their last read, fetch full context (`read --all`), or block via `read --wait`.
- **Baseline settings & minimal diffs**: Runs with `--git` shared remote access, `128K` compaction trigger (`28K -> 128K`), and `500` step limit (`100 -> 500`): [`demos/cooperbench_agent_prompt.patch`](demos/cooperbench_agent_prompt.patch), [`demos/delm_board_prompt.patch`](demos/delm_board_prompt.patch), and [`demos/cooperbench_agent_adapter.patch`](demos/cooperbench_agent_adapter.patch).

## Findings (`50`-Pair Flash Subset)

| Model (thinking) | Coordination | Clean-Merge Pass | Conflicts | Pair Time Med (Mean) | Chatter (Msgs or Writes) | Cost |
| :--- | :--- | :---: | :---: | :---: | :--- | ---: |
| `3.8-flash (med)` | `Single-Agent (solo)`* | `37/50 (74%)` | `—` | `21.1m (27.3m)` | `— (91 compactions)` | `$65.46` |
| `3.8-flash (med)` | `CooperBench (push)`† | `34/50 (68%)` | `0` | `25.5m (25.9m)` | `1,396 msgs (120 compactions)` | `$102.06` |
| `3.8-flash (med)` | `+ Intent Tags (push)` | `35/50 (70%)`‡ | `1` | `19.1m (23.0m)` | `1,086 msgs (83 compactions)` | `$91.17` |
| `3.8-flash (med)` | **`+ DeLM (pull, ours)`** | **`34/50 (68%)`** | `0` | **`18.8m (21.1m)`** | `439 writes / 216 notes (11 compactions)` | **`$89.58`** |
| `3.8-flash (low)` | `CooperBench (push)` | `34/50 (68%)` | `0` | `10.0m (12.5m)` | `1,266 msgs (79 med steps/agent)` | `$48.96` |
| `3.8-flash (low)` | `+ DeLM (pull, ours)` | `30/50 (60%)` | `5` | `23.4m (29.1m)` | `385 writes / 1,026 waits (123 med steps)` | `$82.13` |
| `3-flash (low)` | `CooperBench (push)` | `16/50 (32%)` | `17` | `3.5m (4.3m)` | `895 msgs` | `$23.92` |
| `3-flash (low)` | `+ DeLM (pull, ours)` | `11/50 (22%)` | `29` | `3.5m (4.2m)` | `233 writes / only 23 notes` | `$17.10` |

*\*Sequential single agent implementing both features (no merge step). †Mean of 2 `coop --git` medium reference runs (`35/50` and `33/50`). ‡`36/50 (72%)` total passes including 1 solo-fallback pass on a merge conflict. Note: `writes` counts all agent board mutations (`note`, `status`, `done`, `apply`), whereas `notes` counts substantive `[FACT]`/`[FAIL]` shared-context entries.*

1. **Fastest & comparable accuracy**: On `3.8-flash` (`medium`), DeLM is the fastest method (`18.8m` median / `21.1m` mean pair time—faster than sequential `solo` and `25.5m` CooperBench push) at comparable accuracy (`34/50` clean-merge vs. `35/50` structured intent tags) and lowest multi-agent cost (`$89.58`).
2. **Terse coordination**: Pull sharing cuts chatter by `68.6%` (`439` writes / `216` notes vs. `1,396` msgs) and reduces `128K` compactions by `91%` (`11` vs. `120`).
3. **Failure modes**: Smaller model `3-flash` (`low`) barely writes notes (`23` notes across `50` pairs -> `29` merge conflicts); lower thinking `3.8-flash` (`low`) triggers repeated `read --wait` polling (`1,026` waits -> `123` median steps/agent, slowing pairs to `23.4m` and `$82.13`).

See [`results/delm_midpoint_comparison.md`](results/delm_midpoint_comparison.md) (all `160` evaluated pairs including `core`) and [`examples/README.md`](examples/README.md) (three curated `flash` traces).

## Setup & Reproduction

Repository layout: `src/delm_cb/` (adapter, host board server, prompts, CLI, reporter, showcase compiler), `vendor/delm/` (verbatim upstream `board/` and `prompts/`), `configs/` (`reference_{medium,low}.yaml`), `scripts/run_midpoint_suite.py`, `tests/test_delm_cb.py`, `results/`, `examples/`, and `demos/`.

```bash
export COOPERBENCH_ROOT=/path/to/CooperBench
export PYTHONPATH=src:vendor/delm

# Unit tests (no Docker) and full suite (includes live Docker containers)
"$COOPERBENCH_ROOT/.venv/bin/pytest" -q tests/test_delm_cb.py -k "not docker"
"$COOPERBENCH_ROOT/.venv/bin/pytest" -v tests/test_delm_cb.py

# Single pair or full 4-arm suite (--dry-run first)
"$COOPERBENCH_ROOT/.venv/bin/python" -m delm_cb.cli run -n smoketest-delm -r openai_tiktoken_task -t 0 -f 4,8 -m vertex_ai/gemini-3-flash-preview --agent-config configs/reference_low.yaml --key-slot 1
"$COOPERBENCH_ROOT/.venv/bin/python" scripts/run_midpoint_suite.py --dry-run

# Rebuild reports and interactive HTML showcase + diff patches
"$COOPERBENCH_ROOT/.venv/bin/python" -m delm_cb.cli report core-ref-delm
"$COOPERBENCH_ROOT/.venv/bin/python" -m delm_cb.cli showcase
```
