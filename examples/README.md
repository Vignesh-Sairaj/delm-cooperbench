# Curated Slide-Aligned Trace Walkthroughs

These three trace bundles from the `50`-pair `flash` evaluation include only `conversation.json`, `result.json`, `eval.json`, and `agent*.patch` (omitting multi-megabyte model trajectories and SQLite databases).

| Directory | Arm (`thinking`) | Pair | Outcome | Slide Takeaway Illustrated |
| :--- | :--- | :--- | :--- | :--- |
| `flash_go_chi_26_f1_f2/` | `flash-ref-delm` (`medium`) | `go_chi_task/26` (1,2) | Clean merge, **passed**, `6` writes (`2` `[FACT]` notes), `3` waits, `173` steps, `15.1m`, `$1.89` | **Speedup & terse coordination**: `agent1` (`seq 18`) and `agent2` (`seq 26`) align around `h.ServeHTTP(w, r)` in `mux.go` with two `[FACT]` notes—finishing **2x faster** than push `ref-intents` (`30.8m`, `16` msgs, `$2.49`). |
| `flash_low_outlines_1706_f5_f8/` | `flash-ref-low-delm` (`low`) | `dottxt_ai_outlines_task/1706` (5,8) | Clean merge, **passed**, `21` writes (`7` `[FACT]` notes), **`56` waits** (`44` by `agent1`, `12` by `agent2`), `399` steps, `45.0m`, `$3.07` | **Low-thinking polling inflation**: On `low` thinking, blocking `read --wait` calls swell to `56` waits and `399` total steps vs. `13` waits, `193` steps, `19.7m`, and `$2.07` on `medium` (`flash-ref-delm`). |
| `3_flash_typst_6554_f5_f9/` | `3-flash-ref-low-delm` (`low`) | `typst_task/6554` (5,9) | **Merge conflict**, failed, `4` writes (`0` `[FACT]`/`[FAIL]` notes), `0` waits, `74` steps, `4.8m`, `$0.23` | **Smaller-model under-sharing**: Both workers edit `crates/typst/src/foundations/str.rs` without posting a single note, colliding at `git merge` while both `3.8-flash` DeLM and `3-flash` push (`25` msgs, `$0.74`) pass. |

For interactive side-by-side timeline scrubbing across `17` replays and `4` pairs (`go_chi_task/26`, `huggingface_datasets_task/3997`, `dottxt_ai_outlines_task/1706`, `typst_task/6554`), open [`../demos/delm_showcase.html`](../demos/delm_showcase.html) in a browser (rebuilt with `python -m delm_cb.cli showcase`).
