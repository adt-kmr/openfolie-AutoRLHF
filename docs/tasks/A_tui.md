# Agent A — TUI and application flow

Read `docs/ARCHITECTURE.md` (frozen interfaces) and `docs/UPSTREAM_API_NOTES.md` (verified
library APIs) before writing code. You own exactly these paths:

- `src/autorlhf/cli.py`
- `src/autorlhf/__main__.py`
- `src/autorlhf/tui/**`
- `tests/test_tui_*.py`

Do not edit any other file. Read anything you need.

## What to build

A focused developer tool, not a toy. Information density, clear status, keyboard-first.
Use Textual's native widgets: `DataTable`, `TabbedContent`, `Input`, `TextArea`, `Select`,
`RadioSet`, `ProgressBar`, `RichLog`, `Footer`, `ModalScreen`, `ListView`, `Collapsible`.

### CLI (`cli.py`)

```
autorlhf                      dashboard if runs exist, else first-run setup, then workflow
autorlhf -i | --interactive   same flow with mandatory human review
autorlhf --doctor             environment/dependency diagnostic (calls autorlhf.doctor)
autorlhf --setup              headless credential wizard (calls autorlhf.setup_env)
autorlhf --headless [--run-id ID] [--resume] [--config PATH]
autorlhf --list-runs
autorlhf --export RUN_ID DEST [--with-checkpoints]
autorlhf --version
```

`main(argv: list[str] | None = None) -> int` returns a process exit code; use argparse with
a concise `--help`. `--headless` runs the pipeline through `PipelineRunner` +
`TrainingJob` with `EventBus.print_listener()` and no Textual import at module import time
(import Textual lazily so `--headless`/`--doctor` work on a headless box).

### TUI structure

`tui/app.py` — `AutorlhfApp(App)`. Owns: `Environment`, `RunStore`, `EventBus`, current
`RunPaths`/`RunConfig`, verbosity flag, the worker, and the cancellation plumbing
(`ctx.cancel_event`, `ctx.pause_event`). Screens navigate via a screen stack; `Ctrl+C`
and `q` request graceful stop, never leave persisted state corrupt.

Screens (each in `tui/screens/`):

1. `dashboard.py` — run table (id, created, model, objective/method, status, step, last
   checkpoint) + incomplete-run panel offering **inspect / resume / restart / abandon**.
   Empty state routes to setup.
2. `setup.py` — first-run flow: credentials (masked `Input(password=True)`, never echoed,
   never logged) → model selection → character → rubric → generation settings.
   Offers the W&B / Weave opt-in with the **explicit data-egress notice before enabling
   Weave**.
3. `model_config.py` — HF model id, revision, HF token presence indicator (never the
   value), dtype/attn options, `allow_generic_chat_template` with an explanation that it
   is a fallback. Shows model metadata (`AutoConfig`/`tokenizer` presence, parameter count
   from metadata) without downloading weights.
4. `character.py` — system prompt (TextArea), behavioural constraints list, lore, and
   seed conversations with add / edit / delete.
5. `rubric.py` — the six criteria, editable level texts, weights, tie epsilon, min judge
   confidence.
6. `generation.py` — teacher model, judge model, prompt/candidate counts, source
   (teacher/local/mixed), candidate counts, split fractions, edge-case fraction,
   concurrency/timeouts/retries, seed, verbosity.
7. `training_config.py` — hardware panel (measured vs estimated clearly labelled), the
   recommended preset with **its actual resolved values**, alternate feasible presets,
   Jev's ranking (probabilities + confidence, plus an app-authored rationale — never
   label them as chances of training success), infeasible options disabled with the
   specific reason, and manual editing of numeric hyperparameters and target modules.
   Validate the whole config before allowing a start.
8. `review.py` — interactive review queue. Per item: character summary, user prompt and
   context, each candidate's text, Jev's per-criterion scores, preferred response with
   probabilities/confidence. Human can score each response on every criterion, choose
   A/B/tie, reject, edit text, or regenerate. Jev's recommendation stays visible but is
   never auto-applied. Progress is saved to SQLite as items are decided.
9. `dataset.py` — inspect and edit the generated dataset: table of prompt/candidate/
   status/score, split counts, filters, inline edit of a response, reject/edit actions,
   validation warnings (empty responses, duplicates, malformed conversations, invalid
   preferences, inconsistent prompt IDs, truncation). Edits persist to `dataset.jsonl`.
10. `training.py` — live view: run id, phase, step progress, train/eval loss, LR, elapsed,
    throughput, dataset/preference counts, latest samples with Jev's judgements,
    warnings/retries/errors, checkpoint state and last checkpoint path. Verbosity toggle
    (normal = phase summaries + key samples; verbose = full prompts/responses/rubric
    results/decisions). Controls: pause at a safe boundary, resume, stop gracefully.
11. `evaluation.py` — the base-vs-fine-tuned comparison table (criterion, base,
    fine-tuned, change, counts, sample size), per-criterion deltas, pairwise win/tie
    counts, warnings when coverage is too small or judgements failed, and representative
    base/tuned response pairs with their judgements.
12. `settings.py` — edit credentials + W&B/Weave toggles later without redoing setup.

`tui/widgets/` — reusable pieces (metric tiles, score bars showing the judge's
probabilities, key-value panels, masked-secret field, log pane). `tui/styles.tcss` for
layout.

### Worker integration

Run the pipeline in a Textual thread worker (`@work(thread=True, exclusive=True)`) or a
plain `threading.Thread`; forward everything through `App.call_from_thread`. Subscribe to
the `EventBus` once and dispatch by `event.kind`. Never touch widgets from the worker
thread. Never let an exception in a worker leave the UI wedged: catch, show it, and
persist a `failed` status.

The worker sequence is exactly:

```python
ctx = PipelineContext(env=..., store=..., paths=..., config=..., bus=..., interactive=flag, verbose=...)
runner = PipelineRunner(ctx)
bundle = runner.prepare_dataset()
bundle = runner.run_review(bundle)          # interactive mode blocks in ReviewQueue
baseline = runner.run_baseline_eval()
plan = runner.build_training_plan(bundle, adapter_path=None)
job = TrainingJob(run_id=..., paths=..., config=..., plan=plan, bus=..., env=...,
                  cancel_event=ctx.cancel_event, pause_event=ctx.pause_event)
result = job.run(resume_from=find_resume_checkpoint(paths, objective=plan.objective))
final, comparison = runner.run_final_eval(adapter_path=Path(result.adapter_path) if result.adapter_path else None)
```

Keep a reference to the live `TrainingJob` so the pause/stop buttons and the
interactive-mode `regenerate` action work.

## Tests you own

`tests/test_tui_*.py` using `App.run_test()` (pytest-asyncio, `asyncio_mode = "auto"`):
dashboard with and without runs, first-run setup path, the model/character/rubric forms
(including add/delete of seed conversations), the review screen applying a human override,
the training screen rendering an event stream, and the evaluation table rendering a
`ComparisonReport`. Keep each test under ~2 s and use a temp `AppPaths` (set
`AUTORHLF_HOME` / `AUTORHLF_CONFIG_HOME` env vars) plus a `RunStore` on a temp db. No
network, no model downloads. Also add one test that `--doctor` and `--help` exit 0 via
`main([...])`.

You must also verify manually: `uv run autorlhf --help`, `uv run autorlhf --doctor`,
`uv run autorlhf --headless --config <tiny fixture>` if the pipeline exists yet, and
`uv run autorlhf` rendering the dashboard (it is fine to drive it headlessly via
`App.run_test()` if there is no TTY).

## Definition of done

- `uv run autorlhf --help`, `--doctor`, `--list-runs`, `--version` work.
- Every screen in the list above exists and renders without tracebacks.
- Interactive mode truly blocks on the review queue and records human decisions.
- `uv run pytest tests/test_tui_*.py` passes.
- `uv run ruff check src/autorlhf/tui src/autorlhf/cli.py` is clean.