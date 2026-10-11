# autorlhf — module ownership & interfaces

Frozen contracts. Owners may not change another owner's signatures without the lead
agent updating this file first.

## Ownership

| Owner                     | Paths                                                                                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **lead**                  | `pyproject.toml`, `src/autorlhf/core/**`, `src/autorlhf/config/**`, `src/autorlhf/storage/**`, `src/autorlhf/llm/schemas.py`, `pytest.ini`-level config                        |
| **A — TUI**               | `src/autorlhf/cli.py`, `src/autorlhf/__main__.py`, `src/autorlhf/tui/**`, `tests/test_tui_*.py`                                                                                |
| **B — training**          | `src/autorlhf/training/**`, `tests/test_training_*.py`                                                                                                                         |
| **C — llm + pipeline**    | `src/autorlhf/llm/client.py`, `src/autorlhf/llm/jev.py`, `src/autorlhf/pipeline/**`, `tests/test_llm_*.py`, `tests/test_pipeline_*.py`                                         |
| **D — integrations/docs** | `src/autorlhf/integrations/**`, `src/autorlhf/doctor.py`, `src/autorlhf/setup_env.py`, `.env.example`, `README.md`, `docs/*` (except this file), `tests/test_integration_*.py` |
| **E — foundation tests**  | `tests/test_foundation_*.py`                                                                                                                                                   |

## Dependency direction

```
config/  core/  storage/          (lead, no upstream deps)
   ↑        ↑
llm/     training/                (C, B)
   ↑        ↑
pipeline/ ──┘                     (C)  — imports training.hardware + training.plan
   ↑
tui/  cli.py                      (A)  — imports pipeline + training.jobs
integrations/                     (D)  — imported *by* training/tui, never the reverse
```

`training/**` and `pipeline/**` must be importable **without** importing `textual`.
`training/hardware.py::detect_hardware` must import `torch` lazily (inside the
function) so the TUI can start in well under a second.

## Shared objects handed from lead → agents

- `autorlhf.core.enums` — `RunStatus` (+ `can_transition`), `TrainingMethod`,
  `TrainingObjective`, `FeedbackMode`, `SplitName`, `SampleStatus`, `CriterionId`,
  `PresetId`, `DecisionKind`.
- `autorlhf.core.types` — `ChatMessage`, `HardwareInfo`, `ResourceEstimate`,
  `GenerationRequest`, `GenerationResult`.
- `autorlhf.core.events` — `EventBus.emit(kind, message, *, run_id, phase, step,
fraction, level, verbose_only, **data)`. Kinds: `status|phase|progress|log|sample|
judgment|metric|warning|error|artifact|counters`.
- `autorlhf.core.errors` — typed exceptions (see file).
- `autorlhf.core.logging` — `configure_logging`, `get_logger`, redaction.
- `autorlhf.core.ids` — `content_hash`, `new_run_id`, `slugify`, `run_folder_name`.
- `autorlhf.config.*` — `AppPaths`, `Environment`, `AppSettings`, `SecretStore`,
  `CharacterSpec`, `SeedConversation`, `Rubric`, `Criterion`, `RunConfig`,
  `Hyperparameters`, `PRESETS`, `preset_feasibility`, `estimate_training_memory`,
  `recommend_preset`.
- `autorlhf.storage.*` — `RunStore` (SQLite), `RunPaths`, atomic IO helpers,
  `select_resume_checkpoint`, `is_checkpoint_complete`, `export_run`,
  `build_manifest`, `ResponseCache`.
- `autorlhf.llm.schemas` — Jev wire models + domain models (`Judgement`,
  `PairDecision`, `PreferencePair`, `CandidateResponse`, `SyntheticPrompt`,
  `EvalPrompt`, `EvalResponse`, `EvaluationReport`, `ComparisonReport`).

## Interfaces agents MUST provide

### B — `autorlhf.training`

```python
# hardware.py
def detect_hardware(*, with_bitsandbytes: bool = True) -> HardwareInfo
def estimate_resources(hw, hp: Hyperparameters, *, param_count: int | None,
                       gpu_count: int = 1) -> ResourceEstimate   # may delegate to config.presets

# models.py
@dataclass
class LoadedModel:
    model_id: str
    resolved_revision: str
    tokenizer: Any
    model: Any
    param_count: int
    chat_template_source: str   # "tokenizer" | "generic-fallback"
    warnings: list[str]

def load_tokenizer(cfg: HFModelConfig, *, hf_token: str | None, allow_fallback: bool | None = None) -> tuple[Any, list[str]]
def load_model(cfg: HFModelConfig, hp: Hyperparameters, *, hf_token: str | None, for_training: bool = False,
               device: str | None = None) -> LoadedModel
def build_chat_prompt(tokenizer, messages: list[ChatMessage]) -> str
def count_parameters(model) -> int
def model_param_count_from_metadata(cfg: HFModelConfig, *, hf_token: str | None) -> int | None   # metadata only, no weights

# lora.py
def discover_target_modules(model, *, preferred: list[str] | None = None) -> tuple[list[str], list[str]]
def validate_target_modules(model, targets: list[str]) -> None   # raises TargetModulesError
def infer_family(model) -> str                                   # "llama" | "qwen2" | "gpt2" | ...

# plan.py
class TrainingPlan(BaseModel):
    run_id: str
    objective: TrainingObjective
    method: TrainingMethod
    hyperparameters: Hyperparameters
    preset_id: str | None = None
    rationale: list[str] = []
    jev_ranking: dict | None = None          # {probabilities, confidence, chosen, model_build}
    resource_estimate: ResourceEstimate | None = None
    hardware: HardwareInfo
    sft_examples: list[dict] = []            # records from sft.jsonl
    preference_pairs: list[dict] = []        # records from preferences.jsonl
    train_count: int = 0
    validation_count: int = 0
    warnings: list[str] = []
    def to_json(self) -> str
    def save(self, path: Path) -> None

# jobs.py
class TrainingResult(BaseModel):
    run_id: str
    objective: TrainingObjective
    method: TrainingMethod
    status: str
    steps_completed: int = 0
    adapter_path: str | None = None
    checkpoints: list[str] = []
    resumed_from: str | None = None
    metrics: dict[str, float] = {}
    wandb_url: str | None = None
    error: str | None = None
    warnings: list[str] = []

class TrainingJob:
    def __init__(self, *, run_id: str, paths: RunPaths, config: RunConfig, plan: TrainingPlan,
                 bus: EventBus, cancel_event: threading.Event, pause_event: threading.Event,
                 env: Environment) -> None
    def run(self, *, resume_from: Path | None = None) -> TrainingResult
    def request_pause(self) -> None
    def request_resume(self) -> None
    def request_stop(self) -> None
    @property
    def status(self) -> dict[str, Any]

def find_resume_checkpoint(paths: RunPaths, *, objective: TrainingObjective) -> Path | None
```

`TrainingJob.run()` MUST: emit `metric`/`progress`/`sample`/`warning` events via `bus`;
honour `pause_event` at step boundaries and `cancel_event` for graceful stop; call
`mark_checkpoint_complete()` for each saved checkpoint after `trainer.save_model`;
persist metrics to SQLite via `RunStore` (accept an optional `store: RunStore | None`);
return a `TrainingResult` even when interrupted (never raise for a user-requested stop).

### C — `autorlhf.llm` + `autorlhf.pipeline`

```python
# llm/client.py
class OpenRouterClient:
    def __init__(self, api_key: str, *, timeout_s: float = 120.0, max_retries: int = 4,
                 max_concurrency: int = 4, cache: ResponseCache | None = None,
                 run_id: str = "", app_title: str = "autorlhf") -> None
    def chat(self, *, model: str, messages: list[ChatMessage] | list[dict], temperature: float = 0.8,
             top_p: float = 0.95, max_tokens: int = 512, seed: int | None = None,
             response_format: dict | None = None) -> GenerationResult
    def chat_json(self, *, model: str, messages, schema_hint: str, ...) -> tuple[dict, GenerationResult]
    def decide(self, request: DecisionsRequest) -> DecisionsResponse      # POST /api/alpha/decisions
    def decide_many(self, requests, *, on_progress=None) -> list[DecisionsResponse]   # bounded concurrency
    def verify_credentials(self) -> tuple[bool, str]      # cheap live check
    def usage_summary(self) -> dict[str, float]
    def close(self) -> None
ALL network errors raise autorlhf.core.errors.RemoteAPIError / RateLimitedError; never return "" silently.

# llm/jev.py
class JevEvaluator:
    def __init__(self, client: OpenRouterClient, rubric: Rubric, character: CharacterSpec,
                 *, judge_model: str, cache: ResponseCache | None = None, run_id: str = "",
                 max_retries: int = 3) -> None
    def score_response(self, *, prompt_id: str, candidate: CandidateResponse, prompt: str,
                       context: list[dict[str, str]] = ()) -> Judgement
    def compare_responses(self, *, prompt_id: str, option_a: CandidateResponse, option_b: CandidateResponse,
                          prompt: str, context=(), overall_a: float | None = None,
                          overall_b: float | None = None) -> PairDecision
    def rank_presets(self, *, state: dict, options: dict[str, str]) -> tuple[str | None, dict]
    def check_condition(self, *, instructions: str, state: dict, key: str = "condition") -> tuple[float, dict]

# pipeline/context.py
class PipelineContext:
    def __init__(self, *, env: Environment, store: RunStore, paths: RunPaths, config: RunConfig,
                 bus: EventBus, interactive: bool = False, verbose: bool = False) -> None
    # attributes: env, store, paths, config, bus, interactive, verbose, run_id, hf_token,
    #             openrouter (OpenRouterClient, lazy), cache (ResponseCache), jev (JevEvaluator, lazy),
    #             cancel_event, pause_event, review_queue (ReviewQueue)
    def emit(self, kind, message, **kw) -> Event
    def cancelled(self) -> bool
    def wait_if_paused(self) -> None
    def close(self) -> None

# pipeline/models.py
class DatasetBundle(BaseModel):
    run_id: str
    version: str
    dataset_hash: str
    split_hash: str
    prompts: list[SyntheticPrompt]
    candidates: list[CandidateResponse]
    judgements: dict[str, Judgement]            # candidate_id -> Judgement
    pair_decisions: list[PairDecision]
    preference_pairs: list[PreferencePair]
    train_prompt_ids: list[str]
    validation_prompt_ids: list[str]
    eval_prompt_ids: list[str]
    counters: dict[str, int]
    warnings: list[str] = []
    def to_summary(self) -> dict[str, Any]

class ReviewItem(BaseModel):
    item_id: str                      # == prompt_id
    prompt_id: str
    prompt: str
    context: list[dict[str, str]]
    category: str
    candidates: list[CandidateResponse]
    judgements: dict[str, Judgement]
    comparison: PairDecision | None
    recommended_candidate_id: str | None
    recommendation_reasons: list[str]
    human: ReviewDecision | None = None
    status: str = "pending"

class ReviewDecision(BaseModel):
    item_id: str
    reviewer: str = "human"
    action: Literal["accept","choose_a","choose_b","tie","reject_both","edit","regenerate"] = "accept"
    chosen_id: str | None = None
    rejected_id: str | None = None
    scores: dict[str, dict[str, float]] = {}       # candidate_id -> criterion_id -> score
    edited_texts: dict[str, str] = {}              # candidate_id -> replacement text
    comments: str = ""
    use_chosen_as_sft: bool = True
    created_at: str

# pipeline/review.py
class ReviewQueue:
    def __init__(self, store: RunStore, bus: EventBus, run_id: str) -> None
    def submit_items(self, items: list[ReviewItem]) -> None       # persist + emit bus "sample" events
    def pending(self) -> list[ReviewItem]
    def load(self, item_id: str) -> ReviewItem | None
    def decide(self, decision: ReviewDecision) -> None             # called from the TUI thread
    def wait_for_decision(self, item_id: str, *, cancel_event: threading.Event,
                          timeout: float | None = None) -> ReviewDecision | None
    def stats(self) -> dict[str, int]

# pipeline/orchestrate.py
class PipelineRunner:
    def __init__(self, ctx: PipelineContext) -> None
    def preflight(self) -> dict[str, Any]
    def prepare_dataset(self, *, resume: bool = True) -> DatasetBundle
    def run_review(self, bundle: DatasetBundle) -> DatasetBundle          # automated or interactive
    def plan_training(self, bundle: DatasetBundle) -> TrainingPlan
    def run_baseline_eval(self) -> EvaluationReport
    def run_final_eval(self, *, adapter_path: Path | None) -> tuple[EvaluationReport, ComparisonReport]
    def build_training_plan(self, bundle, *, adapter_path=None) -> TrainingPlan
```

Each runner method emits phase/progress events, persists artifacts, updates counters
via `RunStore`, and writes datasets through `autorlhf.storage.runs` atomic helpers.

### A — `autorlhf.cli`

```python
def main(argv: list[str] | None = None) -> int
# autorlhf            -> dashboard if runs exist else first-run setup, then workflow
# autorlhf -i|--interactive
# autorlhf --doctor
# autorlhf --setup            (headless credential wizard, masked input)
# autorlhf --headless [--run-id ID] [--resume]   (pipeline without Textual, for CI/smoke)
# autorlhf --list-runs / --export RUN_ID DEST / --version
```

The TUI worker thread drives: `prepare_dataset → run_review → run_baseline_eval →
TrainingJob.run() → run_final_eval`, reacting to UI pause/stop by setting
`ctx.pause_event` / `ctx.cancel_event` (and forwarding to the live `TrainingJob`).
Bus events are enqueued onto the Textual loop asynchronously; metric tiles update
immediately, while state-panel reads are coalesced and run in a background thread.
The elapsed timer runs on its own thread and schedules display updates on the Textual
loop. Widget mutation remains on the Textual loop.

## Non-negotiables

- No credential is ever written to SQLite, a log, a dataset, or an export.
- No `except Exception: pass` around a network call that silently yields empty data.
- `gettext`/prose strings are plain; no emoji in source.
- Python: `from __future__ import annotations`, type hints on public functions,
  `ruff` line length 100.
