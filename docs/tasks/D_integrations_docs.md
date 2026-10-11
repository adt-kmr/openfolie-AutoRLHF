# Agent D — integrations, doctor, setup wizard, docs

Read `docs/ARCHITECTURE.md`. You own exactly:

- `src/autorlhf/integrations/**`
- `src/autorlhf/doctor.py`
- `src/autorlhf/setup_env.py`
- `.env.example`
- `README.md`
- `docs/USAGE.md`
- `tests/test_integration_*.py`

Do not edit any other file.

## `integrations/tracking.py`

Optional W&B (training metrics) and Weave (LLM/evaluation traces), each independently
toggleable, neither ever required.

```python
class TrackingState(BaseModel):
    wandb_enabled: bool = False
    weave_enabled: bool = False
    wandb_project: str = ""
    weave_project: str = ""
    wandb_url: str | None = None
    warnings: list[str] = []

class TrainingTracker:
    def __init__(self, *, config: RunConfig, env: Environment, run_id: str,
                 bus: EventBus | None = None) -> None
    @property
    def enabled(self) -> bool
    def start(self) -> TrackingState        # never raises; degrades with a warning
    def log(self, metrics: dict[str, float], *, step: int) -> None
    def finish(self, *, summary: dict | None = None) -> None
    @property
    def report_to(self) -> list[str]       # [] | ["wandb"]

def weave_client(env: Environment, *, config: RunConfig | None = None) -> Any | None
def weave_trace(env, *, config, name: str) -> contextlib.AbstractContextManager
def weave_log(name: str, payload: dict) -> None
```

Requirements:

- Import `wandb`/`weave` **lazily** inside the functions, so the app starts without them.
- `start()` re-raises nothing: authentication/connectivity failures become a `warning`
  event on the bus and a `warnings` entry, and training continues locally.
- `report_to` returns `["wandb"]` only when W&B actually initialised, so TRL never tries to
  reach a tracker that is disabled.
- Both are disabled by default and inert without credentials. A dedicated test must prove
  the whole pipeline is unaffected when both are off.
- **Weave requires explicit consent** (`RunConfig.weave_consent`): `weave_client` returns
  `None` and records a warning when consent is absent, and the warning must state that
  prompts, responses and judgements would be transmitted to an external service.
- Weave tracing is instrumented **explicitly** at the call sites agent C owns
  (`weave_log`/`weave_trace`); do not assume automatic patching of the OpenRouter SDK works.
- No credential is ever logged, printed, or written into SQLite or an export.
- Add a `weave_op`-style helper only if the installed `weave` API needs it — verify against
  the installed version rather than assuming.

## `doctor.py`

```python
class DoctorCheck(BaseModel):
    name: str
    status: str          # "ok" | "warn" | "fail"
    detail: str
    remediation: str = ""

class DoctorReport(BaseModel):
    checks: list[DoctorCheck] = []
    @property
    def worst(self) -> str
    def render(self) -> str        # plain-text report for the CLI

def run_doctor(*, env: Environment, live: bool = False) -> DoctorReport
```

Checks: Python version; `uv`/venv presence; core imports (textual, pydantic, httpx);
training stack imports with versions (torch, transformers, trl, peft, datasets, accelerate);
optional extras (bitsandbytes, wandb, weave) present-but-off; hardware (delegate to
`autorlhf.training.hardware.detect_hardware` inside a try/except so a broken torch install
reports instead of crashing); writability of the config/data dirs; secret file permissions
(warn if not `0600` on POSIX); which credentials are present (report the _masked_ form via
`Environment.secrets.describe()`, never a value); and, when `live=True`, an OpenRouter
credential check via `OpenRouterClient.verify_credentials()` plus a HF token check. Every
failure must carry a `remediation` string. `run_doctor` must never raise.

The CLI (agent A) calls `run_doctor(env=..., live=live)`; print `report.render()` and use
`report.worst` for the exit code (`fail` → 1).

## `setup_env.py`

```python
def interactive_setup(*, env: Environment, ask_wandb: bool = True, force: bool = False) -> AppSettings
def prompt_secret(label: str, *, allow_empty: bool = True) -> str   # masked via getpass
def prompt_yes_no(label: str, *, default: bool = False) -> bool
def configure_wandb(env: Environment) -> None
def configure_weave(env: Environment, *, project: str = "") -> bool   # returns consent
```

- Masked input (`getpass`), never echoed, never logged.
- Validate the OpenRouter key with a minimal live request when a key is entered, and show
  an actionable error on failure (`doctor`-style remediation text) while still allowing the
  user to continue without one.
- Updating one credential must preserve every other line in the secret file (the
  `SecretStore.update` implementation already guarantees this; use it).
- Before enabling Weave, print the explicit notice that prompts, responses and evaluation
  traces may be transmitted to an external service, and require a yes/no answer. Record
  `weave_consent` on the returned settings and warn when declining.
- Write with `0600` permissions on POSIX (handled by `SecretStore`).
- Return the updated `AppSettings`.

## `.env.example`

Placeholders only, exactly the keys in `autorlhf.config.settings.KNOWN_ENV_KEYS`:
`OPENROUTER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`,
`WEAVE_ENABLED`, `WEAVE_PROJECT`, `WANDB_ENABLED`. Add a header comment saying the real
file is `~/.local/share/autorlhf/config.env` (or `$AUTORHLF_HOME/config.env`), that a
project-local `.env` is read but never written, and that `.env` is gitignored.

## `README.md` (full rewrite) and `docs/USAGE.md`

`README.md`: what autorlhf is; the pipeline diagram; install (`uv sync`, optional extras);
credentials; the exact commands (`autorlhf`, `autorlhf -i`, `--doctor`, `--setup`,
`--headless`, `--list-runs`, `--export`); the artifact layout; how to train, resume, and
inspect a run; how base-vs-fine-tuned evaluation works; hardware requirements per method
(LoRA / QLoRA / FFT) and what is gated on what; the honest statement that Jev is AI
feedback (RLAIF) and that human review in `-i` adds real human feedback; that a Jev score
probability is a preference judgement, **not** a probability of training success; privacy
notes for W&B/Weave; and a "what is not implemented" section (write it once the other
agents land — read their code first, do not guess).

`docs/USAGE.md`: a walkthrough of one full run plus a resume walkthrough and a
troubleshooting table mapping each `autorlhf.core.errors` class to its likely cause and
fix. Read `src/autorlhf/core/errors.py` for the list.

## Tests you own (`tests/test_integration_*.py`)

- W&B and Weave both disabled: `TrainingTracker.start()` reports
  `report_to == []`, no import of wandb/weave happens (assert with
  `sys.modules`), and a training-shaped event flow is unaffected.
- `start()` with W&B enabled but `wandb.init` raising → a warning is recorded and no
  exception escapes.
- Weave without consent → `weave_client` returns `None` and the warning mentions data
  egress.
- `run_doctor` returns a report with the expected check names, never raises, and never
  contains a credential value (construct an `Environment` with a fake key and assert the
  key string is absent from `report.render()`).
- `SecretStore` round-trip: `interactive_setup`-style updates preserve an unrelated line
  and the file mode is `0600` on POSIX.
- `render_env_text` preserves comments and unknown keys.

## Definition of done

- `uv run autorlhf --doctor` prints a readable report and exits 0 on this machine.
- `uv run pytest tests/test_integration_*.py` passes.
- `README.md` matches the code that actually exists (check before you write).
- `uv run ruff check src/autorlhf/integrations src/autorlhf/doctor.py src/autorlhf/setup_env.py` is clean.
