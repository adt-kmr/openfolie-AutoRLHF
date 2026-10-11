"""Command line entry point.

Textual is imported lazily inside :func:`_run_tui` so ``--help``, ``--version``,
``--doctor``, ``--list-runs``, ``--export`` and ``--headless`` all work on a box
with no TTY and no Textual installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autorlhf import __version__
from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.errors import AutorlhfError
from autorlhf.core.events import EventBus
from autorlhf.core.ids import new_run_id, run_folder_name
from autorlhf.core.logging import configure_logging
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths, export_run, load_config, save_config

EPILOG = """\
exit codes: 0 success, 1 a check or run failed, 2 the command could not start.

examples:
  autorlhf                      open the dashboard (or first-run setup)
  autorlhf -i                   same, but every comparison needs your decision
  autorlhf --doctor             environment and dependency diagnostics
  autorlhf --setup              credential wizard in the terminal
  autorlhf --headless --config run.json
  autorlhf --list-runs
  autorlhf --export 20260101-120000-ab12 ./bundle
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autorlhf",
        description=(
            "Local-first workbench that turns a character specification into a "
            "fine-tuned Hugging Face causal LM, with Jev as the judge."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"autorlhf {__version__}")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="require a human decision for every comparison",
    )
    parser.add_argument("--doctor", action="store_true", help="run environment diagnostics")
    parser.add_argument(
        "--live",
        action="store_true",
        help="with --doctor: also verify live credentials",
    )
    parser.add_argument("--setup", action="store_true", help="headless credential wizard")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the pipeline without the TUI",
    )
    parser.add_argument("--run-id", default=None, help="run to resume or inspect")
    parser.add_argument("--resume", action="store_true", help="resume the given run")
    parser.add_argument("--config", default=None, help="RunConfig JSON for --headless")
    parser.add_argument("--list-runs", action="store_true", help="list stored runs and exit")
    parser.add_argument(
        "--export",
        nargs=2,
        metavar=("RUN_ID", "DEST"),
        default=None,
        help="copy a credential-free run bundle to DEST",
    )
    parser.add_argument(
        "--with-checkpoints",
        action="store_true",
        help="with --export: include checkpoints",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch; returns the process exit code."""

    parser = build_parser()
    try:
        args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    except SystemExit as exc:  # --help/--version/usage error already printed
        return int(exc.code or 0)

    configure_logging(verbose=args.verbose)

    if args.doctor:
        return _doctor(live=args.live)
    if args.setup:
        return _setup()
    if args.list_runs:
        return _list_runs()
    if args.export:
        return _export(args.export[0], args.export[1], with_checkpoints=args.with_checkpoints)
    if args.headless:
        return _headless(args)
    return _run_tui(interactive=args.interactive, verbose=args.verbose)


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #


def _doctor(*, live: bool) -> int:
    env = Environment()
    try:
        from autorlhf.doctor import run_doctor
    except ImportError as exc:
        print(f"autorlhf doctor is unavailable: {exc}", file=sys.stderr)
        return 2
    try:
        report = run_doctor(env=env, live=live)
    except Exception as exc:  # noqa: BLE001 - the CLI must still exit cleanly
        print(f"autorlhf doctor failed to run: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    return 1 if report.worst == "fail" else 0


def _setup() -> int:
    env = Environment()
    try:
        from autorlhf.setup_env import interactive_setup
    except ImportError as exc:
        print(f"the setup wizard is unavailable: {exc}", file=sys.stderr)
        return 2
    try:
        interactive_setup(env=env)
    except KeyboardInterrupt:
        print("\nsetup cancelled; nothing was written for the interrupted prompt")
        return 130
    return 0


def _list_runs() -> int:
    env = Environment()
    store = RunStore(env.paths.db_path)
    try:
        records = store.list_runs(limit=200)
    finally:
        store.close()
    if not records:
        print("no runs yet. Start one with `autorlhf`.")
        return 0
    header = ("RUN", "CREATED", "STATUS", "OBJ/METHOD", "STEP", "CHECKPOINT", "MODEL")
    rows = [
        (
            record.id,
            record.created_at,
            record.status.value,
            f"{record.objective.value}/{record.method.value}",
            str(record.step),
            record.last_checkpoint or "-",
            record.model_id,
        )
        for record in records
    ]
    widths = [max(len(str(line[i])) for line in (header, *rows)) for i in range(len(header))]
    pattern = "  ".join(f"{{:<{width}}}" for width in widths)
    print(pattern.format(*header))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(pattern.format(*row))
    return 0


def _export(run_id: str, destination: str, *, with_checkpoints: bool) -> int:
    env = Environment()
    store = RunStore(env.paths.db_path)
    try:
        record = store.get_run(run_id)
    finally:
        store.close()
    if record is None:
        print(f"unknown run '{run_id}'", file=sys.stderr)
        return 1
    paths = RunPaths(env.paths.run_dir(record.folder))
    try:
        config = load_config(paths)
        target = export_run(
            run_id=run_id,
            paths=paths,
            config=config,
            destination=Path(destination),
            include_checkpoints=with_checkpoints,
        )
    except FileNotFoundError as exc:
        print(f"run artifacts are incomplete: {exc}", file=sys.stderr)
        return 1
    except AutorlhfError as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(f"exported {run_id} to {target} (credentials are never included)")
    return 0


def _headless(args: argparse.Namespace) -> int:
    from autorlhf.tui.worker import PipelineWorker, default_hooks

    env = Environment()
    store = RunStore(env.paths.db_path)
    try:
        plan = _resolve_headless_run(env, store, args)
        if plan is None:
            return 2
        run_id, paths, config, resume = plan
        bus = EventBus()
        bus.subscribe(bus.print_listener(verbose=args.verbose))
        try:
            hooks = default_hooks()
        except ImportError as exc:
            print(f"the pipeline is unavailable: {exc}", file=sys.stderr)
            return 2
        worker = PipelineWorker(
            run_id=run_id,
            env=env,
            store=store,
            paths=paths,
            config=config,
            bus=bus,
            interactive=args.interactive,
            verbose=args.verbose,
            hooks=hooks,
            resume=resume,
        )
        try:
            worker.run()
        except AutorlhfError as exc:
            print(f"headless run failed: {exc}", file=sys.stderr)
            if getattr(exc, "hint", ""):
                print(f"hint: {exc.hint}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001 - report, never traceback into the shell
            print(f"headless run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if worker.error is not None:
            return 1
        _print_headless_summary(worker)
        return 0
    finally:
        store.close()


def _resolve_headless_run(
    env: Environment, store: RunStore, args: argparse.Namespace
) -> tuple[str, RunPaths, RunConfig, bool] | None:
    """Work out which run ``--headless`` should execute."""

    if args.run_id:
        record = store.get_run(args.run_id)
        if record is None:
            print(f"unknown run '{args.run_id}'", file=sys.stderr)
            return None
        paths = RunPaths(env.paths.run_dir(record.folder))
        try:
            config = load_config(paths)
        except FileNotFoundError as exc:
            print(f"run {args.run_id} has no pinned config: {exc}", file=sys.stderr)
            return None
        return record.id, paths, config, bool(args.resume)

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"config file not found: {config_path}", file=sys.stderr)
            return None
        try:
            config = RunConfig.load(config_path)
        except Exception as exc:  # noqa: BLE001 - a bad file is a usage error
            print(f"could not read {config_path}: {exc}", file=sys.stderr)
            return None
        problems = config.validate_config()
        if problems:
            print("configuration is invalid:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return None
        run_id = new_run_id()
        folder = run_folder_name(run_id, config.slug or config.character.name)
        paths = RunPaths(env.paths.run_dir(folder)).ensure()
        record = store.create_run(
            config,
            run_id=run_id,
            folder=folder,
            config_path=paths.config_path,
            notes="headless",
        )
        save_config(paths, config)
        return record.id, paths, config, False

    print(
        "--headless needs --config PATH (a new run) or --run-id ID (an existing run)",
        file=sys.stderr,
    )
    return None


def _print_headless_summary(worker) -> None:
    result = worker.result
    adapter = getattr(result, "adapter_path", None)
    print("run complete")
    print(f"  run id:       {worker.run_id}")
    print(f"  artifacts:    {worker.paths.root}")
    if adapter:
        print(f"  adapter:      {adapter}")
    if worker.plan is not None:
        print(f"  objective:    {getattr(worker.plan, 'objective', '?')}")
    if worker.comparison is not None:
        delta = getattr(worker.comparison, "overall_delta", None)
        print(f"  overall change: {delta}")


def _run_tui(*, interactive: bool, verbose: bool) -> int:
    try:
        from autorlhf.tui.app import AutorlhfApp
    except ImportError as exc:
        print(f"the Textual UI is unavailable: {exc}", file=sys.stderr)
        return 2
    app = AutorlhfApp(interactive=interactive, verbose=verbose)
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - python -m autorlhf.cli
    raise SystemExit(main())
