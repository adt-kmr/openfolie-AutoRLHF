"""Dataset inspector: table, filters, inline edits and validation warnings."""

from __future__ import annotations

import json

from test_tui_support import dataset_records, make_app, make_run, running, static_text, write_jsonl
from textual.widgets import DataTable, Select

from autorlhf.core.enums import RunStatus
from autorlhf.storage.runs import read_jsonl
from autorlhf.tui.screens.dataset import DatasetScreen, dataset_warnings


def test_dataset_warnings_cover_every_problem_class():
    records = [
        {
            "prompt_id": "p1",
            "candidate_id": "p1-c0",
            "prompt": "a",
            "response": "",
            "status": "accepted",
            "truncated": True,
            "prompt_context": [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
            "judgement": None,
        },
        {
            "prompt_id": "p1",
            "candidate_id": "p1-c0",
            "prompt": "a",
            "response": "same",
            "status": "accepted",
            "judgement": {"overall": 1.0},
        },
        {
            "prompt_id": "p1",
            "candidate_id": "p1-c1",
            "prompt": "different prompt text",
            "response": "same",
            "status": "accepted",
            "prompt_context": [{"role": "wizard", "content": "?"}],
            "judgement": {"overall": 2.0},
        },
    ]
    problems = "\n".join(
        dataset_warnings(records, [{"pair_id": "x", "chosen_id": "a", "rejected_id": "a"}])
    )
    assert "empty response" in problems
    assert "truncated" in problems
    assert "duplicate records" in problems
    assert "does not alternate" in problems
    assert "unknown role" in problems
    assert "accepted but never judged" in problems
    assert "identical candidate responses" in problems
    assert "prompt text differs" in problems
    assert "chosen and rejected are the same" in problems


async def test_dataset_screen_lists_filters_and_edits(app_paths, store, bus, run_config):
    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.GENERATING)
    write_jsonl(run_paths.dataset_path, dataset_records(run_paths, count=3))
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("dataset")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DatasetScreen)
        assert len(screen.records) == 3
        table = screen.query_one("#dataset-table", DataTable)
        assert table.row_count == 3

        # filter by split
        screen.query_one("#dataset-split-filter", Select).value = "eval"
        await pilot.pause()
        assert table.row_count == 2
        screen.query_one("#dataset-split-filter", Select).value = "all"
        await pilot.pause()
        assert table.row_count == 3

        # inline edit of a response rewrites dataset.jsonl
        assert screen.select_index(0) is True
        assert screen.apply_edit("A hand-edited answer.") is True
        await pilot.pause()
        written = read_jsonl(run_paths.dataset_path)
        assert written[0]["response"] == "A hand-edited answer."
        assert written[0]["status"] == "edited"
        assert written[0]["edited"] is True
        assert written[0]["edits"][-1]["by"] == "human"

        # reject action
        assert screen.set_status("rejected") is True
        assert read_jsonl(run_paths.dataset_path)[0]["status"] == "rejected"


async def test_dataset_screen_reports_warnings_and_missing_file(app_paths, store, bus, run_config):
    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.GENERATING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("dataset")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DatasetScreen)
        assert "does not exist yet" in static_text(screen.query_one("#dataset-status"))

        records = dataset_records(run_paths, count=2)
        records[1]["response"] = ""
        write_jsonl(run_paths.dataset_path, records)
        screen.reload()
        await pilot.pause()
        warnings = static_text(screen.query_one("#dataset-warnings"))
        assert "empty response" in warnings
        assert "No warnings" not in warnings


async def test_dataset_screen_without_an_open_run(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("dataset")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DatasetScreen)
        assert screen.reload() == 0
        assert "No run is open" in static_text(screen.query_one("#dataset-status"))


def test_edited_dataset_stays_valid_jsonl(app_paths, store, bus, run_config):
    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.GENERATING)
    write_jsonl(run_paths.dataset_path, dataset_records(run_paths, count=1))
    raw = run_paths.dataset_path.read_text(encoding="utf-8")
    assert json.loads(raw.splitlines()[0])["candidate_id"] == "p0000-c0"
