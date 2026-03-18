"""Tests for ActionLogger — per-item structured JSONL logging."""

import json
from pathlib import Path

import pytest

from lattereview.agentic.logging.action_log import ActionLogger


@pytest.fixture
def logs_dir(tmp_path):
    d = tmp_path / "logs"
    return d


class TestActionLoggerInit:
    def test_creates_directory(self, logs_dir):
        logger = ActionLogger(logs_dir)
        assert logs_dir.exists()

    def test_logs_dir_property(self, logs_dir):
        logger = ActionLogger(logs_dir)
        assert logger.logs_dir == logs_dir


class TestLogging:
    def test_log_creates_file(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("item_0", "review_start")
        assert (logs_dir / "item_item_0.jsonl").exists()

    def test_log_entry_format(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("A-0", "review_start", {"agent": "Rev1"})
        entries = logger.read_log("A-0")
        assert len(entries) == 1
        assert entries[0]["action_type"] == "review_start"
        assert entries[0]["item_id"] == "A-0"
        assert entries[0]["details"]["agent"] == "Rev1"
        assert "timestamp" in entries[0]

    def test_multiple_entries(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("A-0", "review_start")
        logger.log("A-0", "tool_call", {"tool": "search_pubmed"})
        logger.log("A-0", "review_complete", {"cost": 0.001})
        entries = logger.read_log("A-0")
        assert len(entries) == 3
        assert entries[0]["action_type"] == "review_start"
        assert entries[1]["action_type"] == "tool_call"
        assert entries[2]["action_type"] == "review_complete"

    def test_separate_items(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("A-0", "review_start")
        logger.log("A-1", "review_start")
        assert len(logger.read_log("A-0")) == 1
        assert len(logger.read_log("A-1")) == 1

    def test_no_details(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("A-0", "review_start")
        entries = logger.read_log("A-0")
        assert "details" not in entries[0]


class TestReadLog:
    def test_read_nonexistent(self, logs_dir):
        logger = ActionLogger(logs_dir)
        assert logger.read_log("nonexistent") == []

    def test_has_log_true(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("A-0", "start")
        assert logger.has_log("A-0") is True

    def test_has_log_false(self, logs_dir):
        logger = ActionLogger(logs_dir)
        assert logger.has_log("A-0") is False


class TestItemIdSanitization:
    def test_slash_in_item_id(self, logs_dir):
        logger = ActionLogger(logs_dir)
        logger.log("round/A-0", "start")
        entries = logger.read_log("round/A-0")
        assert len(entries) == 1
        # File should have sanitized name
        assert (logs_dir / "item_round_A-0.jsonl").exists()
