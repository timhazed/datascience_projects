"""Tests for src/config/db_config.py — DBConfig."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.db_config import DBConfig


def test_valid_relative_paths():
    c = DBConfig(sqlite_path="data/healthcare.db", faiss_path="data/faiss")
    assert isinstance(c.sqlite_path, Path)
    assert isinstance(c.faiss_path, Path)
    assert c.sqlite_path == Path("data/healthcare.db")


def test_valid_absolute_paths(tmp_path):
    db = tmp_path / "healthcare.db"
    faiss = tmp_path / "faiss"
    c = DBConfig(sqlite_path=db, faiss_path=faiss)
    assert c.sqlite_path == db


def test_missing_sqlite_path_raises():
    with pytest.raises(ValidationError):
        DBConfig(faiss_path="data/faiss")


def test_missing_faiss_path_raises():
    with pytest.raises(ValidationError):
        DBConfig(sqlite_path="data/healthcare.db")


def test_roundtrip():
    c = DBConfig(sqlite_path="data/healthcare.db", faiss_path="data/faiss")
    restored = DBConfig.model_validate(c.model_dump())
    assert restored.sqlite_path == Path("data/healthcare.db")
    assert restored.faiss_path == Path("data/faiss")


def test_checkpoint_db_path_default():
    c = DBConfig(sqlite_path="data/healthcare.db", faiss_path="data/faiss")
    assert c.checkpoint_db_path == "data/checkpoints.db"


def test_checkpoint_db_path_custom():
    c = DBConfig(
        sqlite_path="data/healthcare.db",
        faiss_path="data/faiss",
        checkpoint_db_path="custom/ckpt.db",
    )
    assert c.checkpoint_db_path == "custom/ckpt.db"


def test_checkpointing_enabled_defaults_false():
    c = DBConfig(sqlite_path="data/healthcare.db", faiss_path="data/faiss")
    assert c.checkpointing_enabled is False


def test_checkpointing_enabled_can_be_set():
    c = DBConfig(
        sqlite_path="data/healthcare.db",
        faiss_path="data/faiss",
        checkpointing_enabled=True,
    )
    assert c.checkpointing_enabled is True
