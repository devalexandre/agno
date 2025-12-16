from pathlib import Path

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from sqlalchemy import inspect  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from agno.db.schema import DEFAULT_TABLE_NAMES, build_agno_metadata, compile_ddl_for_metadata  # noqa: E402
from agno.db.sqlite.sqlite import SqliteDb  # noqa: E402


def test_build_agno_metadata_registers_all_tables():
    assert DEFAULT_TABLE_NAMES, "DEFAULT_TABLE_NAMES must not be empty"

    metadata = build_agno_metadata(schema="ai")

    # Ensure every default table name is present (ignore schema prefix)
    actual_names = {table.name for table in metadata.tables.values()}
    assert set(DEFAULT_TABLE_NAMES.values()).issubset(actual_names)

    engine = create_engine("sqlite:///:memory:")
    ddl_statements = compile_ddl_for_metadata(metadata, dialect=engine.dialect)

    assert ddl_statements, "compile_ddl_for_metadata() must return at least one statement"
    assert all(isinstance(s, str) and s.strip() for s in ddl_statements), "DDL statements must be non-empty strings"


def test_sqlite_create_all_tables_and_ddl(tmp_path: Path):
    assert DEFAULT_TABLE_NAMES, "DEFAULT_TABLE_NAMES must not be empty"

    db_path = tmp_path / "agno.db"
    db = SqliteDb(db_file=str(db_path))

    # Should be idempotent
    db.create_all_tables()
    db.create_all_tables()

    inspector = inspect(db.db_engine)
    existing_tables = set(inspector.get_table_names())
    assert existing_tables, "Expected at least one table to exist after create_all_tables()"

    for expected in DEFAULT_TABLE_NAMES.values():
        assert expected in existing_tables

    ddl_statements = db.get_create_all_tables_ddl()
    assert ddl_statements, "get_create_all_tables_ddl() must return at least one statement"

    ddl_sql = "\n".join(ddl_statements).upper()

    # Ensure DDL covers all expected tables (avoid vacuous passes)
    for table_name in DEFAULT_TABLE_NAMES.values():
        assert f"CREATE TABLE {table_name.upper()}" in ddl_sql

    # Also ensure statements look like DDL
    assert any(s.strip().upper().startswith("CREATE TABLE") for s in ddl_statements)
