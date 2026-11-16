import gzip
import json
import builtins

import pytest

from integration.generation.asset_catalogue import build_catalogue, search_catalogue


def test_objaverse_uid_mapping_search_and_unknown_license(tmp_path):
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps({
        "uid-a": {"name": "wooden chair", "description": "A seat", "tags": ["furniture"], "license": "by", "author": {"username": "maker"}, "url": "https://example/a"},
        "uid-b": {"name": "chair sketch", "tags": ["drawing"], "license": None},
        "uid-c": {"name": "wooden table", "license": "by-sa"},
    }), encoding="utf-8")
    database = tmp_path / "catalogue.sqlite"
    assert build_catalogue(source, database) == 3
    results = search_catalogue(database, "WOODEN furniture", license="by")
    assert [item["uid"] for item in results] == ["uid-a"]
    assert results[0]["author"] == "maker"
    assert results[0]["source_url"] == "https://example/a"
    assert any(item["license"] is None for item in search_catalogue(database, "chair"))


def test_gzip_jsonl_and_objaverse_xl_style_fields(tmp_path):
    source, database = tmp_path / "export.jsonl.gz", tmp_path / "catalogue.sqlite"
    rows = [
        {"fileIdentifier": "https://example/model.glb", "name": "red bicycle", "license": "CC-BY-4.0", "source": "github", "fileType": "glb", "sha256": "abc"},
        {"uid": "unknown-license", "name": "bicycle wheel"},
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert build_catalogue(source, database) == 2
    result = search_catalogue(database, "bicycle", license="CC-BY-4.0")
    assert len(result) == 1
    assert result[0]["source"] == "github"
    assert result[0]["file_type"] == "glb"
    assert search_catalogue(database, "wheel")[0]["license"] is None


def test_rebuild_updates_existing_uid(tmp_path):
    source, database = tmp_path / "annotations.json", tmp_path / "catalogue.sqlite"
    source.write_text(json.dumps({"same": {"name": "old name", "license": "by"}}), encoding="utf-8")
    assert build_catalogue(source, database) == 1
    source.write_text(json.dumps({"same": {"name": "new name", "license": "by-sa"}}), encoding="utf-8")
    assert build_catalogue(source, database) == 1
    assert search_catalogue(database, "new name")[0]["license"] == "by-sa"
    assert search_catalogue(database, "old name") == []


def test_directory_mixed_formats_streamed_parquet_and_source_aware_dedup(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder, database = tmp_path / "metadata", tmp_path / "catalogue.sqlite"
    folder.mkdir()
    (folder / "a.json").write_text(json.dumps({"shared": {"name": "json object", "license": "by"}}), encoding="utf-8")
    with gzip.open(folder / "b.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"uid": "shared", "name": "updated json object", "license": "by-sa"}) + "\n")
    table = pa.Table.from_pylist([
        {"fileIdentifier": "shared", "name": "parquet object", "source": "other-source", "license": "CC0", "fileType": "glb", "extra": "kept"},
        {"fileIdentifier": "parquet-only", "name": "parquet lamp", "source": "other-source"},
    ])
    pq.write_table(table, folder / "c.parquet", row_group_size=1)
    assert build_catalogue(folder, database) == 4
    results = search_catalogue(database, "object", limit=10)
    assert len(results) == 2
    by_source = {row["source"]: row for row in results}
    assert by_source[None]["name"] == "updated json object"
    assert by_source["other-source"]["name"] == "parquet object"
    assert by_source["other-source"]["license"] == "CC0"
    assert by_source["other-source"]["file_type"] == "glb"
    assert len(search_catalogue(database, "lamp")) == 1


def test_malformed_jsonl_records_are_skipped_and_missing_license_stays_unknown(tmp_path):
    source, database = tmp_path / "mixed.jsonl", tmp_path / "catalogue.sqlite"
    source.write_text(
        '{"uid":"good","name":"good model"}\n'
        '{broken json}\n'
        '42\n'
        '{"name":"missing identifier"}\n'
        '{"uid":"also-good","name":"another model","license":""}\n',
        encoding="utf-8",
    )
    assert build_catalogue(source, database) == 2
    rows = search_catalogue(database, "model", limit=10)
    assert len(rows) == 2
    assert all(row["license"] is None for row in rows)


def test_multiple_explicit_files_and_same_source_dedup(tmp_path):
    first, second, database = tmp_path / "first.json", tmp_path / "second.jsonl", tmp_path / "catalogue.sqlite"
    first.write_text(json.dumps({"uid-x": {"name": "before", "source": "vendor"}}), encoding="utf-8")
    second.write_text(json.dumps({"uid": "uid-x", "name": "after", "source": "vendor"}) + "\n", encoding="utf-8")
    assert build_catalogue([first, second], database) == 2
    assert len(search_catalogue(database, "after")) == 1
    assert search_catalogue(database, "before") == []


def test_json_workflow_does_not_require_pyarrow(tmp_path, monkeypatch):
    import integration.generation.asset_catalogue as catalogue

    source, database = tmp_path / "metadata.json", tmp_path / "catalogue.sqlite"
    source.write_text(json.dumps({"without-arrow": {"name": "plain JSON"}}), encoding="utf-8")
    original_import = builtins.__import__

    def import_without_pyarrow(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_pyarrow)
    assert catalogue.build_catalogue(source, database) == 1
    assert search_catalogue(database, "plain JSON")[0]["license"] is None
