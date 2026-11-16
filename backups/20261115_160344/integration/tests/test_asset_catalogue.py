import gzip
import json

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
    assert search_catalogue(database, "chair")[1]["license"] is None


def test_gzip_jsonl_and_objaverse_xl_style_fields(tmp_path):
    source, database = tmp_path / "export.jsonl.gz", tmp_path / "catalogue.sqlite"
    rows = [
        {"fileIdentifier": "https://example/model.glb", "name": "red bicycle", "license": "CC-BY-4.0", "source": "github", "fileType": "glb", "sha256": "abc"},
        {"uid": "unknown-license", "name": "bicycle wheel", "license": None},
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
