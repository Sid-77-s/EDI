"""Local metadata catalogue for Objaverse-compatible 3D asset records.

Indexes metadata only; never downloads or opens models. Accepts Objaverse 1.0
UID-keyed JSON and JSONL, including gzip. Missing licenses remain unknown.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Optional, Union

PathLike = Union[str, Path]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8")


def _records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with _open_text(path) as handle:
        first = ""
        while not first:
            first = handle.read(1)
            if not first:
                return
            if not first.isspace():
                break
        handle.seek(0)
        if first == "{":
            payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected UID-keyed JSON object: {path}")
            for uid, item in payload.items():
                if isinstance(item, dict):
                    yield str(uid), item
            return
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            uid = item.get("uid") or item.get("fileIdentifier") or item.get("id")
            if uid:
                yield str(uid), item


def _normalise(uid: str, item: dict[str, Any]) -> tuple:
    tags = item.get("tags", [])
    if isinstance(tags, dict):
        tags = list(tags.values())
    if isinstance(tags, (list, tuple)):
        tags = " ".join(str(x.get("name", x) if isinstance(x, dict) else x) for x in tags)
    else:
        tags = str(tags or "")
    name = str(item.get("name") or item.get("fileIdentifier") or "")
    description = str(item.get("description") or "")
    license_value = item.get("license")
    license_value = str(license_value).strip() if license_value not in (None, "") else None
    author = item.get("author")
    if isinstance(author, dict):
        author = author.get("username") or author.get("name") or author.get("uid")
    source_url = item.get("url") or item.get("source_url") or item.get("fileIdentifier")
    source, file_type = item.get("source"), item.get("fileType") or item.get("file_type")
    search_text = " ".join((uid, name, description, tags, str(author or ""), str(source or ""))).casefold()
    return (uid, name, description, tags, license_value, str(author) if author else None,
            str(source_url) if source_url else None, str(source) if source else None,
            str(file_type) if file_type else None, search_text,
            json.dumps(item, ensure_ascii=False, sort_keys=True))


def build_catalogue(metadata_path: PathLike, database_path: PathLike) -> int:
    """Import metadata records into local SQLite; return the number read."""
    source, target = Path(metadata_path), Path(database_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with sqlite3.connect(str(target)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS assets (
            uid TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
            tags TEXT NOT NULL, license TEXT, author TEXT, source_url TEXT,
            source TEXT, file_type TEXT, search_text TEXT NOT NULL, raw_json TEXT NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_assets_license ON assets(license)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name COLLATE NOCASE)")
        for uid, item in _records(source):
            db.execute("""INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(uid) DO UPDATE SET name=excluded.name, description=excluded.description,
                tags=excluded.tags, license=excluded.license, author=excluded.author,
                source_url=excluded.source_url, source=excluded.source, file_type=excluded.file_type,
                search_text=excluded.search_text, raw_json=excluded.raw_json""", _normalise(uid, item))
            count += 1
        db.commit()
    return count


def search_catalogue(database_path: PathLike, query: str = "", license: Optional[str] = None,
                     limit: int = 20) -> list[dict[str, Any]]:
    """Search text fields and optionally require an exact license value."""
    if limit < 1:
        raise ValueError("limit must be positive")
    where, params = [], []
    for term in query.casefold().split():
        where.append("search_text LIKE ?")
        params.append("%" + term.replace("%", "\\%").replace("_", "\\_") + "%")
    if license is not None:
        where.append("license = ?")
        params.append(license)
    sql = "SELECT uid,name,description,tags,license,author,source_url,source,file_type FROM assets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY name COLLATE NOCASE, uid LIMIT ?"
    params.append(limit)
    with sqlite3.connect(str(database_path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(sql, params)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build/search a local 3D metadata catalogue (metadata only).")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("metadata", type=Path)
    build.add_argument("database", type=Path)
    search = sub.add_parser("search")
    search.add_argument("database", type=Path)
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--license")
    search.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "build":
        print(f"Indexed {build_catalogue(args.metadata, args.database)} metadata records")
    else:
        print(json.dumps(search_catalogue(args.database, args.query, args.license, args.limit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
