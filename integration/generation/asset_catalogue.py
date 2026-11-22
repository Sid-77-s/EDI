"""Local metadata catalogue for Objaverse-compatible 3D asset records.

Indexes metadata only; never downloads or opens models. Accepts UID-keyed JSON,
JSONL (including gzip), and optional Objaverse-XL Parquet metadata. Missing
licenses remain unknown.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union
from urllib.parse import urlparse
import re

PathLike = Union[str, Path]


def _concept_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


class LocalAssetPackResolver:
    """Offline resolver for curated ``.glb`` files and JSON sidecars.

    A sidecar named ``<model>.json`` may contain ``name``, ``aliases``,
    ``license``, ``source``, ``author``, and ``attribution``.  Assets with
    missing license metadata are rejected unless ``allow_unknown_license``.
    """

    def __init__(self, asset_pack_dir: PathLike, allowed_licenses=None,
                 allow_unknown_license: bool = False):
        self.asset_pack_dir = Path(asset_pack_dir)
        self.allowed_licenses = ({str(x).casefold() for x in allowed_licenses}
                                 if allowed_licenses is not None else None)
        self.allow_unknown_license = allow_unknown_license
        self._index: Optional[dict[str, list[tuple[int, str, Path, dict]]]] = None
        self._ensure_index()

    def _ensure_index(self) -> dict[str, list[tuple[int, str, Path, dict]]]:
        if self._index is not None:
            return self._index
        index: dict[str, list[tuple[int, str, Path, dict]]] = {}
        if not self.asset_pack_dir.is_dir():
            self._index = index
            return index
        for path in self.asset_pack_dir.rglob("*.glb"):
            sidecar = path.with_suffix(".json")
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
                if not isinstance(metadata, dict):
                    continue
            except (OSError, json.JSONDecodeError):
                continue
            license_value = metadata.get("license")
            license_key = str(license_value).strip().casefold() if license_value else None
            if license_key is None and not self.allow_unknown_license:
                continue
            if self.allowed_licenses is not None and license_key not in self.allowed_licenses:
                continue
            name = str(metadata.get("name") or path.stem)
            aliases = metadata.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list):
                aliases = []
            names = [name, *[str(alias) for alias in aliases]]
            entry_keys = {_concept_key(item) for item in names if item.strip()}
            for key in entry_keys:
                rank = 0 if _concept_key(name) == key else 1
                index.setdefault(key, []).append((rank, str(path).casefold(), path, metadata))
        self._index = index
        return index

    def resolve(self, concept: str):
        """Return a loaded ``(Model3D, metadata)`` match or ``None``."""
        index = self._ensure_index()
        wanted = _concept_key(concept)
        matches = index.get(wanted)
        if not matches:
            return None
        _, _, path, metadata = min(matches)
        from integration.generation.model3d import load_glb
        model = load_glb(path)
        model.concept = concept
        resolved_name = metadata.get("name") or path.stem
        model.metadata.update({
            "name": resolved_name,
            "aliases": metadata.get("aliases", []),
            "category": metadata.get("category"),
            "subcategory": metadata.get("subcategory"),
            "asset_source": str(path),
            "license": metadata.get("license"),
            "source": metadata.get("source"),
            "author": metadata.get("author"),
            "attribution": metadata.get("attribution"),
            "distinguishing_features": metadata.get("distinguishing_features", []),
            "educational_purpose": metadata.get("educational_purpose"),
        })
        return model, model.metadata


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8")


def _identifier(item: dict[str, Any]) -> Optional[str]:
    value = item.get("uid") or item.get("fileIdentifier") or item.get("id")
    return str(value) if value is not None and str(value).strip() else None


def _json_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
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
            first_line = handle.readline()
            try:
                first_record = json.loads(first_line)
            except json.JSONDecodeError:
                first_record = None
            if isinstance(first_record, dict) and _identifier(first_record):
                handle.seek(0)
            else:
                handle.seek(0)
                try:
                    payload = json.load(handle)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed UID-keyed JSON: {path}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected UID-keyed JSON object: {path}")
                for uid, item in payload.items():
                    if isinstance(item, dict):
                        yield str(uid), item
                return
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                uid = _identifier(item)
                if uid:
                    yield uid, item


def _parquet_records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet metadata requires the optional 'pyarrow' package") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches():
        for row in batch.to_pylist():
            if not isinstance(row, dict):
                continue
            uid = _identifier(row)
            if uid:
                yield uid, row


def _records(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    if path.suffix.lower() == ".parquet":
        yield from _parquet_records(path)
    else:
        yield from _json_records(path)


def _source_key(uid: str, item: dict[str, Any]) -> str:
    source = item.get("source")
    if not source:
        parsed = urlparse(uid)
        source = parsed.netloc.casefold() if parsed.scheme and parsed.netloc else "objaverse"
    return f"{str(source).strip().casefold()}\x1f{uid}"


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
    return (_source_key(uid, item), uid, name, description, tags, license_value,
            str(author) if author else None, str(source_url) if source_url else None,
            str(source) if source else None, str(file_type) if file_type else None,
            search_text, json.dumps(item, ensure_ascii=False, sort_keys=True))


def _input_files(metadata: Union[PathLike, Iterable[PathLike]]) -> list[Path]:
    sources = [metadata] if isinstance(metadata, (str, Path)) else list(metadata)
    files: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".gz", ".parquet"})
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(set(files), key=lambda p: str(p).casefold())


def _ensure_schema(db: sqlite3.Connection) -> None:
    existing = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assets'").fetchone()
    if existing:
        columns = {row[1] for row in db.execute("PRAGMA table_info(assets)")}
        if "asset_key" not in columns:
            db.execute("ALTER TABLE assets RENAME TO assets_legacy")
            db.execute("""CREATE TABLE assets (
                asset_key TEXT PRIMARY KEY, uid TEXT NOT NULL, name TEXT NOT NULL,
                description TEXT NOT NULL, tags TEXT NOT NULL, license TEXT, author TEXT,
                source_url TEXT, source TEXT, file_type TEXT, search_text TEXT NOT NULL,
                raw_json TEXT NOT NULL)""")
            for row in db.execute("SELECT uid,name,description,tags,license,author,source_url,source,file_type,search_text,raw_json FROM assets_legacy").fetchall():
                uid, _, _, _, _, _, _, source, _, _, raw_json = row
                try:
                    raw = json.loads(raw_json)
                except (TypeError, json.JSONDecodeError):
                    raw = {"source": source}
                db.execute("INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (_source_key(uid, raw), *row))
            db.execute("DROP TABLE assets_legacy")
    else:
        db.execute("""CREATE TABLE assets (
            asset_key TEXT PRIMARY KEY, uid TEXT NOT NULL, name TEXT NOT NULL,
            description TEXT NOT NULL, tags TEXT NOT NULL, license TEXT, author TEXT,
            source_url TEXT, source TEXT, file_type TEXT, search_text TEXT NOT NULL,
            raw_json TEXT NOT NULL)""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_assets_license ON assets(license)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name COLLATE NOCASE)")


def build_catalogue(metadata_path: Union[PathLike, Iterable[PathLike]], database_path: PathLike) -> int:
    """Stream metadata files into SQLite; return valid records processed.

    ``metadata_path`` may be one file, a directory (searched recursively), or
    an iterable of file and directory paths. Duplicate source/identifier pairs
    update the same catalogue row; identical identifiers from other sources do not.
    """
    files = _input_files(metadata_path)
    if not files:
        raise FileNotFoundError("No metadata files found")
    target = Path(database_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with sqlite3.connect(str(target)) as db:
        _ensure_schema(db)
        for source_file in files:
            for uid, item in _records(source_file):
                db.execute("""INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(asset_key) DO UPDATE SET uid=excluded.uid, name=excluded.name,
                    description=excluded.description, tags=excluded.tags, license=excluded.license,
                    author=excluded.author, source_url=excluded.source_url, source=excluded.source,
                    file_type=excluded.file_type, search_text=excluded.search_text, raw_json=excluded.raw_json""",
                           _normalise(uid, item))
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
        where.append("instr(search_text, ?) > 0")
        params.append(term)
    if license is not None:
        where.append("license = ?")
        params.append(license)
    sql = "SELECT uid,name,description,tags,license,author,source_url,source,file_type FROM assets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY name COLLATE NOCASE, uid, source LIMIT ?"
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
