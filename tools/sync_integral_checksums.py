"""Add the common integral checksum field to every catalog entry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "datasets" / "catalog.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    for entry in catalog["datasets"]:
        path = entry.get("integral_data_path")
        if not path:
            raise RuntimeError(f"missing integral_data_path for {entry['id']}")
        entry["integral_sha256"] = sha256_file(ROOT / path)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"updated integral checksums for {len(catalog['datasets'])} catalog entries")


if __name__ == "__main__":
    main()
