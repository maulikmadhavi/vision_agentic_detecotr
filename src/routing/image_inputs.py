"""Image inputs: manifest loading and validated image reading with a content-hash identity."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .schemas import ImageRef


@dataclass(frozen=True)
class ManifestEntry:
    uri: str
    image_id: str | None = None
    group_id: str | None = None


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """A directory of images, or a JSONL manifest of {"uri", "image_id"?, "group_id"?}."""
    p = Path(path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return [ManifestEntry(str(f)) for f in sorted(p.iterdir()) if f.suffix.lower() in exts]
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = p.parent
    return [ManifestEntry(str((base / r["uri"]) if not Path(r["uri"]).is_absolute() else r["uri"]),
                          r.get("image_id"), r.get("group_id")) for r in rows]


def image_id(entry: ManifestEntry, data: bytes) -> str:
    """The image id the pipeline records: the manifest's id, else a content-hash id."""
    return entry.image_id or f"sha256-{hashlib.sha256(data).hexdigest()[:16]}"


def load_image(entry: ManifestEntry) -> tuple[Image.Image, ImageRef]:
    data = Path(entry.uri).read_bytes()
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()
    img = Image.open(io.BytesIO(data))
    img.load()
    img = img.convert("RGB")
    sha = hashlib.sha256(data).hexdigest()
    ref = ImageRef(id=image_id(entry, data), uri=entry.uri, width=img.width,
                   height=img.height, sha256=sha, group_id=entry.group_id)
    return img, ref
