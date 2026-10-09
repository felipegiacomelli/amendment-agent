"""Download the manifest's documents from SEC EDGAR, within the fair-access rules."""

import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx

from amendment_agent.manifest import Manifest

SUFFIXES = frozenset({".htm", ".html", ".txt"})


class FetchError(RuntimeError):
    """A document could not be downloaded."""


def raw_path(raw_dir: Path, doc_id: str, url: str) -> Path:
    """Where a document's raw bytes live: `<raw_dir>/<id><suffix from the URL>`."""
    suffix = PurePosixPath(urlparse(url).path).suffix.lower()
    if suffix not in SUFFIXES:
        raise FetchError(f"{doc_id}: unsupported file type {suffix!r} in {url}")
    return raw_dir / f"{doc_id}{suffix}"


def fetch_all(
    manifest: Manifest,
    raw_dir: Path,
    user_agent: str,
    *,
    client: httpx.Client | None = None,
    min_interval: float = 0.15,
) -> list[Path]:
    """Download every document not already in `raw_dir`; return all raw paths.

    SEC fair access: a "Name email" User-Agent and at most 10 requests per second.
    Requests are sequential and at least `min_interval` seconds apart.
    # ponytail: sequential sleep, not a token bucket; four documents.
    """
    if not user_agent.strip():
        raise FetchError("SEC_USER_AGENT is empty; SEC requires 'Name email'")
    raw_dir.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(follow_redirects=True, timeout=60.0)
    paths: list[Path] = []
    last_request: float | None = None
    try:
        for doc in manifest.documents:
            dest = raw_path(raw_dir, doc.id, doc.url)
            paths.append(dest)
            if dest.exists():
                print(f"skip   {doc.id}: {dest} exists")
                continue
            if last_request is not None:
                wait = min_interval - (time.monotonic() - last_request)
                if wait > 0:
                    time.sleep(wait)
            last_request = time.monotonic()
            response = http.get(doc.url, headers={"User-Agent": user_agent})
            if response.status_code != 200:
                raise FetchError(f"{doc.id}: HTTP {response.status_code} for {doc.url}")
            # Write then rename, so an interrupted download is never mistaken for a
            # complete one on the next run.
            part = dest.with_name(dest.name + ".part")
            part.write_bytes(response.content)
            part.replace(dest)
            print(f"fetch  {doc.id}: {dest} ({len(response.content):,} bytes)")
    finally:
        if client is None:
            http.close()
    return paths
