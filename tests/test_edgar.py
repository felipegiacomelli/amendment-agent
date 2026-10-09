from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from amendment_agent.edgar import FetchError, fetch_all
from amendment_agent.manifest import Manifest

type MakeManifest = Callable[..., Manifest]  # the conftest fixture's type

BASE = "https://www.sec.gov/Archives/edgar/data/1/000000000120000001"
UA = "Jane Doe jane@example.com"


def mock_client(
    seen: list[httpx.Request], status: int = 200, body: bytes = b"<p>hi</p>"
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_sends_user_agent_and_writes_one_file_per_document(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    seen: list[httpx.Request] = []
    paths = fetch_all(
        make_manifest(f"{BASE}/a.htm", f"{BASE}/b.txt"),
        tmp_path,
        UA,
        client=mock_client(seen),
        min_interval=0,
    )
    assert [p.name for p in paths] == ["original.htm", "amendment-1.txt"]
    assert [r.headers["User-Agent"] for r in seen] == [UA, UA]
    assert (tmp_path / "original.htm").read_bytes() == b"<p>hi</p>"


def test_empty_user_agent_rejected(tmp_path: Path, make_manifest: MakeManifest) -> None:
    with pytest.raises(FetchError, match="SEC_USER_AGENT"):
        fetch_all(
            make_manifest(f"{BASE}/a.htm"), tmp_path, "  ", client=mock_client([])
        )


def test_existing_file_is_skipped_without_a_request(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    (tmp_path / "original.htm").write_text("cached", encoding="utf-8")
    seen: list[httpx.Request] = []
    fetch_all(make_manifest(f"{BASE}/a.htm"), tmp_path, UA, client=mock_client(seen))
    assert seen == []
    assert (tmp_path / "original.htm").read_text(encoding="utf-8") == "cached"


def test_http_error_names_document_and_leaves_no_file(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    with pytest.raises(FetchError, match=r"original: HTTP 403 for .*/a\.htm"):
        fetch_all(
            make_manifest(f"{BASE}/a.htm"),
            tmp_path,
            UA,
            client=mock_client([], status=403),
        )
    assert list(tmp_path.iterdir()) == []


def test_unsupported_file_type_rejected(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    with pytest.raises(FetchError, match="unsupported file type '.pdf'"):
        fetch_all(make_manifest(f"{BASE}/a.pdf"), tmp_path, UA, client=mock_client([]))


def test_requests_are_spaced_by_min_interval(
    tmp_path: Path, make_manifest: MakeManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [100.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("amendment_agent.edgar.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("amendment_agent.edgar.time.sleep", fake_sleep)
    fetch_all(
        make_manifest(f"{BASE}/a.htm", f"{BASE}/b.htm", f"{BASE}/c.htm"),
        tmp_path,
        UA,
        client=mock_client([]),
        min_interval=0.15,
    )
    assert sleeps == pytest.approx([0.15, 0.15])
