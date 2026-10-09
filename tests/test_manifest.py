from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from amendment_agent.manifest import DocumentType, Manifest, load_manifest
from amendment_agent.settings import MANIFEST_PATH


def doc(
    id: str, type: str = "amendment", amends: str | None = "original"
) -> dict[str, Any]:
    return {
        "id": id,
        "type": type,
        "title": "Document",
        "dated": "2020-01-01",
        "filing_date": "2020-01-02",
        "accession": "0000000001-20-000001",
        "exhibit": "EX-10.1",
        "url": "https://www.sec.gov/Archives/edgar/data/1/x.htm",
        "amends": amends,
    }


def manifest(*documents: dict[str, Any]) -> dict[str, Any]:
    return {
        "company": "Example Corp",
        "cik": "0000000001",
        "documents": list(documents),
    }


ORIGINAL = doc("original", type="credit_agreement", amends=None)


def test_repo_manifest_loads_in_legal_order() -> None:
    loaded = load_manifest(MANIFEST_PATH)
    assert [d.id for d in loaded.documents] == [
        "original",
        "amendment-1",
        "amendment-2",
        "amendment-3",
    ]
    assert loaded.documents[0].type is DocumentType.CREDIT_AGREEMENT
    assert all(d.amends == "original" for d in loaded.documents[1:])
    dated = [d.dated for d in loaded.documents]
    assert dated == sorted(dated)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate document ids"):
        Manifest.model_validate(
            manifest(ORIGINAL, doc("amendment-1"), doc("amendment-1"))
        )


def test_dangling_amends_rejected() -> None:
    with pytest.raises(ValidationError, match="amends unknown document"):
        Manifest.model_validate(manifest(ORIGINAL, doc("amendment-1", amends="nope")))


def test_amendment_must_name_what_it_amends() -> None:
    with pytest.raises(ValidationError, match="must set 'amends'"):
        Manifest.model_validate(manifest(ORIGINAL, doc("amendment-1", amends=None)))


def test_non_sec_url_rejected() -> None:
    bad = doc("original", type="credit_agreement", amends=None) | {
        "url": "https://example.com/x.htm"
    }
    with pytest.raises(ValidationError):
        Manifest.model_validate(manifest(bad))


def test_unquoted_cik_rejected_at_cik(tmp_path: Path) -> None:
    # YAML 1.1 reads 0001757073 as an octal int; it must fail, not become the octal int 515643.
    valid = MANIFEST_PATH.read_text(encoding="utf-8")
    unquoted = valid.replace('cik: "0001757073"', "cik: 0001757073")
    assert unquoted != valid
    path = tmp_path / "manifest.yaml"
    path.write_text(unquoted, encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        load_manifest(path)
    assert [error["loc"] for error in excinfo.value.errors()] == [("cik",)]
