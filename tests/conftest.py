from collections.abc import Callable

import pytest

from amendment_agent.manifest import Manifest

type MakeManifest = Callable[..., Manifest]


@pytest.fixture
def make_manifest() -> MakeManifest:
    """Build a valid Manifest whose first URL is the original, the rest amendments."""

    def make(*urls: str) -> Manifest:
        documents = [
            {
                "id": "original" if i == 0 else f"amendment-{i}",
                "type": "credit_agreement" if i == 0 else "amendment",
                "title": "Document",
                "dated": "2020-01-01",
                "filing_date": "2020-01-02",
                "accession": "0000000001-20-000001",
                "exhibit": "EX-10.1",
                "url": url,
                "amends": None if i == 0 else "original",
            }
            for i, url in enumerate(urls)
        ]
        return Manifest.model_validate(
            {"company": "Example Corp", "cik": "0000000001", "documents": documents}
        )

    return make
