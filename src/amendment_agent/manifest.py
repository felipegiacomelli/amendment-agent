"""The document manifest: which EDGAR filings make up the agreement family."""

import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"


class DocumentType(StrEnum):
    CREDIT_AGREEMENT = "credit_agreement"
    AMENDMENT = "amendment"


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SLUG)
    type: DocumentType
    title: str
    # "Dated as of": the legal order. Filing order differs (Amendment No. 2 was
    # filed after Amendment No. 3), so precedence must come from this field.
    dated: dt.date
    filing_date: dt.date
    accession: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    exhibit: str
    url: str = Field(pattern=r"^https://www\.sec\.gov/\S+$")
    amends: str | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    company: str
    cik: str = Field(pattern=r"^\d{10}$")
    documents: list[Document] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> Self:
        ids = [d.id for d in self.documents]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate document ids: {duplicates}")
        for doc in self.documents:
            if doc.type is DocumentType.AMENDMENT and doc.amends is None:
                raise ValueError(f"amendment {doc.id!r} must set 'amends'")
            if doc.amends is not None and doc.amends not in ids:
                raise ValueError(f"{doc.id!r} amends unknown document {doc.amends!r}")
        return self


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
