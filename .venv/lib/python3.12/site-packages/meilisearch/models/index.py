from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union

from camel_converter import to_snake
from camel_converter.pydantic_base import CamelBase


class IndexStats:
    __dict: Dict

    def __init__(self, doc: Dict[str, Any]) -> None:
        self.__dict = doc
        for key, val in doc.items():
            key = to_snake(key)
            if isinstance(val, dict):
                setattr(self, key, IndexStats(val))
            else:
                setattr(self, key, val)

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict.keys():
            return attr
        raise AttributeError(f"{self.__class__.__name__} object has no attribute {attr}")

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.items())


class Faceting(CamelBase):
    max_values_per_facet: int
    sort_facet_values_by: Optional[Dict[str, str]] = None


class Pagination(CamelBase):
    max_total_hits: int


class MinWordSizeForTypos(CamelBase):
    one_typo: Optional[int] = None
    two_typos: Optional[int] = None


class TypoTolerance(CamelBase):
    enabled: bool = True
    disable_on_attributes: Optional[List[str]] = None
    disable_on_words: Optional[List[str]] = None
    min_word_size_for_typos: Optional[MinWordSizeForTypos] = None


class ProximityPrecision(str, Enum):
    BY_WORD = "byWord"
    BY_ATTRIBUTE = "byAttribute"


class EmbedderDistribution(CamelBase):
    mean: float
    sigma: float


class LocalizedAttributes(CamelBase):
    attribute_patterns: List[str]
    locales: List[str]


class OpenAiEmbedder(CamelBase):
    source: str = "openAi"
    url: Optional[str] = None
    model: Optional[str] = None  # Defaults to text-embedding-3-small
    dimensions: Optional[int] = None  # Uses the model default
    api_key: Optional[str] = None  # Can be provided through a CLI option or environment variable
    document_template: Optional[str] = None
    document_template_max_bytes: Optional[int] = None  # Default to 400
    distribution: Optional[EmbedderDistribution] = None
    binary_quantized: Optional[bool] = None


class HuggingFaceEmbedder(CamelBase):
    source: str = "huggingFace"
    model: Optional[str] = None  # Defaults to BAAI/bge-base-en-v1.5
    revision: Optional[str] = None
    document_template: Optional[str] = None
    document_template_max_bytes: Optional[int] = None  # Default to 400
    distribution: Optional[EmbedderDistribution] = None
    binary_quantized: Optional[bool] = None


class OllamaEmbedder(CamelBase):
    source: str = "ollama"
    url: Optional[str] = None
    api_key: Optional[str] = None
    model: str
    document_template: Optional[str] = None
    document_template_max_bytes: Optional[int] = None  # Default to 400
    distribution: Optional[EmbedderDistribution] = None
    binary_quantized: Optional[bool] = None


class RestEmbedder(CamelBase):
    source: str = "rest"
    url: str
    api_key: Optional[str]  # required for protected APIs
    document_template: Optional[str] = None
    document_template_max_bytes: Optional[int] = None  # Default to 400
    request: Dict[str, Any]
    response: Dict[str, Any]
    distribution: Optional[EmbedderDistribution] = None
    headers: Optional[Dict[str, Any]] = None
    binary_quantized: Optional[bool] = None


class UserProvidedEmbedder(CamelBase):
    source: str = "userProvided"
    dimensions: int
    distribution: Optional[EmbedderDistribution] = None
    binary_quantized: Optional[bool] = None


class Embedders(CamelBase):
    embedders: Dict[
        str,
        Union[
            OpenAiEmbedder, HuggingFaceEmbedder, OllamaEmbedder, RestEmbedder, UserProvidedEmbedder
        ],
    ]
