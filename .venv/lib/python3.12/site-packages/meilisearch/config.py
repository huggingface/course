from __future__ import annotations

from typing import Optional, Tuple


class Config:
    """
    Client's credentials and configuration parameters
    """

    class Paths:
        health = "health"
        keys = "keys"
        version = "version"
        index = "indexes"
        task = "tasks"
        batch = "batches"
        stat = "stats"
        search = "search"
        facet_search = "facet-search"
        multi_search = "multi-search"
        document = "documents"
        similar = "similar"
        setting = "settings"
        ranking_rules = "ranking-rules"
        distinct_attribute = "distinct-attribute"
        searchable_attributes = "searchable-attributes"
        displayed_attributes = "displayed-attributes"
        stop_words = "stop-words"
        synonyms = "synonyms"
        accept_new_fields = "accept-new-fields"
        filterable_attributes = "filterable-attributes"
        sortable_attributes = "sortable-attributes"
        typo_tolerance = "typo-tolerance"
        dumps = "dumps"
        snapshots = "snapshots"
        pagination = "pagination"
        faceting = "faceting"
        dictionary = "dictionary"
        separator_tokens = "separator-tokens"
        non_separator_tokens = "non-separator-tokens"
        swap = "swap-indexes"
        embedders = "embedders"
        search_cutoff_ms = "search-cutoff-ms"
        proximity_precision = "proximity-precision"
        localized_attributes = "localized-attributes"
        edit = "edit"

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        client_agents: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """
        Parameters
        ----------
        url:
            The url to the Meilisearch API (ex: http://localhost:7700)
        api_key:
            The optional API key to access Meilisearch
        """

        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.client_agents = client_agents
        self.paths = self.Paths()
