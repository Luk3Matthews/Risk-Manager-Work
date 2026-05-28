"""
Taxonomy loader — reads taxonomy.yaml and provides lookup structures.
"""

from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).parent / "taxonomy.yaml"


class Taxonomy:
    """Loaded taxonomy providing keyword/phrase lookups and mapping rules."""

    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else TAXONOMY_PATH
        with open(path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

        self.themes: dict[str, dict] = self._data.get("themes", {})
        self.macro_factors: dict[str, dict] = self._data.get("macro_factors", {})
        self.mapping_rules: list[dict] = self._data.get("mapping_rules", [])
        self.entities: dict[str, list] = self._data.get("entities", {})
        self.bloomberg_tickers: dict[str, list] = self._data.get("bloomberg_tickers", {})
        self.direction_signals: dict[str, list[str]] = self._data.get("direction_signals", {})
        self.horizon_signals: dict[str, list[str]] = self._data.get("horizon_signals", {})

        # Build fast lookups
        self._keyword_to_themes: dict[str, list[str]] = {}
        self._phrase_to_themes: dict[str, list[str]] = {}
        self._entity_aliases: dict[str, dict] = {}  # alias_lower -> {name, type, ...}
        self._ticker_by_factor: dict[str, list[dict]] = {}
        self._ticker_by_theme: dict[str, list[dict]] = {}

        self._build_keyword_index()
        self._build_entity_index()
        self._build_ticker_index()

    def _build_keyword_index(self):
        for theme_name, theme_data in self.themes.items():
            for kw in theme_data.get("keywords", []):
                kw_lower = kw.lower()
                self._keyword_to_themes.setdefault(kw_lower, []).append(theme_name)
            for phrase in theme_data.get("phrases", []):
                ph_lower = phrase.lower()
                self._phrase_to_themes.setdefault(ph_lower, []).append(theme_name)

    def _build_entity_index(self):
        for entity_type, entity_list in self.entities.items():
            for entity in entity_list:
                name = entity["name"]
                for alias in entity.get("aliases", []):
                    self._entity_aliases[alias.lower()] = {
                        "name": name,
                        "type": entity_type,
                        "region": entity.get("region", ""),
                    }

    def _build_ticker_index(self):
        for category, tickers in self.bloomberg_tickers.items():
            for t in tickers:
                factor = t.get("factor", "")
                if factor:
                    self._ticker_by_factor.setdefault(factor, []).append(t)
                for theme in t.get("themes", []):
                    self._ticker_by_theme.setdefault(theme, []).append(t)

    def get_themes_for_keyword(self, keyword: str) -> list[str]:
        return self._keyword_to_themes.get(keyword.lower(), [])

    def get_themes_for_phrase(self, phrase: str) -> list[str]:
        return self._phrase_to_themes.get(phrase.lower(), [])

    def get_entity(self, text: str) -> dict | None:
        return self._entity_aliases.get(text.lower())

    def get_tickers_for_factor(self, factor: str) -> list[dict]:
        return self._ticker_by_factor.get(factor, [])

    def get_tickers_for_theme(self, theme: str) -> list[dict]:
        return self._ticker_by_theme.get(theme, [])

    def get_all_tickers_flat(self) -> list[dict]:
        """Return all Bloomberg tickers as a flat list."""
        result = []
        for category, tickers in self.bloomberg_tickers.items():
            for t in tickers:
                t_copy = dict(t)
                t_copy["category"] = category
                result.append(t_copy)
        return result

    def get_mapping_rules_for_theme(self, theme: str, direction: str = "up") -> list[dict]:
        """Get factor mapping rules triggered by a given theme+direction."""
        results = []
        for rule in self.mapping_rules:
            if rule["trigger_theme"] == theme and rule["trigger_direction"] == direction:
                results.append(rule)
        return results


# Module-level singleton
_taxonomy: Taxonomy | None = None


def get_taxonomy(path: str | Path | None = None) -> Taxonomy:
    global _taxonomy
    if _taxonomy is None:
        _taxonomy = Taxonomy(path)
    return _taxonomy
