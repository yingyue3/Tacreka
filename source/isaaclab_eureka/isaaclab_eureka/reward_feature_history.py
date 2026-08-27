"""Persistence helpers for reward-feature generation and selection history."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


class RewardFeatureHistory:
    """Store generated features, grouped by their measurable signals."""

    def __init__(self, log_dir: str | Path):
        self._root = Path(log_dir) / "reward_features"
        self._root.mkdir(parents=True, exist_ok=True)
        self._history_path = self._root / "reward_feature_history.json"
        self._history_lines_path = self._root / "reward_feature_history.jsonl"
        self._entries: dict[str, dict[str, Any]] = {}
        self._occurrences: dict[tuple[int, int], list[str]] = {}
        self._iteration_feature_sets: dict[int, list[list[dict[str, Any]]]] = {}
        self._load()

    @staticmethod
    def _signal_key(feature: dict[str, Any]) -> str:
        signals = feature.get("measurable_signals") or []
        if isinstance(signals, str):
            signals = [signals]
        normalized = sorted(
            {" ".join(str(signal).strip().lower().split()) for signal in signals if str(signal).strip()}
        )
        if normalized:
            return " | ".join(normalized)
        return f"unspecified::{str(feature.get('feature_name', 'unnamed')).strip().lower()}"

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:80] or "feature"

    @staticmethod
    def _without_weight(feature: dict[str, Any]) -> dict[str, Any]:
        """Return a persisted feature copy without optimization-specific weights."""
        feature_copy = deepcopy(feature)
        feature_copy.pop("weight", None)
        return feature_copy

    def _load(self) -> None:
        if not self._history_path.exists():
            return
        with self._history_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        for entry in payload.get("features", []):
            entry["latest_feature"] = self._without_weight(entry.get("latest_feature", {}))
            entry["variants"] = [self._without_weight(variant) for variant in entry.get("variants", [])]
            self._entries[entry["signal_key"]] = entry

    def add_feature_sets(self, iteration: int, feature_sets: Iterable[list[dict[str, Any]]]) -> None:
        """Save every feature separately and merge it into the signal-keyed list."""
        feature_sets = list(feature_sets)
        self._iteration_feature_sets[iteration] = [
            [self._without_weight(feature) for feature in features if isinstance(feature, dict)]
            for features in feature_sets
        ]
        for set_index, features in enumerate(feature_sets):
            occurrence_keys: list[str] = []
            set_dir = self._root / f"iteration_{iteration:03d}" / f"feature_set_{set_index:03d}"
            set_dir.mkdir(parents=True, exist_ok=True)
            for feature_index, feature in enumerate(features):
                if not isinstance(feature, dict):
                    continue
                feature_copy = self._without_weight(feature)
                signal_key = self._signal_key(feature_copy)
                occurrence_keys.append(signal_key)
                feature_path = set_dir / (
                    f"feature_{feature_index:03d}_{self._safe_name(feature_copy.get('feature_name', 'feature'))}.json"
                )
                with feature_path.open("w", encoding="utf-8") as stream:
                    json.dump(feature_copy, stream, indent=2, ensure_ascii=False)

                entry = self._entries.setdefault(
                    signal_key,
                    {
                        "signal_key": signal_key,
                        "measurable_signals": feature_copy.get("measurable_signals", []),
                        "latest_feature": feature_copy,
                        "generated_count": 0,
                        "variants": [],
                        "selection_history": [],
                    },
                )
                entry["latest_feature"] = feature_copy
                entry["generated_count"] += 1
                if feature_copy not in entry["variants"]:
                    entry["variants"].append(feature_copy)
            self._occurrences[(iteration, set_index)] = occurrence_keys
        self.save()
        self._save_iteration(iteration)

    def record_evaluation(
        self,
        iteration: int,
        selected_set_index: int | None,
        success_metrics: dict[int, float | None],
    ) -> None:
        """Record each set's metric and mark membership in the selected set."""
        for (occurrence_iteration, set_index), signal_keys in self._occurrences.items():
            if occurrence_iteration != iteration:
                continue
            selected = selected_set_index is not None and set_index == selected_set_index
            for signal_key in signal_keys:
                self._entries[signal_key]["selection_history"].append(
                    {
                        "iteration": iteration,
                        "feature_set_index": set_index,
                        "selected": selected,
                        "success_metric": success_metrics.get(set_index),
                    }
                )
        self.save()
        self._save_iteration(iteration, selected_set_index, success_metrics)

    def selection_summary(self) -> dict[str, list[dict[str, Any]]]:
        """Return compact frequency-ranked summaries for the feature-generation prompt."""
        summaries = []
        for entry in self.as_list():
            events = entry["selection_history"]
            selected_count = sum(bool(event["selected"]) for event in events)
            evaluated_count = len(events)
            summaries.append(
                {
                    "signal_key": entry["signal_key"],
                    "feature_name": entry["latest_feature"].get("feature_name"),
                    "selected_count": selected_count,
                    "evaluated_count": evaluated_count,
                    "selection_rate": selected_count / evaluated_count if evaluated_count else 0.0,
                    "selected_success_metrics": [
                        event["success_metric"]
                        for event in events
                        if event["selected"] and event["success_metric"] is not None
                    ],
                    "recent_success_metrics": [
                        event["success_metric"]
                        for event in events[-5:]
                        if event["success_metric"] is not None
                    ],
                    "last_iteration": events[-1]["iteration"] if events else None,
                }
            )
        frequent = sorted(summaries, key=lambda item: (-item["selection_rate"], -item["selected_count"]))
        rare = sorted(summaries, key=lambda item: (item["selection_rate"], -item["evaluated_count"]))
        return {"frequently_chosen": frequent, "rarely_chosen": rare}

    def as_list(self) -> list[dict[str, Any]]:
        return [deepcopy(self._entries[key]) for key in sorted(self._entries)]

    def prompt_section(self, template: str) -> str:
        summary = self.selection_summary()
        evaluated_frequent = [
            item for item in summary["frequently_chosen"] if item["evaluated_count"]
        ][:5]
        evaluated_rare = [
            item for item in summary["rarely_chosen"] if item["evaluated_count"]
        ][:5]
        # Full variants and event histories remain in the JSON/JSONL artifacts.
        # Prompts receive only bounded aggregates to avoid unbounded context growth.
        compact_history = sorted(
            summary["frequently_chosen"],
            key=lambda item: (-(item["last_iteration"] or -1), -item["selected_count"]),
        )[:40]
        return template.format(
            FREQUENTLY_CHOSEN_FEATURES=json.dumps(evaluated_frequent, indent=2, ensure_ascii=False),
            RARELY_CHOSEN_FEATURES=json.dumps(evaluated_rare, indent=2, ensure_ascii=False),
            FEATURE_HISTORY_JSON=json.dumps(compact_history, indent=2, ensure_ascii=False),
        )

    def save(self) -> None:
        with self._history_path.open("w", encoding="utf-8") as stream:
            json.dump(
                {"grouping": "measurable_signals", "features": self.as_list()},
                stream,
                indent=2,
                ensure_ascii=False,
            )
        # One signal-grouped feature per line makes repeated versions and all of
        # their evaluation outcomes inspectable as a single record.
        with self._history_lines_path.open("w", encoding="utf-8") as stream:
            for entry in self.as_list():
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _save_iteration(
        self,
        iteration: int,
        selected_set_index: int | None = None,
        success_metrics: dict[int, float | None] | None = None,
    ) -> None:
        payload = {
            "iteration": iteration,
            "selected_feature_set_index": selected_set_index,
            "feature_sets": [],
        }
        success_metrics = success_metrics or {}
        for set_index, features in enumerate(self._iteration_feature_sets.get(iteration, [])):
            payload["feature_sets"].append(
                {
                    "feature_set_index": set_index,
                    "selected": selected_set_index is not None and set_index == selected_set_index,
                    "success_metric": success_metrics.get(set_index),
                    "features": features,
                }
            )
        iteration_path = self._root / f"iteration_{iteration:03d}" / "all_features.json"
        with iteration_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
