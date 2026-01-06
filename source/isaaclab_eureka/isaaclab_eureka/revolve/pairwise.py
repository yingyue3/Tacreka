import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def update_elo(rating_a: float, rating_b: float, result: float, k: float = 32.0) -> Tuple[float, float]:
    """Compute new Elo ratings given a match result for A vs B.

    Args:
        rating_a: Current rating of participant A.
        rating_b: Current rating of participant B.
        result: Match result from A's perspective. 1.0 = A wins, 0.0 = B wins, 0.5 = tie.
        k: Maximum rating change per game.
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_a = rating_a + k * (result - expected_a)
    new_b = rating_b + k * ((1.0 - result) - expected_b)
    return new_a, new_b


class EloRanker:
    """Lightweight Elo tracker for pairwise policy comparisons."""

    def __init__(self, default_rating: float = 1500.0, k: float = 32.0):
        self._ratings: Dict[str, float] = defaultdict(lambda: default_rating)
        self._k = k

    def record_match(self, item_a: str, item_b: str, winner: Optional[str]) -> None:
        """Update ratings given a pairwise result.

        Args:
            item_a: Identifier for first item.
            item_b: Identifier for second item.
            winner: Identifier for the winner. If None, the match is a tie.
        """
        rating_a = self._ratings[item_a]
        rating_b = self._ratings[item_b]
        if winner is None:
            result = 0.5
        elif winner == item_a:
            result = 1.0
        else:
            result = 0.0
        new_a, new_b = update_elo(rating_a, rating_b, result, k=self._k)
        self._ratings[item_a] = new_a
        self._ratings[item_b] = new_b

    @property
    def ratings(self) -> Dict[str, float]:
        return dict(self._ratings)

    def normalized_ratings(self) -> Dict[str, float]:
        """Return ratings normalized to [0, 1] for easy ranking."""
        if not self._ratings:
            return {}
        max_rating = max(self._ratings.values())
        min_rating = min(self._ratings.values())
        if math.isclose(max_rating, min_rating):
            return {k: 0.5 for k in self._ratings}
        return {k: (v - min_rating) / (max_rating - min_rating) for k, v in self._ratings.items()}


def pairwise_preferences_from_metrics(
    scores: List[Tuple[str, float]],
    target: float,
    higher_is_better: bool = True,
) -> List[Tuple[str, str, Optional[str]]]:
    """Build pairwise matchups from a list of (id, score).

    Each consecutive pair is matched. If there is an odd item, it is skipped.
    The winner is determined by closeness to the target, with an optional inversion.
    """
    matches: List[Tuple[str, str, Optional[str]]] = []
    for idx in range(0, len(scores) - 1, 2):
        item_a, score_a = scores[idx]
        item_b, score_b = scores[idx + 1]
        if higher_is_better:
            dist_a = abs(target - score_a)
            dist_b = abs(target - score_b)
        else:
            dist_a = abs(score_a - target)
            dist_b = abs(score_b - target)
        if math.isclose(dist_a, dist_b):
            winner: Optional[str] = None
        elif dist_a < dist_b:
            winner = item_a
        else:
            winner = item_b
        matches.append((item_a, item_b, winner))
    return matches
