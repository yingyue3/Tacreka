import glob
import os
from collections import Counter
from typing import Dict, Optional

import pandas as pd


def update_elo(rating1: float, rating2: float, result: float, k: float = 32.0):
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 - expected1
    new_rating1 = rating1 + k * (result - expected1)
    new_rating2 = rating2 + k * ((1 - result) - expected2)
    return new_rating1, new_rating2


def elo_scores(df: pd.DataFrame) -> Dict[str, float]:
    ratings = {video: 1500 for video in pd.concat([df["Video 1"], df["Video 2"]]).unique()}
    for _, row in df.iterrows():
        video1, video2, selected = row["Video 1"], row["Video 2"], row["Selected"]
        if selected == 1.0:
            result = 1.0
        elif selected == 2.0:
            result = 0.0
        else:
            result = 0.5
        new_elo1, new_elo2 = update_elo(ratings[video1], ratings[video2], result)
        ratings[video1], ratings[video2] = new_elo1, new_elo2

    max_rating = max(ratings.values())
    min_rating = min(ratings.values())
    if max_rating == min_rating:
        return {k: 0.5 for k in ratings}
    normalized_ratings = {k: (v - min_rating) / (max_rating - min_rating) for k, v in ratings.items()}
    return normalized_ratings


def group_feedback(df: pd.DataFrame) -> pd.DataFrame:
    def split_and_add(feedback_list: Optional[str]):
        try:
            return [feedback for feedback in feedback_list.split(", ")]
        except AttributeError:
            return []

    all_videos = set(df["Video 1"].tolist()).union(df["Video 2"].tolist())
    feedback_dict = {k: {"Positive Feedback": [], "Negative Feedback": []} for k in all_videos}
    for _, row in df.iterrows():
        video_1, pos_feedback_1, neg_feedback_1 = (
            row["Video 1"],
            row.get("Positive Feedback 1"),
            row.get("Negative Feedback 1"),
        )
        video_2, pos_feedback_2, neg_feedback_2 = (
            row["Video 2"],
            row.get("Positive Feedback 2"),
            row.get("Negative Feedback 2"),
        )
        feedback_dict[video_1]["Positive Feedback"].extend(split_and_add(pos_feedback_1))
        feedback_dict[video_1]["Negative Feedback"].extend(split_and_add(neg_feedback_1))
        feedback_dict[video_2]["Positive Feedback"].extend(split_and_add(pos_feedback_2))
        feedback_dict[video_2]["Negative Feedback"].extend(split_and_add(neg_feedback_2))

    for k, v in feedback_dict.items():
        all_pos_counter = Counter(v["Positive Feedback"])
        all_pos = [k for k, _ in all_pos_counter.most_common(min(len(all_pos_counter), 2))]
        all_neg_counter = Counter(v["Negative Feedback"])
        all_neg = [k for k, _ in all_neg_counter.most_common(min(len(all_neg_counter), 2))]
        intersection = list(set(all_pos).intersection(set(all_neg)))
        all_pos = [elem for elem in all_pos if elem not in intersection]
        all_neg = [elem for elem in all_neg if elem not in intersection]
        feedback_dict[k] = {"Positive Feedback": all_pos, "Negative Feedback": all_neg}

    feedback_df = pd.DataFrame.from_dict(feedback_dict, orient="index")
    feedback_df.index.name = "Video"
    return feedback_df


def compute_hf_scores(responses_dir: str, generation_id: int) -> Dict[str, float]:
    """Compute normalized Elo scores from human feedback CSVs."""
    response_filename = f"responses_*.csv"
    load_dir = os.path.join(responses_dir, f"generation_{generation_id}")
    response_paths = glob.glob(f"{load_dir}/{response_filename}")
    if generation_id > 0:
        for gid in range(generation_id):
            prev_dir = os.path.join(responses_dir, f"generation_{gid}")
            response_paths += glob.glob(f"{prev_dir}/{response_filename}")

    if len(response_paths) == 0:
        return {}

    df = pd.DataFrame(
        columns=[
            "Video 1",
            "Video 2",
            "Selected",
            "Positive Feedback 1",
            "Negative Feedback 1",
            "Positive Feedback 2",
            "Negative Feedback 2",
        ]
    )
    for response_path in response_paths:
        df_response = pd.read_csv(response_path)
        df = pd.concat([df, df_response], ignore_index=True)

    scores = elo_scores(df)
    return scores
