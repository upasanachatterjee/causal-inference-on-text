"""
Shared community-level aggregation utilities.

Used by multi_treatment_dml_sweep.py and mediation_analysis.py.
"""

import warnings
import numpy as np
import pandas as pd


def _normalize(name: str) -> str:
    """Lowercase + underscores — used to match community names to df column names."""
    return name.lower().replace(' ', '_')


def resolve_topic_name(df: pd.DataFrame, name: str) -> str:
    """
    Return the exact topic column suffix in df that matches `name`.

    Matching is case- and separator-insensitive, so 'donald_trump',
    'Donald Trump', and 'DONALD TRUMP' all resolve to whichever form
    appears in the df columns (e.g. 'Donald Trump').

    Works for both regular topics ('Donald Trump') and community columns
    ('community_11') as long as the df has the corresponding 'topic <name>'
    column.

    Raises ValueError if no match is found.
    """
    topic_lookup: dict[str, str] = {}
    for col in df.columns:
        if col.startswith('topic '):
            suffix = col[len('topic '):]
            topic_lookup[_normalize(suffix)] = suffix

    key = _normalize(name)
    if key not in topic_lookup:
        available = ', '.join(sorted(topic_lookup.values()))
        raise ValueError(
            f"Topic '{name}' (normalized: '{key}') not found in df columns. "
            f"Available topics: {available}"
        )
    return topic_lookup[key]


def build_community_df(df: pd.DataFrame, communities: dict,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Replace topic/sentiment columns with community-level aggregates.

    For each community:
      'topic community_<id>'     = 1 if ANY topic in community is present
      'sentiment community_<id>' = mean sentiment of PRESENT topics (0 if none present)

    Topic names in `communities` are matched to df columns via case- and
    separator-insensitive normalization (e.g. 'economy_and_jobs' matches
    'Economy And Jobs', 'fbi' matches 'FBI').  Communities where no topics
    match df columns are skipped with a warning.

    When verbose=True (default) prints a membership table showing matched
    and unmatched topics per community.
    """
    # Build lookup: normalized topic name → actual column suffix after 'topic '
    topic_lookup: dict[str, str] = {}
    for col in df.columns:
        if col.startswith('topic '):
            suffix = col[len('topic '):]
            topic_lookup[_normalize(suffix)] = suffix

    keep_cols = [c for c in df.columns
                 if not c.startswith('topic ') and not c.startswith('sentiment ')]
    result = df[keep_cols].copy()

    for community_id, topics in communities.items():
        matched: list[str] = []    # actual df column suffixes
        unmatched: list[str] = []  # raw names from communities.json with no df column

        for t in topics:
            actual = topic_lookup.get(_normalize(t))
            if actual is not None and f'sentiment {actual}' in df.columns:
                matched.append(actual)
            else:
                unmatched.append(t)

        if verbose:
            cname = f'community_{community_id}'
            print(f"  {cname}: {len(matched)} matched / {len(topics)} total")
            if matched:
                print(f"    matched  : {', '.join(matched)}")
            if unmatched:
                print(f"    unmatched: {', '.join(unmatched)}")

        if not matched:
            warnings.warn(
                f"Community '{community_id}' has no matching columns in df — skipped.",
                UserWarning, stacklevel=2,
            )
            continue

        topic_cols = [f'topic {t}' for t in matched]
        sent_cols  = [f'sentiment {t}' for t in matched]
        cname = f'community_{community_id}'

        # Presence: 1 if any topic in community is present
        result[f'topic {cname}'] = df[topic_cols].any(axis=1).astype(int)

        # Sentiment: mean over present topics only; 0 if community absent
        present_mask = df[topic_cols].values.astype(bool)       # (n, k)
        sent_values  = df[sent_cols].values.astype(float)       # (n, k)
        masked = np.where(present_mask, sent_values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            comm_sent = np.nanmean(masked, axis=1)              # NaN where all absent
        result[f'sentiment {cname}'] = np.nan_to_num(comm_sent, nan=0.0)

    n_built = sum(1 for c in result.columns if c.startswith('topic community_'))
    if n_built == 0:
        warnings.warn(
            "build_community_df produced 0 community columns. "
            "Check that topic names in communities.json match DataFrame column names "
            "(e.g. 'economy_and_jobs' should match 'Economy And Jobs').",
            UserWarning, stacklevel=2,
        )

    return result
