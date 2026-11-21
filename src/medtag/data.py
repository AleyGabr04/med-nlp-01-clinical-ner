"""Loading BC5CDR and computing basic dataset statistics."""

import logging
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_bc5cdr(data_cfg: dict) -> DatasetDict:
    """Load the dataset described by the `data` section of the config."""
    dsd = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        revision=data_cfg["revision"],
    )
    logger.info("Loaded %s: %s", data_cfg["dataset_name"], {k: len(v) for k, v in dsd.items()})
    return dsd


def get_label_names(data_cfg: dict) -> list[str]:
    """Return tag names indexed by tag id."""
    return list(data_cfg["label_names"])


def label_counts(ds: Dataset, label_names: list[str], tags_column: str = "tags") -> pd.Series:
    """Count tokens per tag, in label_names order."""
    counts = Counter(tag for tags in ds[tags_column] for tag in tags)
    return pd.Series([counts.get(i, 0) for i in range(len(label_names))], index=label_names, name="tokens")


def sentence_lengths(ds: Dataset, tokens_column: str = "tokens") -> pd.Series:
    """Number of tokens in each sentence."""
    return pd.Series([len(tokens) for tokens in ds[tokens_column]], name="length")


def extract_entities(tag_names: list[str]) -> list[tuple[str, int, int]]:
    """Return (type, start, end) spans from IOB2 tags; end is exclusive.

    A stray I- tag (no preceding B-/I- of the same type) starts a new entity,
    matching seqeval's lenient behaviour.
    """
    entities = []
    current_type, start = None, 0
    for i, tag in enumerate(tag_names):
        prefix, _, ent_type = tag.partition("-")
        continues = prefix == "I" and ent_type == current_type
        if current_type is not None and not continues:
            entities.append((current_type, start, i))
            current_type = None
        if prefix in ("B", "I") and not continues:
            current_type, start = ent_type, i
    if current_type is not None:
        entities.append((current_type, start, len(tag_names)))
    return entities


def entity_counts(ds: Dataset, label_names: list[str], tags_column: str = "tags") -> pd.Series:
    """Count entity spans per entity type."""
    counts = Counter(
        ent_type
        for tags in ds[tags_column]
        for ent_type, _, _ in extract_entities([label_names[t] for t in tags])
    )
    return pd.Series(counts, name="entities").sort_index()


def split_summary(
    dsd: DatasetDict,
    label_names: list[str],
    tokens_column: str = "tokens",
    tags_column: str = "tags",
) -> pd.DataFrame:
    """One row per split: sentence, token, and per-type entity counts."""
    rows = {}
    for split, ds in dsd.items():
        lengths = sentence_lengths(ds, tokens_column)
        rows[split] = {
            "sentences": len(ds),
            "tokens": int(lengths.sum()),
            **entity_counts(ds, label_names, tags_column).to_dict(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
