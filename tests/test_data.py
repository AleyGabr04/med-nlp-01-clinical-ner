import pytest
from datasets import Dataset, DatasetDict

from medtag.data import entity_counts, extract_entities, label_counts, sentence_lengths, split_summary

LABELS = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]


@pytest.fixture
def ds() -> Dataset:
    return Dataset.from_dict(
        {
            "tokens": [
                ["Naloxone", "reverses", "clonidine", "."],
                ["acute", "renal", "failure", "and", "lithium", "carbonate"],
                ["severe", "pain"],
            ],
            "tags": [
                [1, 0, 1, 0],
                [2, 3, 3, 0, 1, 4],
                [0, 3],  # stray I-Disease
            ],
        }
    )


def test_label_counts(ds):
    counts = label_counts(ds, LABELS)
    assert counts.to_dict() == {"O": 4, "B-Chemical": 3, "B-Disease": 1, "I-Disease": 3, "I-Chemical": 1}


def test_sentence_lengths(ds):
    assert sentence_lengths(ds).tolist() == [4, 6, 2]


def test_entity_counts(ds):
    assert entity_counts(ds, LABELS).to_dict() == {"Chemical": 3, "Disease": 2}


@pytest.mark.parametrize(
    "tags, expected",
    [
        (["B-Chemical", "B-Chemical"], [("Chemical", 0, 1), ("Chemical", 1, 2)]),
        (["B-Disease", "I-Chemical"], [("Disease", 0, 1), ("Chemical", 1, 2)]),
        (["O", "I-Disease", "I-Disease"], [("Disease", 1, 3)]),
        (["O", "O"], []),
    ],
)
def test_extract_entities_edge_cases(tags, expected):
    assert extract_entities(tags) == expected


def test_split_summary(ds):
    summary = split_summary(DatasetDict(train=ds), LABELS)
    assert summary.loc["train"].to_dict() == {"sentences": 3, "tokens": 12, "Chemical": 3, "Disease": 2}
