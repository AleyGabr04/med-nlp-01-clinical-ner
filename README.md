# MedTag

Clinical NER that tags **diseases** and **chemicals** in biomedical text,
fine-tuning PubMedBERT on BC5CDR.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Settings live in `configs/config.yaml`. Notebooks in `notebooks/` are
Colab-ready: they clone this repo, install it, and call `medtag`.
