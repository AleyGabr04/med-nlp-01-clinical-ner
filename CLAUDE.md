# CLAUDE.md

## Project
MedTag: a clinical NER model that tags DISEASES and CHEMICALS in biomedical
text. Fine-tunes PubMedBERT on BC5CDR. Training runs on Google Colab (GPU);
development happens locally in VS Code.

## How to write code here
- Write code the way an experienced engineer on a small team would: simple,
  readable, and production-ready. Prefer the obvious solution over a clever one.
- Small functions with clear names, type hints, and short docstrings.
- Comments only where the *why* isn't obvious. No tutorial-style comments.
- No unnecessary abstractions, classes, or design patterns. A function is fine.
- Use standard, well-known libraries (transformers, datasets, evaluate, pandas).
- Settings go in one config file, never hardcoded inside functions.
- Use `logging`, not print, inside src/. Notebooks can print.
- Reusable logic goes in src/medtag/. Notebooks only import and call it.

## Working style
- Be concise. Don't explain the code after writing it unless I ask;
  I review it separately.
- Don't create extra docs or summary files unless I ask.
- Ask before adding a new dependency.

## Structure
src/medtag/     package code (data, model, train, evaluate, predict)
notebooks/      Colab notebooks that clone the repo and call src/
tests/          pytest tests for the core logic
configs/        config.yaml