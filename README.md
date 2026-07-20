# MORICHIKA Bengali Hallucination Detection

Private research and competition package for an offline Bengali hallucination-detection system.

The active production candidates use **Gemma 4 E4B**: a base QAT UD-Q4 model and the same base with the MORICHIKA step-10 LoRA. The exploratory 31B notebook is not part of this release.

This repository contains the MORICHIKA pipeline code, Kaggle notebooks, policy and evaluation documentation, reproducibility receipts, admitted training-data releases, and LoRA adapter checkpoints. Base-model files and redistribution-restricted source corpora are not committed; their immutable identifiers, hashes, rights decisions, and private Kaggle dataset references are recorded instead.

## Runtime contract

- Fully offline Kaggle execution.
- Separate contextual and closed-book evidence semantics.
- Context question/answer alignment, calculations, timelines, relations, Unicode/OCR equivalence, negation, and Bengali grammar-policy handling.
- Closed-book retrieval with books/OCR first, curated datasets second, and Wikipedia/other material only as corroboration after semantic compatibility.
- Hash-versioned caches, counter-evidence, deterministic checkpoints, and no silent default labels.

## Security and rights

The repository is private. Credentials, competition test labels, disallowed leakage artifacts, and source material without redistribution permission are excluded. See `manifests/` for exact artifact hashes and rights/admission ledgers.
