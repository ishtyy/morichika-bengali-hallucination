# All measured model and training-run scores

Cutoff: 2026-07-20 11:45 BDT

These ledgers use different tasks and are not pooled. The Challenge350 and
Kaggle1000 model screens are binary Phase-2 proxy panels. The Gemma training
gates are three-class contextual panels (supported, refuted, NEI). Kaggle
Phase-1 public leaderboard scores are a separate hidden-label benchmark.

## Phase-1 leaderboard control

The current user-locked policy submission scored **0.979 public macro-F1**.
This is the strongest leaderboard artifact, but it is not a trained-model score
and must not be compared numerically with the three-class training gates.

## Raw-model Challenge350 screen

| Model | Overall F1 | Contextual F1 | Closed-book F1 | Projected 5k cold time |
|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B QAT UD-Q4 | 0.825645 | 0.914255 | 0.692716 | 4,714 s |
| Gemma 4 E4B QAT UD-Q4 | 0.824151 | 0.880820 | 0.733672 | 2,933 s |
| Qwen3 30B-A3B Q4_K_M | 0.814029 | 0.885704 | 0.705324 | 6,124 s |
| Qwen3.5 4B Q4_K_M | 0.774136 | 0.857130 | 0.649123 | 3,435 s |
| Sarvam-M 24B Q4_K_M | 0.711488 | 0.771346 | 0.593741 | 7,527 s |
| Nemotron-3 Nano 4B Q4_K_M | 0.533765 | 0.634747 | 0.326923 | 4,552 s |

GPT-OSS 20B produced a zero parse rate under the frozen output contract, so
its apparent zero score is invalid/noncomparable rather than a quality score.

## Gemma E4B Kaggle T4x2 raw-model control

On the locked 1,000-row offline Kaggle run: overall macro-F1 `0.813838`,
contextual macro-F1 `0.903306`, closed-book macro-F1 `0.665963`; 1,000 rows in
`302.305 s` (`3.3079 rows/s`), about `2,895 MiB/GPU`, projected 5,000-row
inference `1,511.5 s` (~25.2 minutes). This is the raw base GGUF, not the
trained adapter.

## Historical Gemma training experiments

| Recipe/checkpoint | Main measured result | Verdict |
|---|---|---|
| raw base, Confirmatory600 direct | macro-F1 0.333333 | control only |
| LR 1e-5, uniform loss, step50 | group/source mean 0.331250; supported collapse | reject |
| LR 2e-5, verdict mass .25, step50 | group/source mean ~0.4325; supported collapse | reject |
| LR 1e-5, verdict mass .25, step50 | group/source mean 0.857434; Confirmatory600 direct 0.669388 | retain as sweep seed |
| same recipe, step100 | group/source mean 0.335678; collapse | reject last checkpoint |

The local70 run-B step25 moved group/source mean from `0.426816` to `0.637521`
(group `0.580171`, source `0.694871`) and long F1 from `0.205128` to
`0.269756`. It nevertheless failed the preregistered gate: long-context
refuted F1 was zero and `promotion_allowed=false`.

The rejected 1e-4 heavy recipe reached its first 2M-token gate at step583:
group F1 `0.158730`, source F1 `0.166667`, mean `0.162698`, calibration F1
`0.158730`, long F1 `0.205128`, with supported/refuted collapse. Evaluation
cost was `$2.220681`; the latest durable checkpoint reached 2,733,514 tokens
and `$2.751411`. Best-not-last correctly retained step0. This historical run
does not authorize any trained release.

## Controlled continuation sweep from the retained step50 seed

| LR / verdict mass | Step | Group F1 | Source F1 | Mean | Calibration F1 | Long F1 |
|---|---:|---:|---:|---:|---:|---:|
| 1e-6 / .25 | 0 | 0.782208 | 0.932660 | 0.857434 | 0.718920 | 0.324444 |
| 1e-6 / .25 | 10 | 0.815670 | 1.000000 | 0.907835 | 0.786875 | 0.389404 |
| 1e-6 / .25 | 20 | 0.815670 | 1.000000 | 0.907835 | 0.817814 | 0.389404 |
| 1e-6 / .45 | 0 | 0.782208 | 0.932660 | 0.857434 | 0.718920 | 0.324444 |
| 1e-6 / .45 | 10 | 0.815670 | 1.000000 | 0.907835 | 0.786875 | 0.389404 |
| 1e-6 / .45 | 20 | 0.815670 | 1.000000 | 0.907835 | 0.847368 | 0.436674 |
| 2e-6 / .25 | 0 | 0.782208 | 0.932660 | 0.857434 | 0.718920 | 0.324444 |
| 2e-6 / .25 | 10 | 0.847368 | 1.000000 | 0.923684 | 0.817814 | 0.444444 |
| 2e-6 / .25 | 20 | 0.815670 | 1.000000 | 0.907835 | 0.786875 | 0.389404 |
| 2e-6 / .45 | 0 | 0.782208 | 0.932660 | 0.857434 | 0.718920 | 0.324444 |
| **2e-6 / .45** | **10** | **0.847368** | **1.000000** | **0.923684** | **0.817814** | **0.570370** |
| 2e-6 / .45 | 20 | 0.815670 | 1.000000 | 0.907835 | 0.817814 | 0.444444 |

The current small-gate winner is `2e-6 / verdict .45 / warmup4 / step10`.
Relative to the shared step0 seed, group/source mean improved `+0.066250` and
long-context F1 improved `+0.245926`.

## Guarded continuation of the two retained arms

| Arm | Step | Group F1 | Source F1 | Mean | Calibration F1 | Long F1 |
|---|---:|---:|---:|---:|---:|---:|
| 2e-6 / .25 | 30 | 0.815670 | 1.000000 | 0.907835 | 0.786875 | 0.389404 |
| 2e-6 / .25 | 40 | 0.815670 | 0.932660 | 0.874165 | 0.754091 | 0.324444 |
| 2e-6 / .45 | 30 | 0.815670 | 1.000000 | 0.907835 | 0.786875 | 0.389404 |
| 2e-6 / .45 | 40 | 0.782208 | 0.966583 | 0.874395 | 0.754091 | 0.324444 |

Both arms stopped at step40 after two material-regression strikes. This was a
quality stop, not an OOM. Step10 remains best-not-last.

## Configurations with no score yet

`1.5e-6`, `2.5e-6`, `3e-6`, verdict masses `.35/.55`, warmup `2/8`, QPiSSA,
BF16 PiSSA, BF16 LoRA-GA, and BF16 vanilla attribution control have **not been
run**. Their executor/spec exists and passes static tests, but reporting a score
before completed remote receipts would be fabricated.

