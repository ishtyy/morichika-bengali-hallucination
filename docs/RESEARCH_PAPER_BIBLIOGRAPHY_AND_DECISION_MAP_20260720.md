# Research bibliography and decision map

Last verified: 2026-07-20  
Machine-readable BibTeX: `docs/bichar_research_references_20260720.bib`

This ledger records every paper currently influencing the experimental design
or explicitly considered for a training change. A citation is not evidence
that a method improves this Bengali task; measured local ablations remain the
decision authority.

## Directly used in the implemented design

| Reference | Design use | Scope guard |
|---|---|---|
| Gemma 4 Technical Report, arXiv:2607.02770 | backbone architecture/context/runtime description | does not validate our Bengali fine-tune |
| LoRA, arXiv:2106.09685 | parameter-efficient adapter family | adapter rank/targets remain locally measured choices |
| QLoRA, arXiv:2305.14314 | frozen 4-bit backbone with trainable adapters and memory-efficient fine-tuning | quality and memory claims are remeasured on Gemma 4 E4B |
| MiniCPM, arXiv:2404.06395 | motivates WSD for continuous/domain adaptation | its schedule constants are not copied blindly |
| WSD analysis, arXiv:2410.05192 | motivates warmup, stable exploration, and a decayed branch | Bengali heavy-run boundaries must be preregistered and measured |
| Hyperband, JMLR 18(185) | motivates successive halving and allocating longer trajectories only to promising arms | our deterministic small factorial is not claimed to be full Hyperband |
| Guo et al. calibration, arXiv:1706.04599 | scalar temperature scaling, ECE/Brier reporting | current calibration metrics are in-sample diagnostics and are labeled as such |
| Lost in the Middle, TACL 2024 | front/middle/end evidence-position panels and long-context diagnostics | position robustness must be demonstrated on our Bengali held panels |

## Benchmark and Bengali-language lineage

| Reference | Project relevance | Data-use status |
|---|---|---|
| BenHalluEval, arXiv:2605.31483 | Bengali hallucination task taxonomy, dual contextual/closed-book framing, error analysis background | Phase-2 organizers state there are no Phase-2 samples from the paper datasets; paper is background, not a lookup oracle |
| Sakhawat et al., arXiv:2602.12921 | motivates exact figurative/literal idiom distinctions and culturally grounded Bengali evaluation | dataset rights/split admission remains separate; no automatic truth admission from the paper alone |

## Considered but not adopted without a new ablation

| Reference/method | Why it was considered | Current decision |
|---|---|---|
| LoRA+, arXiv:2402.12354 | separate learning rates for LoRA A/B may improve efficiency | not inserted into the winner-bound run; it is a new optimizer factor |
| LoRA-GA, NeurIPS 2024, DOI:10.52202/079017-1741 | gradient-informed LoRA initialization may accelerate early convergence | retained as a separately hash-bound initialization arm; it is absent from the completed vanilla-LoRA step-0/10/20 measurements |
| PiSSA, NeurIPS 2024, DOI:10.52202/079017-3846 | principal-component adapter initialization and QPiSSA may improve early convergence | retained as a separately hash-bound initialization arm; it is absent from the completed vanilla-LoRA step-0/10/20 measurements |
| GaLore, ICML 2024, PMLR 235 | low-rank gradient projection reduces optimizer-state memory while updating full-rank weights | retained as a distinct optimizer/full-parameter ablation; it is neither PiSSA nor LoRA and is not silently combined with the current QLoRA control |
| DPO, arXiv:2305.18290 | preference optimization/reward alternative | not applicable to the current supervised verdict corpus without chosen/rejected preference construction |
| Focal loss, arXiv:1708.02002 | hard-example weighting and class imbalance | not adopted: the sweep curriculum is verdict-balanced and a new loss needs a leakage-safe ablation |
| label smoothing, contrastive loss, GRPO/reward shaping | possible robustness/calibration changes | not adopted based on citation alone; would confound the measured LR/verdict-mass response surface |

## Stable primary links

- https://arxiv.org/abs/2607.02770
- https://arxiv.org/abs/2106.09685
- https://arxiv.org/abs/2305.14314
- https://arxiv.org/abs/2404.06395
- https://arxiv.org/abs/2410.05192
- https://www.jmlr.org/papers/v18/16-558.html
- https://arxiv.org/abs/1706.04599
- https://aclanthology.org/2024.tacl-1.9/
- https://arxiv.org/abs/2605.31483
- https://arxiv.org/abs/2602.12921
- https://arxiv.org/abs/2402.12354
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/62c4718cc334f6a0a62fb81c4a2095a1-Abstract-Conference.html
- https://proceedings.neurips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html
- https://proceedings.mlr.press/v235/zhao24s.html
- https://arxiv.org/abs/2305.18290
- https://arxiv.org/abs/1708.02002

## Paper-writing discipline

- Cite papers for motivation and method provenance; cite local hash-bound
  artifacts for empirical claims.
- State when a result is a one-seed development-panel observation.
- Never describe the 30-row single-source gate as universal source
  generalization.
- Do not call temperature-scaled macro-F1 an independent gain because a
  positive scalar temperature does not change the argmax label.
- Keep demonstrated results, preregistered hypotheses, and rejected methods in
  separate prose/tables.
- Record adapter initialization and optimizer family explicitly. Vanilla LoRA,
  LoRA-GA, PiSSA/QPiSSA, and GaLore are not interchangeable names or evidence.
