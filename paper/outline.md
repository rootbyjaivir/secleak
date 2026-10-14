# Paper skeleton (fill ONLY from real-data results; nothing below is a result)

**Working title:** How much of "AI detects attacks" survives honest evaluation? An integrity audit of network-IDS benchmarks

## Spine (one sentence, fill after census + experiments)
"X% of surveyed IDS papers use evaluation protocols that leak, and under leak-free evaluation reported F1 drops by Y points (95% CI) on average across Z datasets."

## 1 Introduction
- Problem, why now, spine sentence, 3 contributions (census, audit toolkit + cards, quantified impact)
## 2 Related work (write FIRST; log every close paper in census/SEARCH_LOG.md)
- Arp et al. pitfalls; CICIDS2017 error studies; KDD99 duplication; tabular LLM memorisation (Bordt et al.); PrimeVul-style audits
- Explicit "what is new" paragraph
## 3 Method
- 3.1 Census design (queries, inclusion criteria, codebook, double-coding + kappa)  -> census/codebook.md
- 3.2 Datasets and preprocessing (docs/DATASETS.md)
- 3.3 Protocols: random, random_dedup, temporal_per_class, temporal_global, cross-dataset; justification from deployment
- 3.4 Models and equal tuning budget; validation split built with same protocol
- 3.5 Memorisation probes with controls (optional section)
## 4 Results
- 4.1 Census prevalence (Wilson CIs)
- 4.2 Duplication / overlap / label conflicts (audit cards)
- 4.3 Performance under honest protocols (paired deltas, bootstrap CIs)
- 4.4 Shortcuts
- 4.5 Memorisation (report null results plainly)
## 5 Discussion: recommendations/checklist, threats to validity (census bias, protocol choice, preprocessing)
## 6 Ethics and release: frame as evaluation-practice findings, no individual blame; release code, census data, cards
## Appendix: pre-registration (docs/PREREGISTRATION.md) with any deviations listed
