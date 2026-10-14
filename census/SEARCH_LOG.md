# Search log (fill in; makes the census reproducible)

| date | database | exact query | filters | hits | screened | included |
|---|---|---|---|---|---|---|
| | Google Scholar | "intrusion detection" "CICIDS2017" | 2019-2026 | | | |
| | arXiv | all:"intrusion detection" AND all:"UNSW-NB15" | | | | |
| | IEEE Xplore | ("CIC-IDS2017" OR "UNSW-NB15" OR "BoT-IoT") AND "machine learning" | | | | |

Prior-work sweep queries (novelty check): "intrusion detection" + {data leakage, duplicate records,
temporal split, evaluation pitfalls, label errors, benchmark flaws, realistic evaluation}.

Inclusion: peer-reviewed or arXiv, reports a detection result on >=1 target dataset, English.
Exclusion: surveys without experiments; datasets papers; duplicates of the same work.
Sampling: take the top-N per query by relevance AND a random sample by year to limit popularity bias.
