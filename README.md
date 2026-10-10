# P300 timing robustness study

Private development repository. The study evaluates offline sparse marker changes while preserving EEG waveform samples, flash membership and strict temporal order. Certificates protect prediction invariance within the stated uncertainty set, not correctness.

## Current evidence

- Pure mathematical kernel: six synthetic test methods passed on cloud, including 300 seeded cases checked against an exhaustive oracle over budgets and rivals.
- OpenBMI schedule audit: subject1, two sessions, stored labels uniquely match all audited character schedules. Do not sort markers without preserving flash membership. The original analysis phrase independently matches decoded test metadata; no test performance follows from that check.
- Real development pilot: subjects1 and2, session1, original training recordings only,22 fit characters and11 development characters per person. Clean accuracy10/11 and11/11. At50ms shifts and three changed flashes,18/22 decisions are certified, with one retained clean error. Larger budgets can collapse coverage. Ordered dynamic programming adds zero accepted decisions in this pilot.
- Eight-subject comparison: nominal LDA, shift-augmented LDA and latency-max scoring. Running privately on Kaggle; the latter is an implemented comparison, not exact reproduction of published CBLE.

## Data boundary

All EEG processing occurs on Kaggle. Raw recordings live in temporary cloud storage and are deleted after processing. This repository contains source, protocol and compact metadata/results only. Original test EEG is not used in development. No Professor X work belongs in this repository.

## Reproduction

Cloud entrypoints are `p300_development.py` and `p300_model_comparison.py`; they require NumPy, SciPy, scikit-learn and requests. Record code commit, immutable Kaggle version, source hashes, file hashes, split, seeds, environment and logs for every run. Private kernels:

- https://www.kaggle.com/code/mustaqimnishat/p3-timestamp-kernel-check
- https://www.kaggle.com/code/mustaqimnishat/p3a-openbmi-schedule
- https://www.kaggle.com/code/mustaqimnishat/p300-timing-development-pilot
- https://www.kaggle.com/code/mustaqimnishat/p300-timing-model-comparison

## Attribution and publication

Code was drafted with Codex under the user's direction. Research ownership, interpretation, validation, coauthor roles and AI disclosure require review before submission. No Q1/Q2 readiness or acceptance is asserted by this repository. Robust optimization and latency-aware scoring are established ideas; the proposed contribution is conditional on useful new timing-reliability evidence and adequate comparisons.
