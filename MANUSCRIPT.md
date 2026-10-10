# Decision stability under sparse marker uncertainty in P300 spelling

Development manuscript outline. Not ready for submission. It must not claim invention of BCI timing attacks, robust optimization or latency-aware scoring.

## Question

Can exact, scorer-specific decision-invariance checks provide useful accepted coverage under a clearly bounded offline marker-error model, beyond confidence abstention and latency-aware training/scoring? The important practical outcome is the joint behavior of coverage, accepted errors, attack-induced decision changes and runtime.

## Implemented methods

A fixed additive flash scorer assigns each recorded stimulus a scalar score. Published flash membership is preserved while marker identities are reordered chronologically. Candidate character scores sum fixed event memberships times flash scores. For each allowed integer marker shift, the scorer processes a fully recorded shifted EEG window; neither padding nor circular wrapping creates unobserved samples. The uncertainty budget limits both shift magnitude and the number of changed markers. Strict event order is enforced, and markers outside the evaluated character remain fixed.

For each candidate-versus-rival pair, the independent solution subtracts the largest allowed per-event reductions up to the change budget. When order couples events, an exact dynamic program tracks event position, changed-marker count and last offset, using feasible predecessor minima. This is an application of established discrete optimization, not a new general theorem. Accept only strictly positive pairwise margins at the prespecified numerical tolerance. The resulting statement is prediction invariance within the recorded uncertainty set, not classification correctness or clinical safety.

The initial decoder is shrinkage LDA with fixed continuous preprocessing. The development comparison adds shift-augmented fitting and a latency-max scorer. A published latency-vector method and fixed neural scorer remain necessary comparisons. Witnesses must satisfy the full threat constraints and change the full multiclass decision when EEG windows are independently replayed; a negative pairwise margin alone is not enough to describe a targeted attack.

## Feasibility results

The first two development participants have clean character accuracies10/11 and11/11, respectively. With three changed flashes and a50ms radius,18/22 decisions are accepted and one accepted decision is already incorrect. With six changes at100ms, neither participant retains accepted coverage. All rejected cases in the tested grid have cached-score decision-change witnesses. Ordered optimization provides zero additional acceptances over the independent bound in this pilot. Raw-window replay and stronger comparisons are running in an eight-participant expansion.

These are exploratory results from eleven development characters per person, not independent test performance. Two participants do not support a population claim or a reliable participant-level interval. Repeated shift conditions cannot increase the effective participant count.

## Confirmation requirements

Complete the development comparison and prior-art matrix; freeze relevant budgets, model/threshold choices and success criteria before opening additional participants' test outcomes. Use participant-level paired uncertainty. Verify an independent dataset's character boundaries, original marker identity and recorded support before counting replication. Keep universal abstention and retained incorrect decisions visible. Report all failed methods and results. No Q1/Q2 readiness or acceptance follows from the current feasibility result.

## Sources

Budgeted uncertainty: https://doi.org/10.1287/opre.1030.0065

Prior BCI time-shift attack: https://anil.recoil.org/papers/2022-enhancing-brain-security.pdf

Latency-aware P300 scoring: https://doi.org/10.1080/2326263X.2017.1338010

Original OpenBMI analysis: https://github.com/PatternRecognition/OpenBMI/blob/c0e56825366681ae576801af08eb5fe326a5afd8/GigaScience/Anaylsis_ERP.m
