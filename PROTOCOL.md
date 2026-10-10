# Development and confirmation boundaries

Development participants are OpenBMI subjects1–8, session1. Use original training recordings only: first22 characters fit per participant; last11 are development. Subjects9–54 and original test EEG performance remain unexamined for this study. Do not select a method from confirmation outcomes.

The decoder uses fixed flash membership from the pinned public schedule, with identities carried through chronological sorting. Independent phrase/schedule validation must pass. Convert one-based MATLAB markers to zero-based coordinates before100Hz resampling. Filter each contiguous recording once using a fixed0.5–20Hz zero-phase bandpass; do not filter shifted epochs independently. Thirty-two channels, eight100ms mean bins from0–800ms and a−200–0ms baseline form fixed features. This is an own baseline, not the original toolbox reproduction.

Evaluate nominal shrinkage LDA, LDA with training shifts−50/0/+50ms, and a max-score latency comparison using those three offsets. The latter is not a published latency-vector method reproduction. Models use identical fitting/development boundaries.

Offline uncertainty allows at mostB changed flash markers, each shifted by an integer number of100Hz samples withinD. EvaluateB=0/1/3/6 andD=20/50/100/150ms. Waveforms, flash membership, event count and strict chronological order stay fixed; markers of neighboring characters remain fixed. Invalid recorded windows are excluded from the uncertainty set, never padded or wrapped. A certificate with numerical tolerance1e-10 must have all candidate-versus-rival margins strictly positive. A rejected certificate is not automatically a successful targeted attack; replay witnesses in the full multiclass scorer and verify budget, order and raw-window score agreement.

Report clean accuracy, certified coverage, accepted error, wrong-accepted fraction and verified decision-change fraction per person and condition. Keep zero-coverage risk undefined. Compare confidence abstention using nominal top-two margin without oracle error tie-breaking, preserving tied thresholds. Statistical units are participants, not duplicated shifts or correlated flashes.

Before confirmation: audit code/witnesses; implement a published latency-aware comparison; add a fixed neural flash scorer; finalize plausible timing budgets using measurement/source evidence; define success criteria and matched-coverage comparisons; identify an independent character-reconstructable dataset. Freeze a dated confirmation protocol before accessing new participants' test outcomes. Preserve failed approaches and all amendments.

The two-person pilot is feasibility evidence only. Its minimum gaps of10/20ms show coupling can occur, but order-aware DP produced no extra accepted decisions. Do not manufacture a practical DP benefit or a first/new-general-theorem claim.
