"""Discrete timing robustness for fixed, additive flash scorers.

This is an application of established robust optimization, not a novelty claim.
The uncertainty set is centered on received timestamps. A certificate concerns
prediction invariance, not correctness or physiological authenticity.
"""
from dataclasses import dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class MarginResult:
    margin: float
    offsets: tuple
    changes: int
    exact: bool


def _validate(onsets, tables, budget, candidate, rival):
    if not onsets or len(onsets) != len(tables):
        raise ValueError("one nonempty offset table per event required")
    if not isinstance(budget, Integral) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if any(not isinstance(t, Integral) for t in onsets):
        raise ValueError("onsets must be integer samples")
    if any(b <= a for a, b in zip(onsets, onsets[1:])):
        raise ValueError("received events must be strictly ordered")
    width = len(tables[0][0]) if 0 in tables[0] else 0
    if width < 2 or candidate == rival:
        raise ValueError("distinct candidate and rival required")
    if any(not isinstance(k, Integral) or not 0 <= k < width
           for k in (candidate, rival)):
        raise ValueError("invalid symbol index")
    for table in tables:
        if 0 not in table:
            raise ValueError("nominal offset zero must be admissible")
        for offset, values in table.items():
            if not isinstance(offset, Integral) or len(values) != width:
                raise ValueError("integer offsets and consistent symbol count required")
            if any(not isfinite(v) for v in values):
                raise ValueError("non-finite scores cannot be certified")


def independent_margin(onsets, tables, budget, candidate, rival):
    """Exact for independent offsets; lower bound if event order couples them."""
    _validate(onsets, tables, budget, candidate, rival)
    nominal = sum(tab[0][candidate] - tab[0][rival] for tab in tables)
    reductions = []
    for i, tab in enumerate(tables):
        offset = min(tab, key=lambda d: (tab[d][candidate] - tab[d][rival], d != 0, d))
        reduction = tab[0][candidate] - tab[0][rival] - (tab[offset][candidate] - tab[offset][rival])
        reductions.append((max(0., reduction), i, offset))
    offsets = [0] * len(tables)
    chosen = sorted(reductions, reverse=True)[:min(budget, len(tables))]
    for reduction, i, offset in chosen:
        if reduction > 0:
            offsets[i] = offset
    # All admissible adjacent choices must preserve order for independent exactness.
    independent = all(onsets[i] + max(tables[i]) < onsets[i + 1] + min(tables[i + 1])
                      for i in range(len(tables) - 1))
    return MarginResult(nominal - sum(v[0] for v in chosen), tuple(offsets),
                        sum(d != 0 for d in offsets), independent)


def ordered_margin(onsets, tables, budget, candidate, rival):
    """Exact minimum with strict order and at most budget changed events.

    State is (event, changed-event count, current offset). Prefix minima of the
    preceding event give O(F * B * Q log Q) work for at most Q offsets/event.
    Returned witness is feasible. No model, threshold or offset is fitted here.
    """
    _validate(onsets, tables, budget, candidate, rival)
    budget = min(budget, len(tables))
    previous = {}
    parents = []
    for i, table in enumerate(tables):
        current = {}
        parent = {}
        offsets = sorted(table)
        if i == 0:
            for d in offsets:
                count = int(d != 0)
                if count <= budget:
                    current[count, d] = table[d][candidate] - table[d][rival]
                    parent[count, d] = None
        else:
            for count in range(budget + 1):
                # Previous events are sorted by realized timestamp.
                prev_options = sorted((onsets[i - 1] + d, (count, d), value)
                                      for (used, d), value in previous.items() if used == count)
                position = 0
                best_value, best_key = float("inf"), None
                for d in offsets:
                    timestamp = onsets[i] + d
                    while position < len(prev_options) and prev_options[position][0] < timestamp:
                        _, key, value = prev_options[position]
                        if value < best_value:
                            best_value, best_key = value, key
                        position += 1
                    new_count = count + int(d != 0)
                    if best_key is not None and new_count <= budget:
                        key = new_count, d
                        value = best_value + table[d][candidate] - table[d][rival]
                        if value < current.get(key, float("inf")):
                            current[key] = value
                            parent[key] = best_key
        if not current:
            raise ValueError("no admissible ordered assignment")
        previous = current
        parents.append(parent)
    key = min(previous, key=lambda k: (previous[k], k))
    value, used = previous[key], key[0]
    witness = []
    for parent in reversed(parents):
        witness.append(key[1])
        key = parent[key]
    return MarginResult(value, tuple(reversed(witness)), used, True)


def certify(onsets, tables, budget, candidate, ordered=True, tolerance=1e-10):
    """Accept only strictly positive pairwise margins; reject ties/near ties."""
    if not isfinite(tolerance) or tolerance < 0:
        raise ValueError("nonnegative finite numerical tolerance required")
    method = ordered_margin if ordered else independent_margin
    width = len(tables[0][0])
    results = {rival: method(onsets, tables, budget, candidate, rival)
               for rival in range(width) if rival != candidate}
    return all(result.margin > tolerance for result in results.values()), results


def score_cache(signal, onsets, offsets, start_sample, stop_sample, flash_weights, scorer):
    """Build contribution tables from entirely recorded shifted windows.

    signal is a time-major array. scorer(window) is one fixed scalar flash score.
    flash_weights is event x symbol, fixed before evaluation. Bounds use Python
    half-open indexing. Invalid shifts are excluded, never padded or wrapped.
    Filtering of each contiguous recording must occur before calling this API.
    """
    if not isinstance(start_sample, Integral) or not isinstance(stop_sample, Integral) or stop_sample <= start_sample:
        raise ValueError("integer nonempty window required")
    if len(flash_weights) != len(onsets):
        raise ValueError("one fixed symbol-weight row per event required")
    if any(not isinstance(d, Integral) for d in offsets) or 0 not in offsets:
        raise ValueError("integer offsets including zero required")
    tables = []
    for onset, weights in zip(onsets, flash_weights):
        table = {}
        for d in sorted(set(offsets)):
            first, last = onset + d + start_sample, onset + d + stop_sample
            if first < 0 or last > len(signal):
                continue
            score = float(scorer(signal[first:last]))
            table[d] = tuple(score * w for w in weights)
        if 0 not in table:
            raise ValueError("nominal window outside recording; exclude character consistently")
        tables.append(table)
    return tables
