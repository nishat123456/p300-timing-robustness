"""Independent exhaustive oracle checks. Run on Kaggle, not local EEG."""
import itertools
import math
import random
import unittest
from p300_timing import ordered_margin, independent_margin, certify, score_cache


def exhaustive(onsets, tables, budget, candidate, rival, enforce_order):
    best = math.inf
    for ds in itertools.product(*(list(t) for t in tables)):
        if sum(d != 0 for d in ds) > budget:
            continue
        shifted = [t + d for t, d in zip(onsets, ds)]
        if enforce_order and any(b <= a for a, b in zip(shifted, shifted[1:])):
            continue
        value = sum(tab[d][candidate] - tab[d][rival] for tab, d in zip(tables, ds))
        best = min(best, value)
    return best


class TimingTests(unittest.TestCase):
    def test_random_exhaustive_oracle(self):
        rng = random.Random(20261010)
        for case in range(300):
            n = rng.randint(1, 5)
            onsets = list(itertools.accumulate(rng.randint(1, 4) for _ in range(n)))
            tables = [{d: tuple(rng.uniform(-2, 2) for _ in range(3))
                       for d in (-1, 0, 1)} for _ in range(n)]
            for budget in range(n + 1):
                for rival in (1, 2):
                    with self.subTest(case=case, budget=budget, rival=rival):
                        exact = ordered_margin(onsets, tables, budget, 0, rival)
                        oracle = exhaustive(onsets, tables, budget, 0, rival, True)
                        self.assertAlmostEqual(exact.margin, oracle, places=10)
                        independent = independent_margin(onsets, tables, budget, 0, rival)
                        self.assertAlmostEqual(independent.margin, exhaustive(onsets, tables, budget, 0, rival, False), places=10)
                        self.assertLessEqual(independent.margin, exact.margin + 1e-10)
                        self.assertLessEqual(exact.changes, budget)
                        ts = [t+d for t,d in zip(onsets,exact.offsets)]
                        self.assertTrue(all(b > a for a,b in zip(ts,ts[1:])))
                        witness = sum(tab[d][0]-tab[d][rival] for tab,d in zip(tables,exact.offsets))
                        self.assertAlmostEqual(witness, oracle, places=10)

    def test_coupling_counterexample(self):
        onsets = [10, 11]
        tables = [{-1: (3., 0.), 0: (3., 0.), 1: (-2., 0.)},
                  {-1: (-2., 0.), 0: (3., 0.), 1: (3., 0.)}]
        loose = independent_margin(onsets,tables,2,0,1)
        exact = ordered_margin(onsets,tables,2,0,1)
        self.assertEqual(loose.margin,-4.)
        self.assertFalse(loose.exact)
        self.assertEqual(exact.margin,1.)
        self.assertTrue(certify(onsets,tables,2,0,ordered=True)[0])
        self.assertFalse(certify(onsets,tables,2,0,ordered=False)[0])

    def test_ties_rejected(self):
        self.assertFalse(certify([10],[{0:(0.,0.)}],0,0)[0])

    def test_independent_spacing_and_ragged_offsets(self):
        times = [10,30,60]
        tabs = [{0:(2.,0.),1:(1.,0.)}, {-1:(1.,0.),0:(3.,0.)}, {0:(3.,0.),2:(2.,0.)}]
        for b in (0,1,2,3,8):
            self.assertTrue(independent_margin(times,tabs,b,0,1).exact)
            self.assertAlmostEqual(ordered_margin(times,tabs,b,0,1).margin,independent_margin(times,tabs,b,0,1).margin)

    def test_invalid_inputs_fail_closed(self):
        for times,tabs,b in [([2,1],[{0:(1.,0.)},{0:(1.,0.)}],1),
                             ([1],[{0:(math.nan,0.)}],1),
                             ([1],[{1:(1.,0.)}],1),
                             ([1],[{0:(1.,0.)}],-1)]:
            with self.assertRaises(ValueError):
                ordered_margin(times,tabs,b,0,1)

    def test_recorded_windows_only(self):
        signal = list(range(8))
        tabs = score_cache(signal,[1,5],[-2,0,2],0,3,[(1.,0.),(0.,1.)],sum)
        self.assertEqual(set(tabs[0]),{0,2})
        self.assertEqual(set(tabs[1]),{-2,0})
        self.assertEqual(tabs[0][0],(6.,0.))
        with self.assertRaises(ValueError):
            score_cache(signal,[7],[0],0,3,[(1.,0.)],sum)


if __name__ == '__main__':
    unittest.main()
