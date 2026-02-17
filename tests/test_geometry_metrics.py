import unittest

import numpy as np

from geometry_probe.metrics import (
    RefusalScorer,
    cliff_statistics,
    direction_sensitivity,
    paraphrase_instability,
    predictive_entropy_from_logits,
    top1_top2_logprob_gap,
)


class TestMetrics(unittest.TestCase):
    def test_entropy_and_gap(self):
        logits = [np.array([3.0, 1.0, 0.0]), np.array([0.5, 0.4, 0.1])]
        ent = predictive_entropy_from_logits(logits)
        gap = top1_top2_logprob_gap(logits)
        self.assertEqual(len(ent["token_entropy_values"]), 2)
        self.assertEqual(len(gap["top12_gap_values"]), 2)
        self.assertGreater(gap["top12_gap_mean"], 0.0)

    def test_refusal_regex(self):
        scorer = RefusalScorer(use_embeddings=False)
        s1 = scorer.score("I cannot help with that.")
        s2 = scorer.score("The answer is 42.")
        self.assertGreaterEqual(s1, 0.5)
        self.assertLessEqual(s2, s1)

    def test_instability_and_cliff(self):
        vals = [0.1, 0.2, 0.5, 0.3]
        inst = paraphrase_instability(vals)
        cliff = cliff_statistics(base_value=0.1, edited_values=vals, threshold=0.15)
        self.assertGreater(inst, 0.0)
        self.assertGreaterEqual(cliff["cliff_rate"], 0.0)

    def test_direction_sensitivity(self):
        sens = direction_sensitivity({"a": [0.1, -0.2], "b": [0.01, -0.03]})
        self.assertIn("anisotropy_ratio", sens)
        self.assertGreaterEqual(sens["a"], sens["b"])


if __name__ == "__main__":
    unittest.main()

