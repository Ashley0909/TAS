import random
import unittest

from geometry_probe.perturbations import (
    EntitySwapOp,
    ParaphraseLiteOp,
    RelationSwapOp,
    TemplateEditOp,
    build_operators,
    generate_perturbations,
)


class TestPerturbations(unittest.TestCase):
    def test_build_operators(self):
        ops = build_operators(["paraphrase_lite", "template_edit"])
        self.assertEqual(len(ops), 2)

    def test_generate_perturbations_count(self):
        ops = build_operators(["paraphrase_lite", "entity_swap", "relation_swap"])
        out = generate_perturbations("Who is Alice's father?", ops, k_per_operator=2, seed=11)
        self.assertEqual(len(out), 6)
        self.assertTrue(all(p.text for p in out))

    def test_relation_swap_changes_relation(self):
        op = RelationSwapOp()
        changed = op.apply("Who is the father of Alice?", random.Random(0))
        self.assertIn("mother", changed.lower())

    def test_template_edit_wraps_prompt(self):
        op = TemplateEditOp()
        prompt = "What is the capital of France?"
        changed = op.apply(prompt, random.Random(0))
        self.assertIn(prompt, changed)

    def test_paraphrase_fallback(self):
        op = ParaphraseLiteOp()
        prompt = "Explain this."
        changed = op.apply(prompt, random.Random(0))
        self.assertNotEqual(prompt, changed)

    def test_entity_swap_changes_text(self):
        op = EntitySwapOp()
        prompt = "Tell me about Alice in Paris."
        changed = op.apply(prompt, random.Random(2))
        self.assertNotEqual(prompt, changed)


if __name__ == "__main__":
    unittest.main()

