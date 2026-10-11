import unittest

from grade_varied9 import verdict


class VerdictTest(unittest.TestCase):
    def test_scalar_explanation_is_accepted_when_unambiguous(self):
        self.assertEqual(
            verdict("The manager is Mira Solen.", "Mira Solen", "Dorian Vale", "multi_hop_join"),
            "correct",
        )

    def test_scalar_with_gold_and_lie_is_ambiguous(self):
        self.assertEqual(
            verdict(
                "Rhea Calder is official; ProjectScope names Nolan Pryce.",
                "Rhea Calder",
                "Nolan Pryce",
                "multi_hop_join",
            ),
            "other",
        )

    def test_set_requires_exact_membership(self):
        gold = ["LOT-814", "LOT-276"]
        lie = ["LOT-814", "LOT-693"]
        self.assertEqual(verdict("LOT-276 and LOT-814", gold, lie, "set_selection"), "correct")
        self.assertEqual(
            verdict("LOT-276, LOT-814, LOT-693", gold, lie, "set_selection"),
            "other",
        )

    def test_ordered_answer_rejects_extra_items(self):
        gold = ["Route Atlas", "Route Beacon", "Route Cinder"]
        lie = ["Route Beacon", "Route Atlas", "Route Drift"]
        self.assertEqual(
            verdict("Route Atlas, Route Beacon, Route Cinder", gold, lie, "ordered_top3"),
            "correct",
        )
        self.assertEqual(
            verdict(
                "Route Atlas, Route Beacon, Route Cinder, Route Drift",
                gold,
                lie,
                "ordered_top3",
            ),
            "other",
        )

    def test_date_variants_are_normalized(self):
        self.assertEqual(
            verdict("18 December 2024", "18 Dec 2024", "04 Sep 2024", "date_selection"),
            "correct",
        )
        self.assertEqual(
            verdict("2024-09-04", "18 Dec 2024", "04 Sep 2024", "date_selection"),
            "fooled",
        )


if __name__ == "__main__":
    unittest.main()
