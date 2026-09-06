import unittest

from shopping_grpo.experience.task_split import (
    overlap_report,
    stable_experience_task_split,
)


class ExperienceTaskSplitTest(unittest.TestCase):
    def test_split_is_stable_disjoint_and_complete(self):
        first = stable_experience_task_split(range(1, 21), seed=17)
        second = stable_experience_task_split(reversed(range(1, 21)), seed=17)

        self.assertEqual(first, second)
        discovery, candidate_dev = first
        self.assertFalse(set(discovery) & set(candidate_dev))
        self.assertEqual(set(discovery) | set(candidate_dev), set(range(1, 21)))

    def test_overlap_report_exposes_final_evaluation_leakage(self):
        report = overlap_report(
            source_ids=[1, 2, 3],
            discovery_ids=[1, 2],
            candidate_dev_ids=[3],
            final_evaluation_ids=[2],
        )

        self.assertEqual(report["source_vs_final_evaluation"], [2])
        self.assertEqual(report["discovery_vs_final_evaluation"], [2])


if __name__ == "__main__":
    unittest.main()
