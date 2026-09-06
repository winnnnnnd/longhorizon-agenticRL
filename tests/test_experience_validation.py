import unittest

from shopping_grpo.experience.validation import (
    candidate_gate,
    trajectory_key,
    validate_thresholds,
)


class ExperienceValidationTest(unittest.TestCase):
    def test_pair_key_prefers_sampling_seed(self):
        self.assertEqual(
            trajectory_key(
                {"task_id": 7, "attempt_index": 1, "sampling_seed": 43}
            ),
            (7, 43),
        )

    def test_candidate_gate_uses_frozen_safety_and_token_thresholds(self):
        thresholds = validate_thresholds(
            {
                "schema_version": "shopping-experience-validation-thresholds-v1",
                "bootstrap": {"samples": 100, "seed": 42},
                "candidate": {
                    "minimum_targeted_pairs": 2,
                    "strict_success_delta_exclusive_min": 0.0,
                    "strict_success_ci_lower_min": 0.0,
                    "wrong_purchase_increase_max": 0,
                    "reward_unverifiable_increase_max": 0,
                    "guard_rejection_mean_increase_max": 0.0,
                    "repeat_loop_increase_max": 0,
                    "max_steps_increase_max": 0,
                    "injection_tokens_per_turn_max": 500,
                    "selected_experiences_per_turn_max": 3,
                },
                "store": {
                    "minimum_pairs": 2,
                    "strict_success_delta_min": -0.01,
                    "wrong_purchase_increase_max": 0,
                    "reward_unverifiable_increase_max": 0,
                    "guard_rejection_mean_increase_max": 0.0,
                    "repeat_loop_increase_max": 0,
                    "max_steps_increase_max": 0,
                    "infrastructure_invalid_increase_max": 0,
                },
            }
        )["candidate"]
        comparison = {
            "pairs": 2,
            "strict_success": {
                "paired_delta_mean": 0.5,
                "bootstrap_ci95": [0.0, 0.5],
            },
            "count_deltas": {
                "wrong_purchase": 0,
                "reward_unverifiable": 0,
                "repeat_loop": 0,
                "max_steps": 0,
            },
            "guard_rejection_mean_delta": 0.0,
            "treatment_experience": {
                "mean_tokens_per_turn": 300.0,
                "mean_selected_per_turn": 2.0,
            },
        }

        passed, failures = candidate_gate(comparison, thresholds)

        self.assertTrue(passed)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
