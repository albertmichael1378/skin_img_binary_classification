import unittest

from vgg16classifier.components.evaluation import calculate_binary_metrics


class BinaryEvaluationMetricsTests(unittest.TestCase):
    def test_sensitivity_and_specificity_use_configured_positive_index(self):
        metrics = calculate_binary_metrics(
            true_indices=[0, 0, 1, 1, 1],
            predicted_indices=[0, 1, 0, 1, 1],
            positive_index=0,
        )

        self.assertEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 2 / 3)
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertEqual(metrics["true_negative"], 2)
        self.assertEqual(metrics["false_positive"], 1)

    def test_requires_both_classes_in_evaluation_data(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            calculate_binary_metrics(
                true_indices=[0, 0],
                predicted_indices=[0, 1],
                positive_index=0,
            )


if __name__ == "__main__":
    unittest.main()
