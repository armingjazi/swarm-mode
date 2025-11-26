import unittest

import pandas as pd

from src.norm import InterNormCalculator, IntraNormCalculator


class IntraNormCalculatorTestCase(unittest.TestCase):
    def test_norms(self):
        intra_norm_calculator = IntraNormCalculator()

        market_data = pd.DataFrame(
            {
                "Open": [1, 2, 3, 4, 5],
                "High": [2, 3, 4, 5, 6],
                "Low": [0, 1, 2, 3, 4],
                "Close": [2, 3, 4, 5, 6],
            }
        )

        norm = intra_norm_calculator.calculate(market_data)

        self.assertEqual(norm, 0.5)


class InterNormCalculatorTestCase(unittest.TestCase):
    def test_norms(self):
        inter_norm_calculator = InterNormCalculator()

        market_data = pd.DataFrame(
            {
                "Open": [1, 2, 3, 4, 5],
                "High": [2, 3, 4, 5, 6],
                "Low": [0, 1, 2, 3, 4],
                "Close": [2, 3, 4, 5, 6],
            }
        )

        norm = inter_norm_calculator.calculate(market_data)

        self.assertEqual(norm, 0.3333333333333333)
