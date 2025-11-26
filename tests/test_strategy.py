import unittest
from unittest.mock import Mock

import pandas as pd

from src.strategy import ExponentialDecayOHLCVStrategy


class CalculatorMock:
    def __init__(self):
        self.calculate = Mock()
        self.calculate.return_value = 1

    def calculate(self, market_data):
        pass


class ExponentialDecayOHLCVStrategyTestCase(unittest.TestCase):
    def test_decide_bullish_frame(self):
        strategy = ExponentialDecayOHLCVStrategy(
            coeffs=[0.1, 0.1],
            gamma=0.9,
            window_size=5,
            threshold=0.3,
            norm_calculators=[CalculatorMock(), CalculatorMock()],
        )

        market_data = pd.DataFrame(
            {
                "timestamp": [
                    "2021-01-01 00:00:00",
                    "2021-01-01 01:00:00",
                    "2021-01-01 02:00:00",
                    "2021-01-01 03:00:00",
                    "2021-01-01 04:00:00",
                ],
                "Open": [1, 2, 3, 4, 5],
                "High": [2, 3, 4, 5, 6],
                "Low": [0, 1, 2, 3, 4],
                "Close": [2, 3, 4, 5, 6],
                "Volume": [1, 2, 3, 4, 5],
            }
        )

        signal, confidence, _ = strategy.decide(market_data)

        self.assertEqual(confidence, 0.4855940034369299)
        self.assertEqual(signal, "long")

    def test_decide_bullish_frame_not_enough_data(self):
        strategy = ExponentialDecayOHLCVStrategy(
            coeffs=[0.1, 0.1],
            gamma=0.9,
            window_size=10,
            threshold=0.3,
            norm_calculators=[CalculatorMock(), CalculatorMock()],
        )
        market_data = pd.DataFrame(
            {
                "timestamp": [
                    "2021-01-01 00:00:00",
                    "2021-01-01 01:00:00",
                    "2021-01-01 02:00:00",
                    "2021-01-01 03:00:00",
                    "2021-01-01 04:00:00",
                ],
                "Open": [1, 2, 3, 4, 5],
                "High": [2, 3, 4, 5, 6],
                "Low": [0, 1, 2, 3, 4],
                "Close": [2, 3, 4, 5, 6],
                "Volume": [1, 2, 3, 4, 5],
            }
        )

        signal, confidence, _ = strategy.decide(market_data)

        self.assertEqual(signal, "long")
        self.assertEqual(confidence, 0.4855940034369299)

    def test_decide_bearish_frame(self):
        mock = CalculatorMock()
        mock.calculate.return_value = -1
        strategy = ExponentialDecayOHLCVStrategy(
            coeffs=[0.5],
            gamma=0.9,
            window_size=10,
            threshold=0.3,
            norm_calculators=[mock],
        )

        market_data = pd.DataFrame(
            {
                "timestamp": [
                    "2021-01-01 00:00:00",
                    "2021-01-01 01:00:00",
                    "2021-01-01 02:00:00",
                    "2021-01-01 03:00:00",
                    "2021-01-01 04:00:00",
                ],
                "Open": [6, 5, 4, 3, 2],
                "High": [7, 6, 5, 4, 3],
                "Low": [5, 4, 3, 2, 1],
                "Close": [7, 6, 5, 4, 3],
                "Volume": [5, 3, 2, 1, 0.5],
            }
        )

        signal, confidence, _ = strategy.decide(market_data)

        self.assertEqual(confidence, -0.34247882318249656)
        self.assertEqual(signal, "short")

    def test_decide_hold_frame(self):
        mock = CalculatorMock()
        mock.calculate.return_value = -0.05
        strategy = ExponentialDecayOHLCVStrategy(
            coeffs=[0.5],
            gamma=0.9,
            window_size=10,
            threshold=0.3,
            norm_calculators=[mock],
        )

        market_data = pd.DataFrame(
            {
                "timestamp": [
                    "2021-01-01 00:00:00",
                    "2021-01-01 01:00:00",
                    "2021-01-01 02:00:00",
                    "2021-01-01 03:00:00",
                    "2021-01-01 04:00:00",
                ],
                "Open": [6, 5, 4, 3, 2],
                "High": [7, 6, 5, 4, 3],
                "Low": [5, 4, 3, 2, 1],
                "Close": [6, 5, 4, 3, 2],
                "Volume": [1, 2, 3, 4, 5],
            }
        )

        signal, confidence, _ = strategy.decide(market_data)

        self.assertEqual(confidence, -0.060699250429616235)
        self.assertEqual(signal, "hold")
