from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd

from src.norm import InterNormCalculator, IntraNormCalculator, NormCalculator


class TradeAction(str, Enum):
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class TradingStrategy(Protocol):
    def decide(self, market_data: dict) -> tuple[TradeAction, float, dict]:
        """
        Decide on the action to take based on the market data.

        Args:
            market_data (dict): a dictionary containing the market data.

        Returns: a tuple of action, confidence(0-1), and additional data.
        """
        pass

    def mutate(self, mutation_rate: float) -> None:
        """
        Mutate the strategy.

        Args:
            mutation_rate (float): the rate of mutation.
        """
        pass

    def to_dict(self) -> dict:
        """
        Convert the strategy to a dictionary.

        Returns: a dictionary containing the strategy.
        """
        pass


class ExponentialDecayOHLCVStrategy:
    """
    A strategy that uses exponential decay on OHLCV data to make decisions.

    Args:
        alpha (float): the importance of the relative price difference of open and close to low and high.
        gamma (float): the importance of the inter-interval trend. (close - prev_close) / (high - low)
        epsilon (float): the epsilon parameter
        window_size (int): the window size.
    """

    def __init__(
        self,
        coeffs: list[float],
        gamma: float,
        window_size: int,
        threshold: float,
        norm_calculators: list[NormCalculator] = [
            InterNormCalculator(),
            IntraNormCalculator(),
        ],
    ):
        self.coeffs = coeffs
        self.gamma = gamma
        self.window_size = window_size
        self.threshold = threshold
        self.norm_calculators = norm_calculators

        if (len(coeffs) != len(norm_calculators)) or len(coeffs) == 0:
            raise ValueError(
                "Coefficients and norm calculators must have the same length"
            )

    def decide(self, market_data: pd.DataFrame) -> tuple[TradeAction, float, dict]:
        window_size = min(self.window_size, len(market_data))
        decay_weights = np.exp(-self.gamma * np.arange(window_size)[::-1])

        v_avg = market_data["Volume"].mean()

        sum = 0
        for i, norm_calculator in enumerate(self.norm_calculators):
            norm = norm_calculator.calculate(market_data)
            sum += self.coeffs[i] * norm

        score = sum * decay_weights * market_data["Volume"] / v_avg

        clipped_score = np.clip(score.sum(), -1, 1)

        return (
            TradeAction.LONG
            if clipped_score > self.threshold
            else TradeAction.SHORT
            if clipped_score < -self.threshold
            else TradeAction.HOLD,
            clipped_score,
            {},
        )

    def mutate(self, mutation_rate: float) -> None:
        for i in range(len(self.coeffs)):
            self.coeffs[i] += np.random.normal(0, mutation_rate)
        self.gamma += np.random.normal(0, mutation_rate)
        self.threshold += np.random.normal(0, mutation_rate)
        self.window_size = min(self.window_size + np.random.randint(-1, 2), 2)

    def to_dict(self) -> dict:
        return {
            "type": "exponential_decay_ohlcv",
            "coeffs": self.coeffs,
            "gamma": self.gamma,
            "threshold": self.threshold,
            "window_size": self.window_size,
        }
