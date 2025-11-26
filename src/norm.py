from typing import Protocol

import pandas as pd


class NormCalculator(Protocol):
    def calculate_norm(self, market_data: pd.DataFrame) -> float:
        """
        Calculate the norm of the market data.

        Args:
            market_data (pd.DataFrame): a DataFrame containing the market data.

        Returns: the norm.
        """
        pass


class IntraNormCalculator(NormCalculator):
    def calculate(self, market_data: pd.DataFrame) -> float:
        intracandle_norm = (
            (market_data["Close"] - market_data["Open"])
            / ((market_data["High"] - market_data["Low"]).replace(0, 1e-6))
        ).mean()

        return intracandle_norm


class InterNormCalculator(NormCalculator):
    def calculate(self, market_data: pd.DataFrame) -> float:
        prev_close = market_data["Close"].shift(1)
        prev_low = market_data["Low"].shift(1)

        intercandle_norm = (
            (
                (market_data["Close"] - prev_close.fillna(method="bfill"))
                / (market_data["High"] - prev_low)
            )
            .fillna(method="bfill")
            .mean()
        )

        return intercandle_norm
