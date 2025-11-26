import datetime
from enum import Enum
from typing import Protocol

import pandas as pd


class Interval(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class Exchange(Protocol):
    def get_market_data(
        self, now: datetime, max_history_count: int, interval: Interval
    ) -> dict:
        """
        Get the market data.

        Returns: a dictionary containing the market data.
        """
        pass

    def get_current_price(self) -> float:
        """
        Get the current price.

        Returns: the current price.
        """
        pass

    def get_performance(self) -> dict:
        """
        Get the performance data.

        Returns: a dictionary containing the performance data.
        """
        pass

    def execute_trade(self, trade: dict) -> None:
        """
        Execute a trade.

        Args:
            trade (dict): a dictionary containing the trade data.
        """
        pass


class LocalBTCExchange:
    def __init__(self, path: str):
        # read market data from a file in data folder as pandas dataframe
        self.market_data = pd.read_csv(path)
        self.market_data["timestamp"] = pd.to_datetime(self.market_data["timestamp"])

    def get_market_data(
        self, now: datetime, max_history_count: int, interval: Interval = Interval.HOUR
    ) -> pd.DataFrame:
        return self.market_data[self.market_data["timestamp"] <= now].tail(
            max_history_count
        )

    def get_current_price(self, now: datetime) -> float:
        return self.market_data[self.market_data["timestamp"] == now].iloc[0]["Close"]

    def execute_trade(self, trade: dict) -> None:
        pass
