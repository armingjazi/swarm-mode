import datetime
import random

import numpy as np

from src.exchange import Interval
from src.strategy import ExponentialDecayOHLCVStrategy
from src.trading_agent import TradingAgent


class TradingSystem:
    def __init__(
        self,
        initial_population: int,
        generation_lifespan: int = 52,
        interval: Interval = Interval.HOUR,
    ):
        self.population = initial_population
        self.agents: list[TradingAgent] = []
        self.generations = 100
        self.generation_lifespan = generation_lifespan
        self.times = []

        if interval == Interval.MINUTE:
            self.times = np.arange(
                datetime(2020, 1, 1),
                datetime(2025, 1, 1),
                datetime.timedelta(minutes=1),
            ).astype(datetime)
        elif interval == Interval.HOUR:
            self.times = np.arange(
                datetime(2020, 1, 1), datetime(2025, 1, 1), datetime.timedelta(hours=1)
            ).astype(datetime)
        elif interval == Interval.DAY:
            self.times = np.arange(
                datetime(2020, 1, 1),
                datetime(2025, 1, 1),
                datetime.timedelta(days=1),
            ).astype(datetime)

        np.arange(
            datetime(2020, 1, 1), datetime(2025, 1, 1), datetime.timedelta(hours=1)
        ).astype(datetime)

        self.create_initial_population()

    def create_initial_population(self) -> None:
        """
        Create the initial population of agents.
        """
        parameters = []
        for i in range(self.population):
            parameters.append(
                {
                    "alpha": random.random(),
                    "beta": random.random(),
                    "gamma": random.random(),
                    "epsilon": random.random(),
                    "window_size": random.randint(2, 100),
                }
            )

        strategies = [ExponentialDecayOHLCVStrategy(**param) for param in parameters]
        self.agents = [
            TradingAgent(
                name=f"agent_{i}",
                strategy=strategy,
                initial_capital=100,
                position_size_percent=0.1,
                min_trade_size=5,
                transaction_fee=0.0001,
            )
            for i, strategy in enumerate(strategies)
        ]

    def evaluate(self, start_time: datetime) -> None:
        """
        Evaluate the population.
        """
        # every generation, will live trade for 52 timesteps starting from start_time
        timedelta = datetime.timedelta(hours=1)
        if self.interval == Interval.MINUTE:
            timedelta = datetime.timedelta(minutes=1)
        elif self.interval == Interval.DAY:
            timedelta = datetime.timedelta(days=1)

        for _ in range(self.generation_lifespan):
            time = start_time + timedelta
            for agent in self.agents:
                agent.update(time, 100, self.interval)

    def evolve(self) -> None:
        """
        Evolve the population.
        """
        pass
