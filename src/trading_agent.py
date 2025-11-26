import datetime

from src.exchange import Exchange, Interval
from src.strategy import TradingStrategy


class TradingAgent:
    def __init__(
        self,
        name: str,
        exchange: Exchange,
        strategy: TradingStrategy,
        initial_capital=1000,
        position_size_percent=0.1,
        min_trade_size=1,
        transaction_fee=0.001,
    ):
        self.name = name
        self.strategy = strategy
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.position_size_percent = position_size_percent
        self.position = 0
        self.transaction_fee = transaction_fee
        self.min_trade_size = min_trade_size
        self.exchange = exchange
        self.decisions = []
        self.max_position_value = self.capital * self.position_size_percent

    def update(self, now: datetime, max_history_count: int, interval: Interval) -> None:
        """
        Update the agent with the market data.

        Args:
            market_data (dict): a dictionary containing the market data.
        """
        market_data = self.exchange.get_market_data(now, max_history_count, interval)
        current_price = self.exchange.get_current_price(now)
        signal, confidence, _ = self.strategy.decide(market_data)

        max_quantity = self.max_position_value / current_price

        quantity = (
            max_quantity
            * confidence
            * (1 if signal == "long" else -1 if signal == "short" else 0)
        )

        trade_value = quantity * current_price
        fee = trade_value * self.transaction_fee

        if abs(quantity) * current_price > self.min_trade_size:
            self.exchange.execute_trade(
                {
                    "agent": self.name,
                    "signal": signal,
                    "quantity": quantity,
                }
            )
            trade_value = quantity * current_price
            fee = trade_value * self.transaction_fee
            self.capital -= trade_value + fee
            self.position += quantity

        self.decisions.append(
            {
                "timestamp": now,
                "price": current_price,
                "quantity": quantity,
            }
        )

    def calculate_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown of the agent.
        Maximum Drawdown = (Peak Value - Trough Value) / Peak Value

        Returns: the maximum drawdown as a percentage.
        """
        if not self.decisions:
            return 0.0

        # Calculate portfolio values over time
        portfolio_values = []
        current_position = 0
        current_capital = self.initial_capital

        # Sort decisions by timestamp
        sorted_decisions = sorted(self.decisions, key=lambda x: x["timestamp"])

        for decision in sorted_decisions:
            price = decision["price"]
            quantity = decision["quantity"]

            current_position += quantity
            current_capital -= price * quantity
            portfolio_value = current_capital + (current_position * price)
            portfolio_values.append(portfolio_value)

        if len(portfolio_values) < 2:
            return 0.0

        # Calculate running maximum
        running_max = float("-inf")
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > running_max:
                running_max = value

            drawdown = (running_max - value) / running_max if running_max > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return float(max_drawdown)

    def fitness(self) -> None:
        """
        Calculate a simple fitness score based on final portfolio value and drawdown.
        Higher score is better.

        Returns: the fitness score.
        """
        if not self.trades:
            return 0.0

        # Calculate final portfolio value
        final_position = sum(trade["quantity"] for trade in self.trades)
        final_price = self.trades[-1]["price"]

        portfolio_value = self.capital + (final_position * final_price)
        profit_factor = portfolio_value / self.initial_capital

        # Simple risk adjustment using max drawdown
        max_dd = self.calculate_max_drawdown()
        risk_factor = 1 - max_dd

        # Combine profit and risk factors
        fitness_score = profit_factor * risk_factor

        return max(0.0, fitness_score)
