import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd

from src.exchange import Interval
from src.strategy import TradeAction
from src.trading_agent import TradingAgent


class MockStrategy:
    def __init__(self):
        self.decide = Mock()

    def decide(self, market_data: dict) -> tuple[TradeAction, float, dict]:
        pass


class MockExchange:
    def __init__(self):
        self.execute_trade = Mock()
        self.get_market_data = Mock()
        self.get_current_price = Mock()

    def get_market_data(self, now, count, interval):
        pass

    def get_current_price(self, now):
        pass


class TradingAgentTradeTestCase(unittest.TestCase):
    def test_update_makes_simple_long_trade(self):
        exchange = MockExchange()
        exchange.get_market_data.return_value = pd.DataFrame(
            {
                "timestamp": ["2021-01-01 00:00:00"],
                "Close": [60000],
            }
        )
        exchange.get_current_price.return_value = 60000
        strategy = MockStrategy()
        strategy.decide.return_value = "long", 1.0, {}

        agent = TradingAgent(
            name="test",
            strategy=strategy,
            exchange=exchange,
            initial_capital=1000,
            position_size_percent=0.1,
            min_trade_size=1,
            transaction_fee=0.001,
        )

        agent.update(now="2021-01-01 00:00:00", max_history_count=100, interval="hour")

        exchange.execute_trade.assert_called_once_with(
            {
                "agent": "test",
                "signal": "long",
                "quantity": 0.0016666666666666668,
            }
        )
        self.assertEqual(agent.capital, 899.9)
        self.assertEqual(agent.position, 0.0016666666666666668)

    def test_update_adds_to_position(self):
        exchange = MockExchange()
        exchange.get_market_data.return_value = pd.DataFrame(
            {
                "timestamp": ["2021-01-01 00:00:00"],
                "Close": [60000],
            }
        )
        exchange.get_current_price.return_value = 60000
        strategy = MockStrategy()

        agent = TradingAgent(
            name="test",
            strategy=strategy,
            exchange=exchange,
            initial_capital=1000,
            position_size_percent=0.1,
            min_trade_size=1,
            transaction_fee=0.001,
        )

        strategy.decide.return_value = "long", 1.0, {}
        agent.update(now="2021-01-01 00:00:00", max_history_count=100, interval="hour")

        strategy.decide.return_value = "long", 1.0, {}
        agent.update(now="2021-01-01 00:00:00", max_history_count=100, interval="hour")

        ## assert called twice
        self.assertEqual(exchange.execute_trade.call_count, 2)
        exchange.execute_trade.assert_called_with(
            {
                "agent": "test",
                "signal": "long",
                "quantity": 0.0016666666666666668,
            }
        )
        self.assertEqual(agent.capital, 799.8)
        self.assertEqual(agent.position, 0.0033333333333333335)

    def test_update_does_not_make_trade_for_small_trade_size(self):
        exchange = MockExchange()
        exchange.get_market_data.return_value = pd.DataFrame(
            {
                "timestamp": ["2021-01-01 00:00:00"],
                "Close": [60000],
            }
        )
        exchange.get_current_price.return_value = 60000
        strategy = MockStrategy()
        strategy.decide.return_value = "long", 1.0, {}

        agent = TradingAgent(
            name="test",
            strategy=strategy,
            exchange=exchange,
            initial_capital=1000,
            position_size_percent=0.1,
            min_trade_size=1000,
            transaction_fee=0.001,
        )

        agent.update(now="2021-01-01 00:00:00", max_history_count=100, interval="hour")

        exchange.execute_trade.assert_not_called()
        self.assertEqual(agent.capital, 1000)
        self.assertEqual(agent.position, 0)


class TradingAgentMaxDrawDownTestCase(unittest.TestCase):
    def setUp(self):
        self.base_date = datetime(2024, 1, 1)

    def create_price_data(self, prices):
        dates = [self.base_date + timedelta(days=i) for i in range(len(prices))]
        return pd.DataFrame(
            {
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
                "Volume": [1000] * len(prices),
            },
            index=dates,
        )

    def test_no_trades(self):
        prices = [100.0, 110.0, 120.0]

        exchange = MockExchange()
        exchange.get_market_data.return_value = self.create_price_data(prices)
        strategy = MockStrategy()
        strategy.decide.return_value = "hold", 0, {}
        agent = TradingAgent(
            name="test_agent",
            exchange=exchange,
            strategy=strategy,
            initial_capital=1000,
            position_size_percent=0.1,
            min_trade_size=1,
        )

        for i in range(len(prices)):
            current_time = self.base_date + timedelta(days=i)
            exchange.get_current_price.return_value = prices[i]
            agent.update(current_time, 10, Interval.DAY)

        self.assertEqual(agent.calculate_max_drawdown(), 0.0)

    def test_drawdown_with_recovery(self):
        """Test max drawdown with a price drop and recovery"""
        # Price sequence: 100 -> 80 -> 120
        prices = [100.0, 80.0, 120.0]
        signals = [
            ("long", 1, {}),  # Buy at 100
            ("hold", 0, {}),  # Hold through drop
            ("short", 1, {}),  # Sell at 120
        ]

        exchange = MockExchange()
        exchange.get_market_data.return_value = self.create_price_data(prices)
        strategy = MockStrategy()
        agent = TradingAgent(
            name="test_agent",
            exchange=exchange,
            strategy=strategy,
            initial_capital=1000,
            position_size_percent=1.0,  # Allow full capital usage for clearer math
            min_trade_size=1,
        )

        # Execute trades according to the sequence
        for i in range(len(prices)):
            current_time = self.base_date + timedelta(days=i)
            strategy.decide.return_value = signals[i]
            exchange.get_current_price.return_value = prices[i]
            agent.update(current_time, 10, Interval.DAY)

        # Verify drawdown
        # Initial position: Buy 10 shares at 100 = 1000
        # After drop: 10 shares * 80 = 800 (20% drawdown)
        # After recovery: 10 shares * 120 = 1200
        self.assertAlmostEqual(agent.calculate_max_drawdown(), 0.2, places=2)

    def test_multiple_drawdowns(self):
        """Test max drawdown with multiple drawdown periods"""
        # Price sequence creating multiple drawdowns
        prices = [100.0, 90.0, 110.0, 85.0, 95.0, 80.0, 100.0]
        signals = [
            ("long", 1, {}),  # Buy at 100
            ("hold", 0, {}),  # Hold at 90
            ("hold", 0, {}),  # Hold at 110
            ("hold", 0, {}),  # Hold at 85
            ("hold", 0, {}),  # Hold at 95
            ("hold", 0, {}),  # Hold at 80
            ("short", 1, {}),  # Sell at 100
        ]

        exchange = MockExchange()
        exchange.get_market_data.return_value = self.create_price_data(prices)
        strategy = MockStrategy()
        agent = TradingAgent(
            name="test_agent",
            exchange=exchange,
            strategy=strategy,
            initial_capital=1000,
            position_size_percent=1.0,
            min_trade_size=1,
        )

        for i in range(len(prices)):
            current_time = self.base_date + timedelta(days=i)
            strategy.decide.return_value = signals[i]
            exchange.get_current_price.return_value = prices[i]
            agent.update(current_time, 10, Interval.DAY)

        # Initial position: Buy 10 shares at 100 = 1000
        # After first drop: 10 shares * 90 = 900 (10% drawdown)
        # After recovery: 10 shares * 110 = 1100
        # After second drop: 10 shares * 85 = 850 (22.7% drawdown)
        # After recovery: 10 shares * 95 = 950
        # After third drop: 10 shares * 80 = 800 (27.27% drawdown)
        # After recovery: 10 shares * 100 = 1000
        self.assertAlmostEqual(agent.calculate_max_drawdown(), 0.27, places=2)
