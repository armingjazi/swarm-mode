import unittest

from src.exchange import LocalBTCExchange


class LocalBTCExchangeTestCase(unittest.TestCase):
    def test_get_marketdata(self):
        exchange = LocalBTCExchange("data/combined_data_X:BTCUSD_hourly_2013_2025.csv")

        market_data = exchange.get_market_data(
            now="2021-01-01 00:00:00", max_history_count=100
        )

        self.assertEqual(market_data.shape, (100, 6))
        self.assertEqual(market_data.iloc[0]["Open"], 26202.37)
        self.assertEqual(market_data.iloc[-1]["Close"], 29066.58)
