import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from valuscope.core.data_fetcher import YahooFinanceFetcher

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestYahooFinanceFetcher(unittest.TestCase):
    """
    Test cases for the YahooFinanceFetcher class.
    """

    def setUp(self):
        """Set up test fixtures."""
        self.ticker = "AAPL"
        self.fetcher = YahooFinanceFetcher(self.ticker)

    def test_init(self):
        """Test the initialization of the YahooFinanceFetcher class."""
        self.assertEqual(self.fetcher.ticker, "AAPL")
        self.assertIsInstance(self.fetcher.financial_data, dict)
        self.assertEqual(len(self.fetcher.financial_data), 0)

    @patch("yfinance.Ticker")
    def test_fetch_balance_sheet(self, mock_ticker):
        """Test fetching the balance sheet."""
        # Create a mock balance sheet
        mock_bs = pd.DataFrame(
            {
                "2023-09-30": [100000000, 50000000],
                "2022-09-30": [90000000, 45000000],
            },
            index=["Total Assets", "Total Liabilities"],
        )

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.balance_sheet = mock_bs
        mock_ticker_instance.quarterly_balance_sheet = mock_bs

        # Test annual balance sheet
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.fetch_balance_sheet(quarterly=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.loc["Total Assets", "2023-09-30"], 100000000)

        # Test that the data was stored in the financial_data dictionary
        self.assertIn("balance_sheet", self.fetcher.financial_data)
        self.assertIs(result, self.fetcher.financial_data["balance_sheet"])

        # Test quarterly balance sheet
        result = self.fetcher.fetch_balance_sheet(quarterly=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("yfinance.Ticker")
    def test_fetch_income_statement(self, mock_ticker):
        """Test fetching the income statement."""
        # Create a mock income statement
        mock_is = pd.DataFrame(
            {
                "2023-09-30": [200000000, 50000000, 30000000],
                "2022-09-30": [180000000, 45000000, 27000000],
            },
            index=["Total Revenue", "Gross Profit", "Net Income"],
        )

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.income_stmt = mock_is
        mock_ticker_instance.quarterly_income_stmt = mock_is

        # Test annual income statement
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.fetch_income_statement(quarterly=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.loc["Total Revenue", "2023-09-30"], 200000000)

        # Test that the data was stored in the financial_data dictionary
        self.assertIn("income_statement", self.fetcher.financial_data)
        self.assertIs(result, self.fetcher.financial_data["income_statement"])

        # Test quarterly income statement
        result = self.fetcher.fetch_income_statement(quarterly=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (3, 2))

    @patch("yfinance.Ticker")
    def test_fetch_cash_flow(self, mock_ticker):
        """Test fetching the cash flow statement."""
        # Create a mock cash flow statement
        mock_cf = pd.DataFrame(
            {
                "2023-09-30": [40000000, -10000000],
                "2022-09-30": [35000000, -9000000],
            },
            index=["Operating Cash Flow", "Capital Expenditure"],
        )

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.cashflow = mock_cf
        mock_ticker_instance.quarterly_cashflow = mock_cf

        # Test annual cash flow statement
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.fetch_cash_flow(quarterly=False)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.loc["Operating Cash Flow", "2023-09-30"], 40000000)

        # Test that the data was stored in the financial_data dictionary
        self.assertIn("cash_flow", self.fetcher.financial_data)
        self.assertIs(result, self.fetcher.financial_data["cash_flow"])

        # Test quarterly cash flow statement
        result = self.fetcher.fetch_cash_flow(quarterly=True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (2, 2))

    @patch("yfinance.Ticker")
    def test_fetch_all_financial_data(self, mock_ticker):
        """Test fetching all financial data at once."""
        # Create mock financial statements
        mock_bs = pd.DataFrame(
            {"2023-09-30": [100000000], "2022-09-30": [90000000]},
            index=["Total Assets"],
        )
        mock_is = pd.DataFrame(
            {"2023-09-30": [200000000], "2022-09-30": [180000000]},
            index=["Total Revenue"],
        )
        mock_cf = pd.DataFrame(
            {"2023-09-30": [40000000], "2022-09-30": [35000000]},
            index=["Operating Cash Flow"],
        )

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.balance_sheet = mock_bs
        mock_ticker_instance.income_stmt = mock_is
        mock_ticker_instance.cashflow = mock_cf

        # Test fetching all financial data
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.fetch_all_financial_data(quarterly=False)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("balance_sheet", result)
        self.assertIn("income_statement", result)
        self.assertIn("cash_flow", result)

    @patch("yfinance.Ticker")
    def test_get_historical_prices(self, mock_ticker):
        """Test fetching historical prices."""
        # Create a mock historical prices dataframe
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        mock_hist = pd.DataFrame(
            {
                "Open": [150.0, 152.0, 153.0, 151.0, 155.0],
                "High": [155.0, 156.0, 157.0, 153.0, 159.0],
                "Low": [148.0, 150.0, 151.0, 149.0, 153.0],
                "Close": [152.0, 153.0, 151.0, 155.0, 158.0],
                "Volume": [1000000, 1100000, 900000, 1200000, 1500000],
            },
            index=dates,
        )

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.history.return_value = mock_hist

        # Test with default period
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.get_historical_prices()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (5, 5))
        self.assertEqual(result.iloc[0]["Close"], 152.0)

        # Test that the data was stored in the financial_data dictionary
        self.assertIn("historical_prices", self.fetcher.financial_data)
        self.assertIs(result, self.fetcher.financial_data["historical_prices"])

        # Test with specific date range
        start_date = "2023-01-01"
        end_date = "2023-01-05"
        result = self.fetcher.get_historical_prices(
            start_date=start_date, end_date=end_date
        )
        mock_ticker_instance.history.assert_called_with(start=start_date, end=end_date)

    @patch("yfinance.Ticker")
    def test_get_company_info(self, mock_ticker):
        """Test fetching company information."""
        # Create a mock company info dictionary
        mock_info = {
            "longName": "Apple Inc.",
            "industry": "Consumer Electronics",
            "sector": "Technology",
            "marketCap": 3000000000000,
            "currentPrice": 190.5,
        }

        # Set up the mock
        mock_ticker_instance = mock_ticker.return_value
        mock_ticker_instance.info = mock_info

        # Test getting company info
        self.fetcher.yf_ticker = mock_ticker_instance
        result = self.fetcher.get_company_info()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["longName"], "Apple Inc.")
        self.assertEqual(result["industry"], "Consumer Electronics")

        # Test that the data was stored in the financial_data dictionary
        self.assertIn("company_info", self.fetcher.financial_data)
        self.assertIs(result, self.fetcher.financial_data["company_info"])

    def test_get_stored_data(self):
        """Test retrieving stored data."""
        # Prepare some test data
        mock_data = {
            "balance_sheet": pd.DataFrame({"2023": [1, 2]}, index=["a", "b"]),
            "income_statement": pd.DataFrame({"2023": [3, 4]}, index=["c", "d"]),
        }
        self.fetcher.financial_data = mock_data

        # Test getting all data
        result = self.fetcher.get_stored_data()
        self.assertEqual(len(result), 2)
        self.assertIn("balance_sheet", result)
        self.assertIn("income_statement", result)

        # Test getting specific data type
        result = self.fetcher.get_stored_data("balance_sheet")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.iloc[0, 0], 1)

        # Test getting non-existent data type
        result = self.fetcher.get_stored_data("non_existent")
        self.assertTrue(result.empty)

    @patch("os.makedirs")
    @patch("pandas.DataFrame.to_csv")
    def test_save_data_to_csv(self, mock_to_csv, mock_makedirs):
        """Test saving data to CSV files."""
        # Prepare some test data
        mock_data = {
            "balance_sheet": pd.DataFrame({"2023": [1, 2]}, index=["a", "b"]),
            "income_statement": pd.DataFrame({"2023": [3, 4]}, index=["c", "d"]),
            "company_info": {"name": "Apple Inc.", "sector": "Technology"},
        }
        self.fetcher.financial_data = mock_data

        # Test saving to default directory
        result = self.fetcher.save_data_to_csv()

        # Check that makedirs was called
        mock_makedirs.assert_called_once()

        # Check that to_csv was called twice for DataFrames
        self.assertEqual(mock_to_csv.call_count, 3)

        # Check the returned list has the right length
        self.assertEqual(len(result), 3)

    def test_error_handling(self):
        """Test error handling in the fetch methods."""
        # Create a fetcher with an invalid ticker
        invalid_fetcher = YahooFinanceFetcher("INVALID")

        # Test that fetch methods return empty DataFrames on error
        result = invalid_fetcher.fetch_balance_sheet()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        result = invalid_fetcher.fetch_income_statement()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        result = invalid_fetcher.fetch_cash_flow()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        result = invalid_fetcher.get_historical_prices()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

        result = invalid_fetcher.get_company_info()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
