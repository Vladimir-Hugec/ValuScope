import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
import logging
import os
import requests
import time
import yfinance as yf

# Set up logging
logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """
    A class to fetch financial data using Yahoo Finance API via yfinance.
    This includes balance sheets, income statements, and cash flow statements.
    """

    def __init__(self, ticker):
        """
        Initialize the YahooFinanceFetcher with a company ticker.

        Args:
            ticker (str): The stock ticker symbol of the company (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker.upper()
        self.financial_data = {}
        self.yf_ticker = yf.Ticker(self.ticker)
        logger.info(f"Initialized YahooFinanceFetcher for {self.ticker}")

    def fetch_balance_sheet(self, quarterly=False):
        """
        Fetch balance sheet data for the company.

        Args:
            quarterly (bool, optional): If True, fetch quarterly data. If False, fetch annual data.

        Returns:
            pd.DataFrame: Balance sheet data
        """
        try:
            logger.info(
                f"Fetching {'quarterly' if quarterly else 'annual'} balance sheet data for {self.ticker}"
            )

            # Get balance sheet data from yfinance
            balance_sheet = (
                self.yf_ticker.balance_sheet
                if not quarterly
                else self.yf_ticker.quarterly_balance_sheet
            )

            if balance_sheet is not None and not balance_sheet.empty:
                # Store the data
                self.financial_data["balance_sheet"] = balance_sheet
                return balance_sheet
            else:
                logger.warning(f"No balance sheet data found for {self.ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching balance sheet for {self.ticker}: {str(e)}")
            return pd.DataFrame()

    def fetch_income_statement(self, quarterly=False):
        """
        Fetch income statement data for the company.

        Args:
            quarterly (bool, optional): If True, fetch quarterly data. If False, fetch annual data.

        Returns:
            pd.DataFrame: Income statement data
        """
        try:
            logger.info(
                f"Fetching {'quarterly' if quarterly else 'annual'} income statement data for {self.ticker}"
            )

            # Get income statement data from yfinance
            income_stmt = (
                self.yf_ticker.income_stmt
                if not quarterly
                else self.yf_ticker.quarterly_income_stmt
            )

            if income_stmt is not None and not income_stmt.empty:
                # Store the data
                self.financial_data["income_statement"] = income_stmt
                return income_stmt
            else:
                logger.warning(f"No income statement data found for {self.ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching income statement for {self.ticker}: {str(e)}")
            return pd.DataFrame()

    def fetch_cash_flow(self, quarterly=False):
        """
        Fetch cash flow statement data for the company.

        Args:
            quarterly (bool, optional): If True, fetch quarterly data. If False, fetch annual data.

        Returns:
            pd.DataFrame: Cash flow statement data
        """
        try:
            logger.info(
                f"Fetching {'quarterly' if quarterly else 'annual'} cash flow statement data for {self.ticker}"
            )

            # Get cash flow data from yfinance
            cash_flow = (
                self.yf_ticker.cashflow
                if not quarterly
                else self.yf_ticker.quarterly_cashflow
            )

            if cash_flow is not None and not cash_flow.empty:
                # Store the data
                self.financial_data["cash_flow"] = cash_flow
                return cash_flow
            else:
                logger.warning(f"No cash flow statement data found for {self.ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(
                f"Error fetching cash flow statement for {self.ticker}: {str(e)}"
            )
            return pd.DataFrame()

    def fetch_all_financial_data(self, quarterly=False):
        """
        Fetch all financial data (balance sheet, income statement, cash flow) for the company.

        Args:
            quarterly (bool, optional): If True, fetch quarterly data. If False, fetch annual data.

        Returns:
            dict: Dictionary containing all financial data DataFrames
        """
        logger.info(f"Fetching all financial data for {self.ticker}")
        self.fetch_balance_sheet(quarterly)
        self.fetch_income_statement(quarterly)
        self.fetch_cash_flow(quarterly)

        return self.financial_data

    def get_historical_prices(self, start_date=None, end_date=None, period="1y"):
        """
        Get historical stock price data.

        Args:
            start_date (datetime, optional): Start date for the data.
            end_date (datetime, optional): End date for the data.
            period (str, optional): Period string for yfinance (e.g. '1d', '1mo', '1y').
                                   Only used if start_date and end_date are None.

        Returns:
            pd.DataFrame: Historical price data
        """
        try:
            logger.info(f"Fetching historical price data for {self.ticker}")

            if start_date and end_date:
                # Convert to string format if datetime objects
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime("%Y-%m-%d")
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime("%Y-%m-%d")

                hist_data = self.yf_ticker.history(start=start_date, end=end_date)
            else:
                hist_data = self.yf_ticker.history(period=period)

            if not hist_data.empty:
                self.financial_data["historical_prices"] = hist_data
                return hist_data
            else:
                logger.warning(f"No historical price data found for {self.ticker}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(
                f"Error fetching historical prices for {self.ticker}: {str(e)}"
            )
            return pd.DataFrame()

    def get_company_info(self):
        """
        Get company information.

        Returns:
            dict: Company information
        """
        try:
            logger.info(f"Fetching company info for {self.ticker}")

            info = self.yf_ticker.info
            if info:
                self.financial_data["company_info"] = info
                return info
            else:
                logger.warning(f"No company info found for {self.ticker}")
                return {}

        except Exception as e:
            logger.error(f"Error fetching company info for {self.ticker}: {str(e)}")
            return {}

    def get_stored_data(self, data_type=None):
        """
        Get the stored financial data.

        Args:
            data_type (str, optional): Type of financial data to return ('balance_sheet', 'income_statement', 'cash_flow').
                                      If None, returns all data.

        Returns:
            pd.DataFrame or dict: The requested financial data
        """
        if data_type is None:
            return self.financial_data
        elif data_type in self.financial_data:
            return self.financial_data[data_type]
        else:
            logger.warning(
                f"Data type '{data_type}' not found in stored financial data"
            )
            return pd.DataFrame()

    def save_data_to_csv(self, directory=".", prefix=None):
        """
        Save all fetched financial data to CSV files.

        Args:
            directory (str): Directory to save files to
            prefix (str, optional): Prefix for filenames, defaults to ticker symbol

        Returns:
            list: List of saved file paths
        """
        import os

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        saved_files = []
        prefix = prefix or self.ticker

        for data_type, data in self.financial_data.items():
            if data_type == "company_info":
                # Convert dictionary to DataFrame for saving
                df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
                filename = os.path.join(directory, f"{prefix}_{data_type}.csv")
                df.to_csv(filename, index=False)
                saved_files.append(filename)
                logger.info(f"Saved {data_type} data to {filename}")
            elif isinstance(data, pd.DataFrame) and not data.empty:
                filename = os.path.join(directory, f"{prefix}_{data_type}.csv")
                data.to_csv(filename)
                saved_files.append(filename)
                logger.info(f"Saved {data_type} data to {filename}")

        return saved_files


# Example usage
if __name__ == "__main__":
    # Make sure to install required packages
    # pip install pandas pandas-datareader yfinance

    # Example usage of the YahooFinanceFetcher
    fetcher = YahooFinanceFetcher("AAPL")

    # Get company info
    company_info = fetcher.get_company_info()
    if company_info:
        print(f"\nCompany Name: {company_info.get('longName', 'N/A')}")
        print(f"Industry: {company_info.get('industry', 'N/A')}")
        print(f"Sector: {company_info.get('sector', 'N/A')}")
        print(f"Market Cap: {company_info.get('marketCap', 'N/A')}")

    # Fetch all financial data for Apple
    financial_data = fetcher.fetch_all_financial_data()

    # Get historical price data
    historical_prices = fetcher.get_historical_prices(period="1y")

    # Print the keys of available data
    print(f"\nAvailable financial data types: {list(financial_data.keys())}")

    # Display a sample of each data type
    for data_type, df in financial_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"\n{data_type.upper()} SAMPLE:")
            print(df.head())

    # Save the data to CSV files
    fetcher.save_data_to_csv()
