import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from valuscope.core.data_fetcher import YahooFinanceFetcher
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """
    A class for analyzing financial data of companies.
    """

    def __init__(self, ticker):
        """
        Initialize the FinancialAnalyzer with a company ticker.

        Args:
            ticker (str): The stock ticker symbol of the company (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker.upper()
        self.fetcher = YahooFinanceFetcher(ticker)
        self.data = {}
        logger.info(f"Initialized FinancialAnalyzer for {self.ticker}")

    def fetch_data(self, quarterly=False):
        """
        Fetch all financial data for the company.

        Args:
            quarterly (bool, optional): If True, fetch quarterly data. If False, fetch annual data.

        Returns:
            dict: Dictionary containing all financial data
        """
        # Fetch company info
        company_info = self.fetcher.get_company_info()

        # Fetch financial statements
        financial_data = self.fetcher.fetch_all_financial_data(quarterly)

        # Fetch historical prices
        hist_prices = self.fetcher.get_historical_prices(period="5y")

        # Store all data
        self.data = self.fetcher.get_stored_data()

        return self.data

    def calculate_financial_ratios(self):
        """
        Calculate key financial ratios based on the fetched data.

        Returns:
            pd.DataFrame: DataFrame with calculated financial ratios
        """
        if (
            not self.data
            or "balance_sheet" not in self.data
            or "income_statement" not in self.data
        ):
            logger.warning("Financial data not fetched. Call fetch_data() first.")
            return pd.DataFrame()

        try:
            # Get the relevant data
            balance_sheet = self.data["balance_sheet"]
            income_stmt = self.data["income_statement"]

            # Initialize dictionary to store ratios
            ratios = {}

            # For each year in the data
            for col in balance_sheet.columns:
                year = col

                # Skip if the required data is not available
                if col not in income_stmt.columns:
                    continue

                # Extract data for ratio calculations
                try:
                    # Profitability Ratios
                    total_revenue = (
                        income_stmt.loc["Total Revenue", col]
                        if "Total Revenue" in income_stmt.index
                        else np.nan
                    )
                    net_income = (
                        income_stmt.loc["Net Income", col]
                        if "Net Income" in income_stmt.index
                        else np.nan
                    )

                    # Balance Sheet Items
                    total_assets = (
                        balance_sheet.loc["Total Assets", col]
                        if "Total Assets" in balance_sheet.index
                        else np.nan
                    )
                    total_equity = (
                        balance_sheet.loc["Total Equity Gross Minority Interest", col]
                        if "Total Equity Gross Minority Interest" in balance_sheet.index
                        else np.nan
                    )
                    total_debt = (
                        balance_sheet.loc["Total Debt", col]
                        if "Total Debt" in balance_sheet.index
                        else np.nan
                    )

                    # Calculate ratios
                    if not np.isnan(net_income) and not np.isnan(total_equity):
                        roe = net_income / total_equity  # Return on Equity
                    else:
                        roe = np.nan

                    if not np.isnan(net_income) and not np.isnan(total_assets):
                        roa = net_income / total_assets  # Return on Assets
                    else:
                        roa = np.nan

                    if (
                        not np.isnan(net_income)
                        and not np.isnan(total_revenue)
                        and total_revenue != 0
                    ):
                        profit_margin = net_income / total_revenue  # Net Profit Margin
                    else:
                        profit_margin = np.nan

                    if (
                        not np.isnan(total_debt)
                        and not np.isnan(total_equity)
                        and total_equity != 0
                    ):
                        debt_to_equity = total_debt / total_equity  # Debt to Equity
                    else:
                        debt_to_equity = np.nan

                    # Store the calculated ratios
                    if year not in ratios:
                        ratios[year] = {}

                    ratios[year]["ROE"] = roe
                    ratios[year]["ROA"] = roa
                    ratios[year]["Profit Margin"] = profit_margin
                    ratios[year]["Debt to Equity"] = debt_to_equity

                except Exception as e:
                    logger.error(f"Error calculating ratios for {year}: {str(e)}")

            # Convert to DataFrame
            ratios_df = pd.DataFrame(ratios).T
            return ratios_df

        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}")
            return pd.DataFrame()

    def plot_financial_trends(self, metrics=None, save_path=None):
        """
        Plot financial trends for the company.

        Args:
            metrics (list, optional): List of metrics to plot.
                                     If None, plots all available metrics.
            save_path (str, optional): Path to save the plot.
                                      If None, displays the plot.
        """
        if not self.data or "income_statement" not in self.data:
            logger.warning("Financial data not fetched. Call fetch_data() first.")
            return

        try:
            # Get income statement data
            income_stmt = self.data["income_statement"]

            # If metrics not specified, use some default metrics
            if metrics is None:
                metrics = []
                for metric in ["Total Revenue", "Gross Profit", "Net Income"]:
                    if metric in income_stmt.index:
                        metrics.append(metric)

            # Initialize the plot
            plt.figure(figsize=(12, 6))

            # Plot each metric
            for metric in metrics:
                if metric in income_stmt.index:
                    plt.plot(
                        income_stmt.columns,
                        income_stmt.loc[metric],
                        marker="o",
                        label=metric,
                    )

            # Set plot attributes
            plt.title(f"Financial Trends for {self.ticker}")
            plt.xlabel("Year")
            plt.ylabel("USD")
            plt.legend()
            plt.grid(True)

            # Format y-axis to show in billions
            plt.ticklabel_format(style="plain", axis="y", scilimits=(0, 0))

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)

            # Save or show the plot
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting financial trends: {str(e)}")

    def plot_stock_performance(self, compare_tickers=None, period="1y", save_path=None):
        """
        Plot stock performance compared to benchmark or other companies.

        Args:
            compare_tickers (list, optional): List of tickers to compare with.
            period (str, optional): Period for historical data (e.g., '1d', '1mo', '1y', '5y').
            save_path (str, optional): Path to save the plot.
                                      If None, displays the plot.
        """
        if "historical_prices" not in self.data:
            logger.warning("Historical price data not fetched. Fetching now...")
            self.fetcher.get_historical_prices(period=period)
            self.data = self.fetcher.get_stored_data()

        try:
            # Get historical prices
            hist_prices = self.data["historical_prices"]

            # Initialize the plot
            plt.figure(figsize=(12, 6))

            # Calculate percentage change for the main ticker
            close_prices = hist_prices["Close"]
            normalized = close_prices / close_prices.iloc[0] * 100
            plt.plot(normalized.index, normalized, label=self.ticker)

            # Calculate percentage change for comparison tickers
            if compare_tickers:
                for ticker in compare_tickers:
                    try:
                        # Fetch data for comparison ticker
                        comp_fetcher = YahooFinanceFetcher(ticker)
                        comp_prices = comp_fetcher.get_historical_prices(period=period)

                        if not comp_prices.empty:
                            # Normalize prices to the same starting point
                            comp_close = comp_prices["Close"]
                            comp_normalized = comp_close / comp_close.iloc[0] * 100

                            # Plot the comparison ticker
                            plt.plot(
                                comp_normalized.index, comp_normalized, label=ticker
                            )
                    except Exception as e:
                        logger.error(
                            f"Error fetching comparison data for {ticker}: {str(e)}"
                        )

            # Set plot attributes
            plt.title(f"Stock Performance Comparison: {self.ticker}")
            plt.xlabel("Date")
            plt.ylabel("Normalized Price (%)")
            plt.legend()
            plt.grid(True)

            # Save or show the plot
            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting stock performance: {str(e)}")
