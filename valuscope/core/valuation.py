import pandas as pd
import numpy as np
from valuscope.core.data_fetcher import YahooFinanceFetcher
import logging
import requests
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


class DCFValuationModel:
    """
    A class for performing Discounted Cash Flow (DCF) valuation on companies.
    """

    def __init__(self, ticker):
        """
        Initialize the DCFValuationModel with a company ticker.

        Args:
            ticker (str): The stock ticker symbol of the company (e.g., 'AAPL' for Apple)
        """
        self.ticker = ticker.upper()
        self.fetcher = YahooFinanceFetcher(ticker)
        self.data = {}
        self.company_info = {}
        self.growth_assumptions = {
            "revenue_growth": 0.05,  # 5% annual growth
            "terminal_growth": 0.025,  # 2.5% terminal growth
            "margin_improvement": 0.002,  # 0.2% annual margin improvement
        }
        self.valuation_parameters = {
            "discount_rate": 0.09,  # 9% discount rate (WACC)
            "projection_years": 5,  # 5-year projection
            "terminal_multiple": 15,  # Terminal EV/EBITDA multiple
        }
        logger.info(f"Initialized DCFValuationModel for {self.ticker}")

    def fetch_data(self):
        """
        Fetch all required financial data for the DCF model.

        Returns:
            dict: Dictionary containing all financial data
        """
        try:
            # Fetch company info
            self.company_info = self.fetcher.get_company_info()

            # Fetch financial statements
            self.fetcher.fetch_all_financial_data(quarterly=False)

            # Store all data
            self.data = self.fetcher.get_stored_data()

            return self.data
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            return {}

    def set_growth_assumptions(self, **kwargs):
        """
        Set growth assumptions for the DCF model.

        Args:
            **kwargs: Key-value pairs of growth assumptions to update
        """
        self.growth_assumptions.update(kwargs)
        logger.info(f"Updated growth assumptions: {self.growth_assumptions}")

    def set_valuation_parameters(self, **kwargs):
        """
        Set valuation parameters for the DCF model.

        Args:
            **kwargs: Key-value pairs of valuation parameters to update
        """
        self.valuation_parameters.update(kwargs)
        logger.info(f"Updated valuation parameters: {self.valuation_parameters}")

    def calculate_current_discount_rate(
        self, use_market_data=True, beta=None, equity_risk_premium=None, tax_rate=0.21
    ):
        """
        Calculate the discount rate (WACC) based on current market data.

        Args:
            use_market_data (bool): Whether to use current market data for risk-free rate
            beta (float, optional): Company's beta, if None will use the one from the fetched data
            equity_risk_premium (float, optional): Market equity risk premium, defaults to 5.5% if None
            tax_rate (float): Corporate tax rate, defaults to 21%

        Returns:
            float: Calculated WACC (discount rate)
        """
        try:
            # Get current risk-free rate from 10-year Treasury
            if use_market_data:
                risk_free_rate = self._get_current_risk_free_rate()
                if risk_free_rate is None:
                    # Fall back to default if couldn't fetch current rate
                    risk_free_rate = 0.04  # 4% as fallback
                    logger.warning(
                        f"Couldn't fetch current risk-free rate, using default: {risk_free_rate}"
                    )
                else:
                    logger.info(f"Using current risk-free rate: {risk_free_rate}")
            else:
                risk_free_rate = 0.04  # Default 4% if not using market data

            # Get beta for the company
            if beta is None:
                beta = self.company_info.get("beta", 1.0)
                if beta is None or np.isnan(beta):
                    beta = 1.0  # Default to market beta if not available
                    logger.warning(
                        f"Beta not available for {self.ticker}, using default: {beta}"
                    )
                else:
                    logger.info(f"Using beta: {beta} for {self.ticker}")

            # Use standard equity risk premium if not provided
            if equity_risk_premium is None:
                equity_risk_premium = 0.055  # 5.5% standard equity risk premium

            # Calculate cost of equity using CAPM
            # Cost of Equity = Risk Free Rate + Beta * Equity Risk Premium
            cost_of_equity = risk_free_rate + beta * equity_risk_premium

            # Get debt information
            total_debt = 0
            if "balance_sheet" in self.data and not self.data["balance_sheet"].empty:
                balance_sheet = self.data["balance_sheet"]
                latest_year = (
                    balance_sheet.columns[0] if not balance_sheet.empty else None
                )

                if latest_year and "Total Debt" in balance_sheet.index:
                    total_debt = balance_sheet.loc["Total Debt", latest_year]
                    if np.isnan(total_debt):
                        total_debt = 0

            # Calculate cost of debt (yield on debt or estimate based on credit quality)
            # For simplicity, estimate cost of debt as risk-free rate + credit spread
            credit_spread = (
                0.02  # Typical credit spread, can be refined based on credit rating
            )
            cost_of_debt = risk_free_rate + credit_spread

            # Calculate after-tax cost of debt
            after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)

            # Calculate market value of equity
            shares_outstanding = self.company_info.get("sharesOutstanding", 0)
            current_price = self.company_info.get("currentPrice", 0)

            if (
                np.isnan(shares_outstanding)
                or np.isnan(current_price)
                or shares_outstanding == 0
                or current_price == 0
            ):
                # If we can't calculate market cap, default to simplified WACC calculation
                logger.warning(
                    f"Market cap not available for {self.ticker}, using simplified WACC calculation"
                )
                # Default weights if we can't calculate them
                weight_of_equity = 0.7  # Typical 70/30 split for many companies
                weight_of_debt = 0.3
            else:
                market_cap = shares_outstanding * current_price

                # Calculate total capital
                total_capital = market_cap + total_debt

                if total_capital == 0:
                    weight_of_equity = 1.0
                    weight_of_debt = 0.0
                else:
                    # Calculate weights
                    weight_of_equity = market_cap / total_capital
                    weight_of_debt = total_debt / total_capital

            # Calculate WACC
            wacc = (weight_of_equity * cost_of_equity) + (
                weight_of_debt * after_tax_cost_of_debt
            )

            logger.info(f"Calculated WACC for {self.ticker}: {wacc:.4f} ({wacc:.2%})")

            # Update valuation parameters with the calculated WACC
            self.valuation_parameters["discount_rate"] = wacc

            return wacc

        except Exception as e:
            logger.error(f"Error calculating current discount rate: {str(e)}")
            # Fall back to default discount rate
            return self.valuation_parameters["discount_rate"]

    def _get_current_risk_free_rate(self):
        """
        Get the current risk-free rate from the 10-year Treasury yield.

        Returns:
            float: Current risk-free rate as decimal
        """
        try:
            # Try to fetch from U.S. Treasury website or financial API
            # For demonstration, we'll use a simple API request
            url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
            params = {
                "filter": "security_desc:eq:Treasury Bonds",
                "sort": "-record_date",
                "limit": 1,
            }

            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        # Convert percent to decimal
                        rate = float(data["data"][0]["avg_interest_rate_amt"]) / 100
                        return rate
            except:
                # If Treasury API fails, try an alternative method
                pass

            # If the above fails, we can try alternative methods:
            # 1. Fixed hard-coded current value (less ideal but works as fallback)
            current_10yr_treasury = 0.0416  # 4.16% as of latest data
            return current_10yr_treasury

        except Exception as e:
            logger.error(f"Error fetching current risk-free rate: {str(e)}")
            return None

    def _extract_historical_financials(self):
        """
        Extract historical financials from the fetched data.

        Returns:
            dict: Dictionary containing historical financial metrics
        """
        if (
            not self.data
            or "income_statement" not in self.data
            or "cash_flow" not in self.data
        ):
            logger.warning("Financial data not fetched. Call fetch_data() first.")
            return {}

        try:
            # Get the relevant dataframes
            income_stmt = self.data["income_statement"]
            cash_flow = self.data["cash_flow"]
            balance_sheet = self.data.get("balance_sheet", pd.DataFrame())

            # Extract key metrics
            historical = {}

            # Get the most recent year (column) from income statement
            latest_year = income_stmt.columns[0] if not income_stmt.empty else None

            if latest_year:
                # Extract key financial metrics
                historical["revenue"] = (
                    income_stmt.loc["Total Revenue", latest_year]
                    if "Total Revenue" in income_stmt.index
                    else np.nan
                )
                historical["ebitda"] = (
                    income_stmt.loc["Normalized EBITDA", latest_year]
                    if "Normalized EBITDA" in income_stmt.index
                    else np.nan
                )
                historical["ebit"] = (
                    income_stmt.loc["EBIT", latest_year]
                    if "EBIT" in income_stmt.index
                    else np.nan
                )
                historical["net_income"] = (
                    income_stmt.loc["Net Income", latest_year]
                    if "Net Income" in income_stmt.index
                    else np.nan
                )

                # Cash flow metrics
                historical["free_cash_flow"] = (
                    cash_flow.loc["Free Cash Flow", latest_year]
                    if "Free Cash Flow" in cash_flow.index
                    else np.nan
                )
                historical["capex"] = (
                    cash_flow.loc["Capital Expenditure", latest_year]
                    if "Capital Expenditure" in cash_flow.index
                    else np.nan
                )

                # Balance sheet metrics
                if not balance_sheet.empty and latest_year in balance_sheet.columns:
                    historical["total_debt"] = (
                        balance_sheet.loc["Total Debt", latest_year]
                        if "Total Debt" in balance_sheet.index
                        else np.nan
                    )
                    historical["cash"] = (
                        balance_sheet.loc["Cash And Cash Equivalents", latest_year]
                        if "Cash And Cash Equivalents" in balance_sheet.index
                        else np.nan
                    )

            # Calculate margins
            if (
                not np.isnan(historical.get("revenue", np.nan))
                and historical["revenue"] > 0
            ):
                if not np.isnan(historical.get("ebitda", np.nan)):
                    historical["ebitda_margin"] = (
                        historical["ebitda"] / historical["revenue"]
                    )
                if not np.isnan(historical.get("ebit", np.nan)):
                    historical["ebit_margin"] = (
                        historical["ebit"] / historical["revenue"]
                    )
                if not np.isnan(historical.get("net_income", np.nan)):
                    historical["net_margin"] = (
                        historical["net_income"] / historical["revenue"]
                    )
                if not np.isnan(historical.get("free_cash_flow", np.nan)):
                    historical["fcf_margin"] = (
                        historical["free_cash_flow"] / historical["revenue"]
                    )

            # Get number of shares
            historical["shares_outstanding"] = self.company_info.get(
                "sharesOutstanding", np.nan
            )

            # Get current stock price
            historical["current_price"] = self.company_info.get("currentPrice", np.nan)

            # Calculate market cap
            if not np.isnan(
                historical.get("shares_outstanding", np.nan)
            ) and not np.isnan(historical.get("current_price", np.nan)):
                historical["market_cap"] = (
                    historical["shares_outstanding"] * historical["current_price"]
                )

            # Calculate enterprise value
            if not np.isnan(historical.get("market_cap", np.nan)):
                ev = historical["market_cap"]
                if not np.isnan(historical.get("total_debt", np.nan)):
                    ev += historical["total_debt"]
                if not np.isnan(historical.get("cash", np.nan)):
                    ev -= historical["cash"]
                historical["enterprise_value"] = ev

            return historical

        except Exception as e:
            logger.error(f"Error extracting historical financials: {str(e)}")
            return {}

    def _project_financials(self, historical):
        """
        Project future financials based on historical data and growth assumptions.

        Args:
            historical (dict): Dictionary containing historical financial metrics

        Returns:
            dict: Dictionary containing projected financial metrics
        """
        if not historical:
            logger.warning("No historical data to project from.")
            return {}

        try:
            # Get growth assumptions
            revenue_growth = self.growth_assumptions["revenue_growth"]
            margin_improvement = self.growth_assumptions["margin_improvement"]
            projection_years = self.valuation_parameters["projection_years"]

            # Initialize projections dictionary
            projections = {"years": [f"Year {i+1}" for i in range(projection_years)]}

            # Project revenue
            if not np.isnan(historical.get("revenue", np.nan)):
                base_revenue = historical["revenue"]
                projected_revenue = []
                for i in range(projection_years):
                    # Calculate revenue with compounding growth
                    next_revenue = base_revenue * (1 + revenue_growth) ** (i + 1)
                    projected_revenue.append(next_revenue)
                projections["revenue"] = projected_revenue

                # Project EBITDA
                if not np.isnan(historical.get("ebitda_margin", np.nan)):
                    base_margin = historical["ebitda_margin"]
                    projected_ebitda = []
                    for i in range(projection_years):
                        # Increasing margin over time
                        margin = min(base_margin + margin_improvement * (i + 1), 0.4)
                        ebitda = projected_revenue[i] * margin
                        projected_ebitda.append(ebitda)
                    projections["ebitda"] = projected_ebitda
                    projections["ebitda_margin"] = [
                        projections["ebitda"][i] / projections["revenue"][i]
                        for i in range(projection_years)
                    ]

                # Project FCF (Free Cash Flow)
                if not np.isnan(historical.get("fcf_margin", np.nan)):
                    base_fcf_margin = historical["fcf_margin"]
                    projected_fcf = []
                    for i in range(projection_years):
                        # Gradually improving FCF margin
                        margin = min(
                            base_fcf_margin + margin_improvement * (i + 1), 0.35
                        )
                        fcf = projected_revenue[i] * margin
                        projected_fcf.append(fcf)
                    projections["fcf"] = projected_fcf
                    projections["fcf_margin"] = [
                        projections["fcf"][i] / projections["revenue"][i]
                        for i in range(projection_years)
                    ]
                # If FCF margin not available, estimate from EBITDA
                elif "ebitda" in projections:
                    # Assume capex to revenue ratio similar to historical
                    capex_to_revenue = (
                        abs(historical.get("capex", 0)) / historical["revenue"]
                        if not np.isnan(historical.get("capex", np.nan))
                        and not np.isnan(historical.get("revenue", np.nan))
                        and historical["revenue"] > 0
                        else 0.05  # Default 5% capex to revenue
                    )
                    projected_fcf = []
                    for i in range(projection_years):
                        # Simplified FCF calculation: EBITDA - Capex
                        capex = projected_revenue[i] * capex_to_revenue
                        fcf = projections["ebitda"][i] - capex
                        projected_fcf.append(fcf)
                    projections["fcf"] = projected_fcf
                    projections["fcf_margin"] = [
                        projections["fcf"][i] / projections["revenue"][i]
                        for i in range(projection_years)
                    ]

            return projections

        except Exception as e:
            logger.error(f"Error projecting financials: {str(e)}")
            return {}

    def _calculate_terminal_value(self, projections):
        """
        Calculate the terminal value for the DCF model.

        Args:
            projections (dict): Dictionary containing projected financial metrics

        Returns:
            float: Terminal value
        """
        if not projections or "fcf" not in projections or not projections["fcf"]:
            logger.warning("No projections data to calculate terminal value from.")
            return np.nan

        try:
            # Get the terminal growth rate and discount rate
            terminal_growth = self.growth_assumptions["terminal_growth"]
            discount_rate = self.valuation_parameters["discount_rate"]
            terminal_multiple = self.valuation_parameters.get("terminal_multiple", 15)

            # Get the last projected FCF
            last_fcf = projections["fcf"][-1]

            # Calculate terminal value using perpetuity growth method
            # Terminal Value = FCF(n+1) / (r - g)
            terminal_value_pg = (last_fcf * (1 + terminal_growth)) / (
                discount_rate - terminal_growth
            )

            # Calculate terminal value using exit multiple method
            # Terminal Value = EBITDA(n) * Multiple
            if "ebitda" in projections and projections["ebitda"]:
                last_ebitda = projections["ebitda"][-1]
                terminal_value_multiple = last_ebitda * terminal_multiple
            else:
                terminal_value_multiple = np.nan

            # Use the perpetuity growth method as default terminal value
            terminal_value = terminal_value_pg

            # If both methods available, use average
            if not np.isnan(terminal_value_multiple):
                terminal_value = (terminal_value_pg + terminal_value_multiple) / 2

            return terminal_value

        except Exception as e:
            logger.error(f"Error calculating terminal value: {str(e)}")
            return np.nan

    def perform_dcf_valuation(self, use_current_discount_rate=False):
        """
        Perform Discounted Cash Flow valuation.

        Args:
            use_current_discount_rate (bool): Whether to calculate discount rate
                                            based on current market data

        Returns:
            dict: Dictionary containing valuation results
        """
        # Check if data has been fetched
        if not self.data:
            logger.warning("No data available. Call fetch_data() first.")
            return None

        try:
            # If requested, calculate the discount rate based on current market data
            if use_current_discount_rate:
                self.calculate_current_discount_rate()
                logger.info(
                    f"Using dynamically calculated discount rate: {self.valuation_parameters['discount_rate']:.2%}"
                )
            else:
                logger.info(
                    f"Using predefined discount rate: {self.valuation_parameters['discount_rate']:.2%}"
                )

            # Extract historical financials
            historical = self._extract_historical_financials()
            if not historical:
                logger.warning("Could not extract historical financials.")
                return None

            # Project future financials
            projections = self._project_financials(historical)
            if not projections:
                logger.warning("Could not project future financials.")
                return None

            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(projections)
            if np.isnan(terminal_value):
                logger.warning("Could not calculate terminal value.")
                return None

            # Get discount rate
            discount_rate = self.valuation_parameters["discount_rate"]
            projection_years = self.valuation_parameters["projection_years"]

            # Calculate present value of projected FCFs
            pv_fcf = []
            if "fcf" in projections:
                for i, fcf in enumerate(projections["fcf"]):
                    pv = fcf / ((1 + discount_rate) ** (i + 1))
                    pv_fcf.append(pv)

            # Calculate present value of terminal value
            pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)

            # Calculate enterprise value
            enterprise_value = sum(pv_fcf) + pv_terminal

            # Calculate equity value
            equity_value = enterprise_value
            if not np.isnan(historical.get("total_debt", np.nan)):
                equity_value -= historical["total_debt"]
            if not np.isnan(historical.get("cash", np.nan)):
                equity_value += historical["cash"]

            # Calculate per share value
            per_share_value = np.nan
            if (
                not np.isnan(historical.get("shares_outstanding", np.nan))
                and historical["shares_outstanding"] > 0
            ):
                per_share_value = equity_value / historical["shares_outstanding"]

            # Calculate upside potential
            upside_potential = np.nan
            if (
                not np.isnan(per_share_value)
                and not np.isnan(historical.get("current_price", np.nan))
                and historical["current_price"] > 0
            ):
                upside_potential = (
                    per_share_value - historical["current_price"]
                ) / historical["current_price"]

            # Prepare results
            results = {
                "historical": historical,
                "projections": projections,
                "terminal_value": terminal_value,
                "pv_fcf": pv_fcf,
                "pv_terminal": pv_terminal,
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "per_share_value": per_share_value,
                "current_price": historical.get("current_price", np.nan),
                "upside_potential": upside_potential,
                "discount_rate": discount_rate,
                "risk_free_rate": (
                    self._get_current_risk_free_rate()
                    if use_current_discount_rate
                    else None
                ),
            }

            return results

        except Exception as e:
            logger.error(f"Error performing DCF valuation: {str(e)}")
            return None

    def display_valuation_results(self, results):
        """
        Display the results of the DCF valuation.

        Args:
            results (dict): Dictionary containing valuation results
        """
        if not results:
            print("No valuation results available.")
            return

        try:
            # Helper function for formatting currency values
            def format_currency(value):
                if np.isnan(value):
                    return "N/A"
                if abs(value) >= 1e9:
                    return f"${value/1e9:.2f}B"
                elif abs(value) >= 1e6:
                    return f"${value/1e6:.2f}M"
                else:
                    return f"${value:,.2f}"

            # Helper function for formatting percentages
            def format_percentage(value):
                if np.isnan(value):
                    return "N/A"
                return f"{value:.2%}"

            # Print company info
            print(f"\n{'=' * 50}")
            print(f"{self.ticker} - DCF Valuation Results")
            print(f"{'=' * 50}")

            # Print historical financials
            historical = results["historical"]
            print("\nHistorical Financials:")
            print(f"Revenue: {format_currency(historical.get('revenue', np.nan))}")
            print(f"EBITDA: {format_currency(historical.get('ebitda', np.nan))}")
            print(
                f"EBITDA Margin: {format_percentage(historical.get('ebitda_margin', np.nan))}"
            )
            print(f"Free Cash Flow: {format_currency(historical.get('fcf', np.nan))}")
            print(
                f"FCF Margin: {format_percentage(historical.get('fcf_margin', np.nan))}"
            )

            # Print growth assumptions
            print("\nGrowth Assumptions:")
            print(
                f"Revenue Growth: {format_percentage(self.growth_assumptions['revenue_growth'])}"
            )
            print(
                f"Terminal Growth: {format_percentage(self.growth_assumptions['terminal_growth'])}"
            )
            print(
                f"Margin Improvement: {format_percentage(self.growth_assumptions['margin_improvement'])}"
            )

            # Print valuation parameters
            print("\nValuation Parameters:")
            print(
                f"Discount Rate (WACC): {format_percentage(self.valuation_parameters['discount_rate'])}"
            )
            print(
                f"Projection Years: {self.valuation_parameters['projection_years']} years"
            )
            print(
                f"Terminal Multiple: {self.valuation_parameters.get('terminal_multiple', 'N/A')}x"
            )

            # Print DCF components
            print("\nDCF Components:")
            print(
                f"PV of Projected FCF: {format_currency(sum(results.get('pv_fcf', [0])))}"
            )
            print(
                f"PV of Terminal Value: {format_currency(results.get('pv_terminal', np.nan))}"
            )
            print(
                f"Enterprise Value: {format_currency(results.get('enterprise_value', np.nan))}"
            )
            print(
                f"Equity Value: {format_currency(results.get('equity_value', np.nan))}"
            )

            # Print per share value and upside
            print("\nValuation Results:")
            print(
                f"Implied Share Value: {format_currency(results.get('per_share_value', np.nan))}"
            )
            print(
                f"Current Market Price: {format_currency(results.get('current_price', np.nan))}"
            )
            print(
                f"Upside Potential: {format_percentage(results.get('upside_potential', np.nan))}"
            )

            # Print investment recommendation
            print("\nInvestment Recommendation:")
            upside = results.get("upside_potential", np.nan)
            if np.isnan(upside):
                recommendation = "N/A - insufficient data"
            elif upside > 0.2:
                recommendation = "Strong Buy"
            elif upside > 0.05:
                recommendation = "Buy"
            elif upside > -0.05:
                recommendation = "Hold"
            elif upside > -0.2:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            print(f"Recommendation: {recommendation}")

            print(f"\n{'=' * 50}\n")

        except Exception as e:
            logger.error(f"Error displaying valuation results: {str(e)}")

    def perform_sensitivity_analysis(
        self, variable1, values1, variable2=None, values2=None
    ):
        """
        Perform sensitivity analysis on key variables.

        Args:
            variable1 (str): First variable to analyze (e.g., 'discount_rate', 'revenue_growth')
            values1 (list): List of values for the first variable
            variable2 (str, optional): Second variable to analyze
            values2 (list, optional): List of values for the second variable

        Returns:
            pd.DataFrame: Sensitivity analysis results
        """
        try:
            # Store original values
            original_growth = self.growth_assumptions.copy()
            original_valuation = self.valuation_parameters.copy()

            # Check if we're doing a one-dimensional or two-dimensional analysis
            if variable2 is None or values2 is None:
                # One-dimensional analysis
                results = []
                for val1 in values1:
                    # Set the variable value
                    if variable1 in self.growth_assumptions:
                        self.growth_assumptions[variable1] = val1
                    elif variable1 in self.valuation_parameters:
                        self.valuation_parameters[variable1] = val1
                    else:
                        logger.warning(f"Variable {variable1} not recognized.")
                        continue

                    # Perform valuation
                    valuation = self.perform_dcf_valuation()
                    if valuation:
                        results.append(
                            {
                                "variable_value": val1,
                                "per_share_value": valuation.get(
                                    "per_share_value", np.nan
                                ),
                                "upside_potential": valuation.get(
                                    "upside_potential", np.nan
                                ),
                            }
                        )

                # Create DataFrame from results
                df = pd.DataFrame(results)
                df.set_index("variable_value", inplace=True)

            else:
                # Two-dimensional analysis
                results = {}
                for val1 in values1:
                    results[val1] = {}
                    for val2 in values2:
                        # Set the variables
                        if variable1 in self.growth_assumptions:
                            self.growth_assumptions[variable1] = val1
                        elif variable1 in self.valuation_parameters:
                            self.valuation_parameters[variable1] = val1

                        if variable2 in self.growth_assumptions:
                            self.growth_assumptions[variable2] = val2
                        elif variable2 in self.valuation_parameters:
                            self.valuation_parameters[variable2] = val2

                        # Perform valuation
                        valuation = self.perform_dcf_valuation()
                        if valuation:
                            results[val1][val2] = valuation.get(
                                "per_share_value", np.nan
                            )

                # Create DataFrame from results
                df = pd.DataFrame(results)

            # Restore original values
            self.growth_assumptions = original_growth
            self.valuation_parameters = original_valuation

            return df

        except Exception as e:
            logger.error(f"Error performing sensitivity analysis: {str(e)}")
            # Restore original values
            self.growth_assumptions = original_growth
            self.valuation_parameters = original_valuation
            return pd.DataFrame()
