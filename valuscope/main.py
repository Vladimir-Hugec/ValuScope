"""
Main module for the ValuScope financial analysis toolkit.
Provides the main command-line interface and orchestrates the analysis workflow.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from valuscope.core.data_fetcher import YahooFinanceFetcher
from valuscope.core.analysis import FinancialAnalyzer
from valuscope.core.valuation import DCFValuationModel
from valuscope.templates import get_report_template, get_assumptions_html_template, get_visualization_html_templates

logger = logging.getLogger(__name__)


def run_data_fetcher(ticker, output_dir):
    """Run the data fetcher component"""
    logger.info(f"Running data fetcher for {ticker}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create csv subfolder
    csv_dir = os.path.join(output_dir, "data")
    os.makedirs(csv_dir, exist_ok=True)

    fetcher = YahooFinanceFetcher(ticker)

    # Get company info
    company_info = fetcher.get_company_info()
    if company_info:
        logger.info(f"Company: {company_info.get('longName', ticker)}")
        logger.info(f"Industry: {company_info.get('industry', 'N/A')}")
        logger.info(f"Market Cap: ${company_info.get('marketCap', 'N/A'):,}")

    logger.info("Fetching financial statements...")
    fetcher.fetch_all_financial_data(quarterly=False)

    logger.info("Fetching historical stock prices...")
    fetcher.get_historical_prices(period="5y")

    logger.info(f"Saving data to {csv_dir}...")
    saved_files = fetcher.save_data_to_csv(directory=csv_dir)

    logger.info(f"Data fetcher completed. Saved {len(saved_files)} files.")
    return fetcher.get_stored_data()


def run_financial_analysis(ticker, output_dir, comparison_tickers=None):
    """Run the financial analysis component"""
    logger.info(f"Running financial analysis for {ticker}")

    os.makedirs(output_dir, exist_ok=True)

    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    analyzer = FinancialAnalyzer(ticker)

    logger.info("Fetching data for analysis...")
    analyzer.fetch_data()

    logger.info("Calculating financial ratios...")
    ratios = analyzer.calculate_financial_ratios()

    if not ratios.empty:
        logger.info("\nFinancial Ratios:\n" + str(ratios))
        # Save ratios to CSV
        ratios_file = os.path.join(csv_dir, f"{ticker}_financial_ratios.csv")
        ratios.to_csv(ratios_file)
        logger.info(f"Financial ratios saved to {ratios_file}")
    else:
        logger.warning("Could not calculate financial ratios")

    logger.info("Generating financial trends visualization...")
    financial_trends_file = os.path.join(viz_dir, f"{ticker}_financial_trends.png")
    analyzer.plot_financial_trends(save_path=financial_trends_file)

    # Plot stock performance comparison
    if comparison_tickers:
        logger.info(
            f"Comparing stock performance with {', '.join(comparison_tickers)}..."
        )
        performance_file = os.path.join(viz_dir, f"{ticker}_stock_performance.png")
        analyzer.plot_stock_performance(
            compare_tickers=comparison_tickers, period="1y", save_path=performance_file
        )

    logger.info("Financial analysis completed.")
    return analyzer.data


def run_dcf_valuation(
    ticker, output_dir, custom_assumptions=None, use_current_discount_rate=True
):
    """Run the DCF valuation component"""
    logger.info(f"Running DCF valuation for {ticker}")

    os.makedirs(output_dir, exist_ok=True)

    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    model = DCFValuationModel(ticker)

    logger.info("Fetching data for DCF valuation...")
    model.fetch_data()

    # Set custom assumptions if provided
    if custom_assumptions:
        if "growth" in custom_assumptions:
            model.set_growth_assumptions(**custom_assumptions["growth"])
        if "valuation" in custom_assumptions:
            model.set_valuation_parameters(**custom_assumptions["valuation"])

    logger.info("Performing DCF valuation...")
    if use_current_discount_rate:
        logger.info(
            "Using dynamic discount rate calculation based on current market data"
        )
        results = model.perform_dcf_valuation(use_current_discount_rate=True)
    else:
        logger.info("Using static discount rate instead of dynamic calculation")
        results = model.perform_dcf_valuation(use_current_discount_rate=False)

    if results:
        # Save valuation summary to text file
        summary_file = os.path.join(output_dir, f"{ticker}_valuation_summary.txt")
        original_stdout = sys.stdout
        with open(summary_file, "w") as f:
            sys.stdout = f
            model.display_valuation_results(results)
            sys.stdout = original_stdout
        logger.info(f"Valuation summary saved to {summary_file}")

        # Perform sensitivity analysis
        logger.info("Performing sensitivity analysis...")

        sensitivity = model.perform_sensitivity_analysis(
            "discount_rate",
            [
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
            ],  # Vary base rate by ±30% in 10% increments
            "terminal_growth",
            [
                0.02,
                0.025,
                0.03,
                0.035,
                0.04,
                0.045,
                0.05,
            ],  # Terminal growth from 2% to 5%
        )

        # Save sensitivity analysis to CSV
        sensitivity_file = os.path.join(csv_dir, f"{ticker}_sensitivity_analysis.csv")
        sensitivity.to_csv(sensitivity_file)
        logger.info(f"Sensitivity analysis saved to {sensitivity_file}")

        # Create a heatmap visualization of the sensitivity analysis
        try:
            import seaborn as sns

            # Make sure all values in the sensitivity DataFrame are numeric
            sensitivity_numeric = sensitivity.apply(pd.to_numeric, errors="coerce")

            # Create a copy to avoid modifying the original DataFrame
            sensitivity_display = sensitivity_numeric.copy()

            # Convert index and columns to formatted percentages for display if they're discount rates or growth rates
            if all(
                isinstance(idx, (int, float))
                or (isinstance(idx, str) and idx.replace(".", "", 1).isdigit())
                for idx in sensitivity_numeric.index
            ):
                # Format the index as percentages
                sensitivity_display.index = [
                    f"{float(idx):.2%}" for idx in sensitivity_numeric.index
                ]

            if all(
                isinstance(col, (int, float))
                or (isinstance(col, str) and col.replace(".", "", 1).isdigit())
                for col in sensitivity_numeric.columns
            ):
                # Format the columns as percentages
                sensitivity_display.columns = [
                    f"{float(col):.2%}" for col in sensitivity_numeric.columns
                ]

            # Plot only if we have valid numeric data
            if (
                not sensitivity_numeric.empty
                and not sensitivity_numeric.isnull().all().all()
            ):
                plt.figure(figsize=(10, 8))

                # Determine if values are stock prices or percentages
                is_stock_price = (
                    np.mean(sensitivity_numeric.values) > 1
                )  # likely stock prices, no penny stock users allowed

                # Format annotations
                if is_stock_price:
                    # Handle dollar formatting for the annotations
                    annot = np.vectorize(lambda x: f"${x:.2f}")(
                        sensitivity_numeric.values
                    )
                    fmt = ""  # Empty format as we've pre-formatted the annotations
                else:
                    # Handle percentage formatting for the annotations
                    annot = np.vectorize(lambda x: f"{x:.2%}")(
                        sensitivity_numeric.values
                    )
                    fmt = ""  # Empty format as we've pre-formatted the annotations

                # Create the heatmap with the numeric data but display formatting
                ax = sns.heatmap(
                    sensitivity_numeric, annot=annot, cmap="YlGnBu", fmt=fmt
                )

                # Set x and y tick labels using the percentage-formatted display DataFrame
                ax.set_xticklabels(sensitivity_display.columns)
                ax.set_yticklabels(sensitivity_display.index)

                plt.title(
                    f"{ticker} - Sensitivity Analysis: Discount Rate vs Terminal Growth"
                )

                plt.xlabel("Discount Rate (WACC)")
                plt.ylabel("Terminal Growth Rate")
                plt.tight_layout()

                sensitivity_plot = os.path.join(
                    viz_dir, f"{ticker}_sensitivity_heatmap.png"
                )
                plt.savefig(sensitivity_plot)
                logger.info(f"Sensitivity heatmap saved to {sensitivity_plot}")
            else:
                logger.warning(
                    "Could not create sensitivity heatmap due to non-numeric values"
                )
        except (ImportError, ValueError, TypeError) as e:
            logger.warning(f"Could not create sensitivity heatmap: {str(e)}")

        # Create the equilibrium plot showing Terminal Growth Rate vs Discount Rate pairs
        # that yield the current stock price
        try:
            logger.info("Generating equilibrium plot...")
            equilibrium_plot = os.path.join(viz_dir, f"{ticker}_equilibrium_plot.png")

            # Create the equilibrium plot with a resolution of 15 points for each axis
            # This gives a good balance between precision and calculation time
            _, eq_df, reg_params = model.plot_discount_growth_equilibrium(
                save_path=equilibrium_plot, resolution=15
            )

            if eq_df is not None and not eq_df.empty:
                # Save equilibrium points to CSV
                eq_points_file = os.path.join(
                    csv_dir, f"{ticker}_equilibrium_points.csv"
                )
                eq_df.to_csv(eq_points_file)
                logger.info(f"Equilibrium points saved to {eq_points_file}")

                logger.info(f"Equilibrium plot saved to {equilibrium_plot}")
                logger.info(
                    f"Regression equation: Terminal Growth = {reg_params['slope']:.4f} × Discount Rate + {reg_params['intercept']:.4f}"
                )
            else:
                logger.warning(
                    "Could not generate equilibrium plot: no equilibrium points found"
                )

        except (ImportError, ValueError, TypeError) as e:
            logger.warning(f"Could not create equilibrium plot: {str(e)}")
            
        # Create the revenue-terminal growth equilibrium plot showing combinations
        # that yield the current stock price while holding discount rate constant
        try:
            logger.info("Generating revenue-terminal growth equilibrium plot...")
            rev_growth_plot = os.path.join(viz_dir, f"{ticker}_rev_term_growth_equilibrium.png")

            # Create the equilibrium plot with a resolution of 15 points for each axis
            # This gives a good balance between precision and calculation time
            try:
                _, eq_df, reg_params = model.plot_revenue_terminal_growth_equilibrium(
                    save_path=rev_growth_plot, resolution=15
                )

                if eq_df is not None and not eq_df.empty:
                    # Save equilibrium points to CSV
                    eq_points_file = os.path.join(
                        csv_dir, f"{ticker}_rev_term_growth_points.csv"
                    )
                    eq_df.to_csv(eq_points_file)
                    logger.info(f"Revenue-terminal growth equilibrium points saved to {eq_points_file}")

                    logger.info(f"Revenue-terminal growth equilibrium plot saved to {rev_growth_plot}")
                    
                    # Log regression equation based on degree
                    if reg_params.get('degree') == 1:
                        logger.info(
                            f"Regression equation: Terminal Growth = {reg_params['slope']:.4f} × Revenue Growth + {reg_params['intercept']:.4f}"
                        )
                    else:
                        logger.info(
                            f"Polynomial regression of degree {reg_params.get('degree')} found for revenue-terminal growth relationship"
                        )
                else:
                    logger.warning(
                        "Could not generate revenue-terminal growth equilibrium plot: no equilibrium points found"
                    )
            except ImportError as e:
                logger.warning(f"ImportError creating revenue-terminal growth plot: {str(e)}. Make sure scikit-learn is installed.")
            except Exception as e:
                logger.warning(f"Unexpected error creating revenue-terminal growth plot: {str(e)}")

        except (ImportError, ValueError, TypeError) as e:
            logger.warning(f"Could not create revenue-terminal growth equilibrium plot: {str(e)}")
            
    else:
        logger.warning("DCF valuation could not be completed")

    logger.info("DCF valuation completed.")
    return results


def generate_report(
    ticker, data, analysis_data, valuation_results, output_dir, custom_assumptions=None
):
    """Generate an HTML report summarizing all results"""
    logger.info(f"Generating final report for {ticker}")

    os.makedirs(output_dir, exist_ok=True)

    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")

    report_file = os.path.join(output_dir, f"{ticker}_financial_analysis_report.html")

    company_info = data.get("company_info", {})

    # Try to load financial ratios from saved file if not in analysis_data
    ratios = None
    ratios_file = os.path.join(csv_dir, f"{ticker}_financial_ratios.csv")
    try:
        if os.path.exists(ratios_file):
            ratios = pd.read_csv(ratios_file, index_col=0)
        else:
            # Try to calculate ratios if we have the necessary data
            balance_sheet = data.get("balance_sheet", pd.DataFrame())
            income_stmt = data.get("income_statement", pd.DataFrame())
            if not balance_sheet.empty and not income_stmt.empty:
                analyzer = FinancialAnalyzer(ticker)
                analyzer.data = data
                ratios = analyzer.calculate_financial_ratios()
    except Exception as e:
        logger.warning(f"Could not load financial ratios: {str(e)}")

    # Get key financial metrics
    try:
        balance_sheet = data.get("balance_sheet", pd.DataFrame())
        income_stmt = data.get("income_statement", pd.DataFrame())
        cash_flow = data.get("cash_flow", pd.DataFrame())

        # Get key financial metrics from the most recent year
        if not income_stmt.empty:
            latest_year = income_stmt.columns[0]
            # Safely extract values
            revenue = (
                income_stmt.loc["Total Revenue", latest_year]
                if "Total Revenue" in income_stmt.index
                else "N/A"
            )
            net_income = (
                income_stmt.loc["Net Income", latest_year]
                if "Net Income" in income_stmt.index
                else "N/A"
            )
        else:
            revenue = "N/A"
            net_income = "N/A"

        # Get DCF valuation results
        if valuation_results:
            target_price = valuation_results.get("per_share_value", "N/A")
            current_price = valuation_results.get("current_price", "N/A")
            upside = valuation_results.get("upside_potential", "N/A")

            if upside != "N/A":
                if upside > 0.2:
                    recommendation = "Strong Buy"
                elif upside > 0.05:
                    recommendation = "Buy"
                elif upside > -0.05:
                    recommendation = "Hold"
                elif upside > -0.2:
                    recommendation = "Sell"
                else:
                    recommendation = "Strong Sell"
            else:
                recommendation = "N/A"
        else:
            target_price = "N/A"
            current_price = "N/A"
            upside = "N/A"
            recommendation = "N/A"

        # Format number values for display
        def format_value(value):
            if value == "N/A":
                return "N/A"
            try:
                return f"${value:,.2f}"
            except (ValueError, TypeError):
                return str(value)

        # Format percentage values
        def format_percentage(value):
            if value == "N/A":
                return "N/A"
            try:
                return f"{value:.2%}"
            except (ValueError, TypeError):
                return str(value)

        # Check if visualization files exist
        financial_trends_path = os.path.join(viz_dir, f"{ticker}_financial_trends.png")
        stock_performance_path = os.path.join(
            viz_dir, f"{ticker}_stock_performance.png"
        )
        sensitivity_heatmap_path = os.path.join(
            viz_dir, f"{ticker}_sensitivity_heatmap.png"
        )
        equilibrium_plot_path = os.path.join(viz_dir, f"{ticker}_equilibrium_plot.png")
        rev_term_growth_plot_path = os.path.join(viz_dir, f"{ticker}_rev_term_growth_equilibrium.png")

        has_financial_trends = os.path.exists(financial_trends_path)
        has_stock_performance = os.path.exists(stock_performance_path)
        has_sensitivity_heatmap = os.path.exists(sensitivity_heatmap_path)
        has_equilibrium_plot = os.path.exists(equilibrium_plot_path)
        has_rev_term_growth_plot = os.path.exists(rev_term_growth_plot_path)

        # Create ratio HTML if we have ratios
        ratios_html = (
            ratios.to_html()
            if ratios is not None and not ratios.empty
            else "<p>No financial ratio data available</p>"
        )

        # Get visualization HTML using the templates module
        visualization_html = get_visualization_html_templates(
            ticker,
            has_financial_trends,
            has_stock_performance,
            has_sensitivity_heatmap,
            has_equilibrium_plot,
            has_rev_term_growth_plot
        )
        
        financial_trends_html = visualization_html['financial_trends_html']
        stock_performance_html = visualization_html['stock_performance_html']
        sensitivity_heatmap_html = visualization_html['sensitivity_heatmap_html']
        equilibrium_plot_html = visualization_html['equilibrium_plot_html']
        rev_term_growth_plot_html = visualization_html['rev_term_growth_plot_html']

        template = get_report_template()

        # Prepare assumptions HTML if valuation results are available
        assumptions_html = ""
        if valuation_results:
            # Get the assumptions used for the DCF model
            revenue_growth = format_percentage(
                custom_assumptions.get("growth", {}).get("revenue_growth", 0.05)
                if custom_assumptions
                else 0.05
            )
            terminal_growth = format_percentage(
                custom_assumptions.get("growth", {}).get("terminal_growth", 0.025)
                if custom_assumptions
                else 0.025
            )
            discount_rate = format_percentage(
                valuation_results.get(
                    "discount_rate",
                    (
                        custom_assumptions.get("valuation", {}).get(
                            "discount_rate", 0.09
                        )
                        if custom_assumptions
                        else 0.09
                    ),
                )
            )

            # Get risk-free rate if available
            risk_free_rate = format_percentage(
                valuation_results.get("risk_free_rate", "N/A")
                if valuation_results.get("risk_free_rate") is not None
                else "N/A"
            )

            # Get the assumptions template and format it with values
            assumptions_template = get_assumptions_html_template()
            assumptions_html = assumptions_template.format(
                revenue_growth=revenue_growth,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                risk_free_rate=risk_free_rate,
            )

        # Format the template with data
        html_content = template.format(
            ticker=ticker,
            company_name=company_info.get("longName", ticker),
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            industry=company_info.get("industry", "N/A"),
            sector=company_info.get("sector", "N/A"),
            market_cap=company_info.get("marketCap", "N/A"),
            current_price=format_value(current_price),
            revenue=format_value(revenue),
            net_income=format_value(net_income),
            target_price=format_value(target_price),
            upside=format_percentage(upside),
            ratios_html=ratios_html,
            financial_trends_html=financial_trends_html,
            stock_performance_html=stock_performance_html,
            sensitivity_heatmap_html=sensitivity_heatmap_html,
            equilibrium_plot_html=equilibrium_plot_html,
            rev_term_growth_plot_html=rev_term_growth_plot_html,
            recommendation=recommendation,
            assumptions_html=assumptions_html,
        )

        # Write to file
        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"Financial analysis report saved to {report_file}")

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")

    return report_file


def main():
    """Main function to run the end-to-end financial analysis"""
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Financial Analysis Toolkit")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--output", type=str, default="output", help="Output directory for results"
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=["MSFT", "GOOGL"],
        help="Tickers to compare with (default: MSFT GOOGL)",
    )
    parser.add_argument(
        "--growth-rate",
        type=float,
        default=0.05,
        help="Revenue growth rate assumption (default: 0.05)",
    )
    parser.add_argument(
        "--terminal-growth",
        type=float,
        default=0.025,
        help="Terminal growth rate assumption (default: 0.025)",
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.09,
        help="Discount rate assumption (default: 0.09)",
    )
    parser.add_argument(
        "--static-discount",
        action="store_true",
        help="Use static discount rate instead of calculating dynamically from market data",
    )

    # Parse arguments
    args = parser.parse_args()

    ticker = args.ticker.upper()
    output_dir = os.path.join(args.output, ticker)

    # Create main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Set up logging to file in the logs directory
    log_file = os.path.join(logs_dir, f"{ticker}_financial_analysis.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    logger.info(f"Starting end-to-end financial analysis for {ticker}")
    logger.info(f"Results will be saved to {output_dir}")
    logger.info(f"Logs will be saved to {log_file}")

    # Custom assumptions for the DCF model
    custom_assumptions = {
        "growth": {
            "revenue_growth": args.growth_rate,
            "terminal_growth": args.terminal_growth,
        },
        "valuation": {
            "discount_rate": args.discount_rate,
        },
    }

    try:
        # Step 1: Run data fetcher
        data = run_data_fetcher(ticker, output_dir)

        # Step 2: Run financial analysis
        analysis_data = run_financial_analysis(ticker, output_dir, args.compare)

        # Step 3: Run DCF valuation
        valuation_results = run_dcf_valuation(
            ticker, output_dir, custom_assumptions, not args.static_discount
        )

        # Step 4: Generate final report
        report_file = generate_report(
            ticker,
            data,
            analysis_data,
            valuation_results,
            output_dir,
            custom_assumptions,
        )

        logger.info(f"End-to-end financial analysis completed for {ticker}")
        logger.info(f"Final report available at: {report_file}")

        return 0  # Success exit code

    except Exception as e:
        logger.error(f"Error in end-to-end analysis: {str(e)}")
        return 1  # Error exit code


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    sys.exit(main())
