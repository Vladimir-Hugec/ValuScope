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

# Get logger for this module
logger = logging.getLogger(__name__)


def run_data_fetcher(ticker, output_dir):
    """Run the data fetcher component"""
    logger.info(f"Running data fetcher for {ticker}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create csv subfolder
    csv_dir = os.path.join(output_dir, "data")
    os.makedirs(csv_dir, exist_ok=True)

    # Initialize fetcher and get data
    fetcher = YahooFinanceFetcher(ticker)

    # Get company info
    company_info = fetcher.get_company_info()
    if company_info:
        logger.info(f"Company: {company_info.get('longName', ticker)}")
        logger.info(f"Industry: {company_info.get('industry', 'N/A')}")
        logger.info(f"Market Cap: ${company_info.get('marketCap', 'N/A'):,}")

    # Fetch financial statements
    logger.info("Fetching financial statements...")
    fetcher.fetch_all_financial_data(quarterly=False)

    # Fetch historical prices
    logger.info("Fetching historical stock prices...")
    fetcher.get_historical_prices(period="5y")

    # Save data to CSV files in the output directory
    logger.info(f"Saving data to {csv_dir}...")
    saved_files = fetcher.save_data_to_csv(directory=csv_dir)

    logger.info(f"Data fetcher completed. Saved {len(saved_files)} files.")
    return fetcher.get_stored_data()


def run_financial_analysis(ticker, output_dir, comparison_tickers=None):
    """Run the financial analysis component"""
    logger.info(f"Running financial analysis for {ticker}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create csv and visualization subfolders
    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = FinancialAnalyzer(ticker)

    # Fetch data
    logger.info("Fetching data for analysis...")
    analyzer.fetch_data()

    # Calculate financial ratios
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

    # Plot financial trends
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

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create csv and visualization subfolders
    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    # Initialize DCF model
    model = DCFValuationModel(ticker)

    # Fetch data
    logger.info("Fetching data for DCF valuation...")
    model.fetch_data()

    # Set custom assumptions if provided
    if custom_assumptions:
        if "growth" in custom_assumptions:
            model.set_growth_assumptions(**custom_assumptions["growth"])
        if "valuation" in custom_assumptions:
            model.set_valuation_parameters(**custom_assumptions["valuation"])

    # Perform DCF valuation
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
        # Use relative adjustments to the dynamic discount rate (±20%) rather than fixed values
        sensitivity = model.perform_sensitivity_analysis(
            "discount_rate",
            [0.8, 0.9, 1.0, 1.1, 1.2],  # Vary base rate by ±20%
            "terminal_growth",
            [0.02, 0.025, 0.03, 0.035, 0.04],
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

            # Plot only if we have valid numeric data
            if (
                not sensitivity_numeric.empty
                and not sensitivity_numeric.isnull().all().all()
            ):
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(
                    sensitivity_numeric, annot=True, cmap="YlGnBu", fmt=".2f"
                )
                plt.title(
                    f"{ticker} - Sensitivity Analysis: Discount Rate vs Terminal Growth"
                )
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
    else:
        logger.warning("DCF valuation could not be completed")

    logger.info("DCF valuation completed.")
    return results


def generate_report(ticker, data, analysis_data, valuation_results, output_dir):
    """Generate an HTML report summarizing all results"""
    logger.info(f"Generating final report for {ticker}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for subfolders
    csv_dir = os.path.join(output_dir, "data")
    viz_dir = os.path.join(output_dir, "visualizations")

    report_file = os.path.join(output_dir, f"{ticker}_financial_analysis_report.html")

    # Get company info
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

        has_financial_trends = os.path.exists(financial_trends_path)
        has_stock_performance = os.path.exists(stock_performance_path)
        has_sensitivity_heatmap = os.path.exists(sensitivity_heatmap_path)

        # Create ratio HTML if we have ratios
        ratios_html = (
            ratios.to_html()
            if ratios is not None and not ratios.empty
            else "<p>No financial ratio data available</p>"
        )

        # Create HTML report with relative paths to visualization files
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{ticker} Financial Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin-top: 20px; margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }}
                .metrics {{ display: flex; flex-wrap: wrap; }}
                .metric-box {{ background-color: #f9f9f9; margin: 10px; padding: 15px; width: 200px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .image-container {{ margin: 20px 0; }}
                .recommendation {{ font-size: 24px; font-weight: bold; text-align: center; margin: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{company_info.get('longName', ticker)} Financial Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Company Overview</h2>
                <p><strong>Ticker:</strong> {ticker}</p>
                <p><strong>Industry:</strong> {company_info.get('industry', 'N/A')}</p>
                <p><strong>Sector:</strong> {company_info.get('sector', 'N/A')}</p>
                <p><strong>Market Cap:</strong> ${company_info.get('marketCap', 'N/A'):,}</p>
                <p><strong>Current Price:</strong> {format_value(current_price)}</p>
            </div>
            
            <div class="section">
                <h2>Key Financial Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Revenue</h3>
                        <p>{format_value(revenue)}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Net Income</h3>
                        <p>{format_value(net_income)}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Target Price</h3>
                        <p>{format_value(target_price)}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Upside Potential</h3>
                        <p>{format_percentage(upside)}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Financial Ratios</h2>
                {ratios_html}
            </div>
            
            <div class="section">
                <h2>Financial Visualizations</h2>
                {'<div class="image-container"><h3>Financial Trends</h3><img src="visualizations/' + ticker + '_financial_trends.png" alt="Financial Trends" style="max-width:100%;"></div>' if has_financial_trends else '<p>Financial trends visualization not available</p>'}
                {'<div class="image-container"><h3>Stock Performance Comparison</h3><img src="visualizations/' + ticker + '_stock_performance.png" alt="Stock Performance" style="max-width:100%;"></div>' if has_stock_performance else '<p>Stock performance comparison visualization not available</p>'}
                {'<div class="image-container"><h3>Sensitivity Analysis</h3><img src="visualizations/' + ticker + '_sensitivity_heatmap.png" alt="Sensitivity Analysis" style="max-width:100%;"></div>' if has_sensitivity_heatmap else '<p>Sensitivity analysis visualization not available</p>'}
            </div>
            
            <div class="section">
                <h2>Valuation Summary</h2>
                <div class="recommendation">
                    Investment Recommendation: {recommendation}
                </div>
                <p><strong>Target Price:</strong> {format_value(target_price)}</p>
                <p><strong>Current Price:</strong> {format_value(current_price)}</p>
                <p><strong>Upside Potential:</strong> {format_percentage(upside)}</p>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p>This report is for educational and research purposes only. It should not be considered as financial advice. Always do your own research before making investment decisions.</p>
            </div>
        </body>
        </html>
        """

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
            ticker, data, analysis_data, valuation_results, output_dir
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

    # Run the main function
    sys.exit(main())
