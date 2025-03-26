# ValuScope: Financial Analysis Toolkit

A Python-based financial analysis toolkit that fetches financial data and provides analysis tools for stock valuation and financial ratio calculation.

## Features

- **Data Fetching**: Retrieves financial data from Yahoo Finance API including:

  - Company information
  - Balance sheets
  - Income statements
  - Cash flow statements
  - Historical stock prices

- **Financial Analysis**:

  - Calculation of key financial ratios (ROE, ROA, Profit Margin, Debt-to-Equity)
  - Visualization of financial trends
  - Stock price performance comparison with other companies

- **Valuation Models**:

  - Discounted Cash Flow (DCF) valuation
  - Sensitivity analysis for key valuation parameters
  - Terminal value calculation using multiple methods
  - Dynamic discount rate calculation using current market data (default)

- **End-to-End Analysis**:
  - Automated pipeline combining all analysis steps
  - HTML report generation with visualizations
  - Command-line interface with customizable parameters
  - Organized output with separate folders for data, visualizations, and logs

## Dynamic Discount Rate Calculation

One of the key features of ValuScope is the ability to calculate the discount rate (WACC) dynamically using current market data. This is now the default behavior, with static values used as fallbacks when necessary:

- Uses current 10-year Treasury yield as the risk-free rate
- Retrieves company beta from Yahoo Finance
- Applies standard equity risk premium (5.5%)
- Calculates cost of equity using CAPM
- Determines the company's debt structure and calculates after-tax cost of debt
- Computes WACC based on the company's current capital structure
- Uses intelligent fallback mechanisms when data is unavailable

This feature provides more accurate valuations by adapting to current market conditions rather than relying on fixed assumptions.

## Project Structure

The project follows a clean, modular structure:

```
valuscope/                  # Main package directory
├── __init__.py             # Package initialization
├── __main__.py             # Entry point for running as a module
├── main.py                 # Main application logic and CLI
└── core/                   # Core functionality modules
    ├── __init__.py         # Core subpackage initialization
    ├── data_fetcher.py     # Financial data retrieval (Yahoo Finance)
    ├── analysis.py         # Financial ratio calculation and visualization
    ├── valuation.py        # DCF and other valuation models
    └── tests.py            # Unit tests for the core modules
```

## Installation

### Option 1: Installing from Source

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the package in development mode:
   ```
   pip install -e .
   ```

### Option 2: Normal Installation

```
pip install valuscope
```

## Usage

### Command Line Interface

After installation, you can use the package from the command line:

```bash
# Using the console script
valuscope AAPL --output output_folder

# Or as a Python module
python -m valuscope AAPL --output output_folder
```

This will:

1. Fetch all financial data for Apple
2. Calculate financial ratios
3. Generate visualizations
4. Perform DCF valuation
5. Create a comprehensive HTML report

Additional command-line options:

```bash
# Compare with different companies
python -m valuscope AAPL --compare MSFT GOOGL AMZN

# Customize DCF valuation parameters
python -m valuscope AAPL --growth-rate 0.08 --terminal-growth 0.03 --discount-rate 0.095

# Use static discount rate instead of dynamic calculation
python -m valuscope AAPL --static-discount

# Specify output directory
python -m valuscope AAPL --output my_analysis
```

### Using as a Python Package

#### Data Fetcher

```python
from valuscope.core.data_fetcher import YahooFinanceFetcher

# Initialize the data fetcher with a ticker symbol
fetcher = YahooFinanceFetcher("AAPL")

# Get company information
company_info = fetcher.get_company_info()
print(f"Company: {company_info.get('longName')}")
print(f"Industry: {company_info.get('industry')}")
print(f"Market Cap: ${company_info.get('marketCap'):,}")

# Fetch financial statements (annual)
balance_sheet = fetcher.fetch_balance_sheet(quarterly=False)
income_stmt = fetcher.fetch_income_statement(quarterly=False)
cash_flow = fetcher.fetch_cash_flow(quarterly=False)

# Fetch historical prices
hist_prices = fetcher.get_historical_prices(period='1y')

# Get all available data in one call
fetcher.fetch_all_financial_data(quarterly=False)
data = fetcher.get_stored_data()
```

#### Financial Analysis

```python
from valuscope.core.analysis import FinancialAnalyzer

# Initialize analyzer with a ticker
analyzer = FinancialAnalyzer("AAPL")

# Fetch all financial data
analyzer.fetch_data()

# Calculate financial ratios
ratios = analyzer.calculate_financial_ratios()
print(ratios)

# Plot financial trends (Revenue, Gross Profit, Net Income)
analyzer.plot_financial_trends(save_path="financial_trends.png")

# Compare stock performance with other companies
analyzer.plot_stock_performance(
    compare_tickers=["MSFT", "GOOGL"],
    period="1y",
    save_path="stock_performance.png"
)
```

#### Discounted Cash Flow Valuation

```python
from valuscope.core.valuation import DCFValuationModel

# Initialize DCF model with a ticker
model = DCFValuationModel("AAPL")

# Fetch financial data
model.fetch_data()

# Customize assumptions (optional)
model.set_growth_assumptions(
    revenue_growth=0.08,  # 8% annual growth
    terminal_growth=0.03,  # 3% terminal growth
)

model.set_valuation_parameters(
    discount_rate=0.095,  # 9.5% discount rate
    projection_years=5,
    terminal_multiple=14
)

# Perform DCF valuation
results = model.perform_dcf_valuation()

# Or use static discount rate instead of dynamic calculation
results_static = model.perform_dcf_valuation(use_current_discount_rate=False)

# Display results
model.display_valuation_results(results)

# Perform sensitivity analysis on the dynamic discount rate (varying by ±20%)
# and terminal growth rate
sensitivity = model.perform_sensitivity_analysis(
    'discount_rate', [0.8, 0.9, 1.0, 1.1, 1.2],  # Relative multipliers to base rate
    'terminal_growth', [0.02, 0.025, 0.03, 0.035, 0.04]  # Absolute values
)
print(sensitivity)
```

## Generated Reports and Visualizations

The end-to-end analysis generates the following outputs in an organized folder structure:

1. **Main Output Folder**:

   - HTML financial analysis report
   - Valuation summary text file

2. **Data Subfolder**:

   - CSV files for balance sheet, income statement, cash flow, and historical prices
   - Financial ratios CSV
   - Sensitivity analysis CSV

3. **Visualizations Subfolder**:

   - Financial trends chart
   - Stock performance comparison chart
   - Sensitivity analysis heatmap

4. **Logs Subfolder**:
   - Analysis log file with detailed execution information

## Sample Report

Below is an example of what the generated financial analysis report looks like:

<details>
<summary>Click to see a sample report</summary>

```html
<!DOCTYPE html>
<html>
  <head>
    <title>AAPL Financial Analysis Report</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 20px;
      }
      .header {
        background-color: #4caf50;
        color: white;
        padding: 20px;
        text-align: center;
      }
      .section {
        margin-top: 20px;
        margin-bottom: 20px;
        padding: 15px;
        border: 1px solid #ddd;
      }
      .metrics {
        display: flex;
        flex-wrap: wrap;
      }
      .metric-box {
        background-color: #f9f9f9;
        margin: 10px;
        padding: 15px;
        width: 200px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
      }
      table {
        border-collapse: collapse;
        width: 100%;
      }
      th,
      td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #ddd;
      }
      th {
        background-color: #f2f2f2;
      }
      .image-container {
        margin: 20px 0;
      }
      .recommendation {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 20px;
      }
      .acronym-explanation {
        margin-bottom: 10px;
        font-style: italic;
        color: #666;
      }
      .assumptions-box {
        margin-top: 10px;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        font-size: 0.9em;
      }
    </style>
  </head>
  <body>
    <div class="header">
      <h1>Apple Inc. Financial Analysis Report</h1>
      <p>Generated on 2025-03-25 19:58:24</p>
    </div>

    <div class="section">
      <h2>Company Overview</h2>
      <p><strong>Ticker:</strong> AAPL</p>
      <p><strong>Industry:</strong> Consumer Electronics</p>
      <p><strong>Sector:</strong> Technology</p>
      <p><strong>Market Cap:</strong> $3,361,194,901,504</p>
      <p><strong>Current Price:</strong> $223.75</p>
    </div>

    <div class="section">
      <h2>Key Financial Metrics</h2>
      <div class="metrics">
        <div class="metric-box">
          <h3>Revenue</h3>
          <p>$391,035,000,000.00</p>
        </div>
        <div class="metric-box">
          <h3>Net Income</h3>
          <p>$93,736,000,000.00</p>
        </div>
        <div class="metric-box">
          <h3>Target Price</h3>
          <p>$116.51</p>
        </div>
        <div class="metric-box">
          <h3>Upside Potential</h3>
          <p>-47.93%</p>
        </div>
      </div>
      <div class="assumptions-box">
        <p>
          <strong>DCF Valuation Assumptions:</strong><br />
          Revenue Growth: 5.00% | Terminal Growth: 2.50% | Discount Rate (WACC):
          10.61% | Risk-Free Rate: 4.31%
        </p>
      </div>
    </div>

    <!-- Additional sections omitted for brevity -->
  </body>
</html>
```

### Report Screenshots

The report includes comprehensive visualizations like sensitivity analysis heatmaps showing how the target price changes with different discount rates and terminal growth rates:

![Sensitivity Analysis](https://github.com/Vladimir-Hugec/ValuScope/raw/docs/images/sample_sensitivity_heatmap.png)

</details>

## Development

### Running Tests

```bash
# Run the built-in tests
python -m unittest valuscope.core.tests
```

### Building the Package

```bash
# Install build tools
pip install build

# Build the package
python -m build
```

## Contributing

This project is not accepting contributions at this time.

## Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research before making investment decisions.
