"""
Templates module for the ValuScope financial analysis toolkit.
Contains HTML templates for report generation.
"""


def get_report_template():
    """
    Returns the HTML template for the financial analysis report.

    This template uses Python's string formatting to insert dynamic content.
    """
    return """
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
                .acronym-explanation {{ margin-bottom: 10px; font-style: italic; color: #666; }}
                .assumptions-box {{ margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{company_name} Financial Analysis Report</h1>
                <p>Generated on {generation_date}</p>
            </div>
            
            <div class="section">
                <h2>Company Overview</h2>
                <p><strong>Ticker:</strong> {ticker}</p>
                <p><strong>Industry:</strong> {industry}</p>
                <p><strong>Sector:</strong> {sector}</p>
                <p><strong>Market Cap:</strong> ${market_cap:,}</p>
                <p><strong>Current Price:</strong> {current_price}</p>
            </div>
            
            <div class="section">
                <h2>Key Financial Metrics</h2>
                <div class="metrics">
                    <div class="metric-box">
                        <h3>Revenue</h3>
                        <p>{revenue}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Net Income</h3>
                        <p>{net_income}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Target Price</h3>
                        <p>{target_price}</p>
                    </div>
                    <div class="metric-box">
                        <h3>Upside Potential</h3>
                        <p>{upside}</p>
                    </div>
                </div>
                {assumptions_html}
            </div>
            
            <div class="section">
                <h3 class="mt-4">Financial Ratios</h3>
                <div class="acronym-explanation">
                    <strong>ROE</strong> - Return on Equity: Net Income / Total Equity<br>
                    <strong>ROA</strong> - Return on Assets: Net Income / Total Assets<br>
                    <strong>Profit Margin</strong> - Net Income / Total Revenue<br>
                    <strong>Debt to Equity</strong> - Total Debt / Total Equity
                </div>
                {ratios_html}
            </div>
            
            <div class="section">
                <h2>Financial Visualizations</h2>
                {financial_trends_html}
                {stock_performance_html}
                {sensitivity_heatmap_html}
            </div>
            
            <div class="section">
                <h2>Valuation Summary</h2>
                <div class="recommendation">
                    Investment Recommendation: {recommendation}
                </div>
                <p><strong>Target Price:</strong> {target_price}</p>
                <p><strong>Current Price:</strong> {current_price}</p>
                <p><strong>Upside Potential:</strong> {upside}</p>
            </div>
            
            <div class="section">
                <h2>Disclaimer</h2>
                <p>This report is for educational and research purposes only. It should not be considered as financial advice. Always do your own research before making investment decisions.</p>
            </div>
        </body>
        </html>
        """


def get_assumptions_html_template():
    """
    Returns the HTML template for the DCF valuation assumptions section.

    This is designed to be shown in the Key Financial Metrics section
    to explain how the target price was derived from the DCF model.
    """
    return """
    <div class="assumptions-box">
        <p>
            <strong>DCF Valuation Assumptions:</strong><br>
            Revenue Growth: {revenue_growth} | Terminal Growth: {terminal_growth} | Discount Rate (WACC): {discount_rate} | Risk-Free Rate: {risk_free_rate}
        </p>
    </div>
    """
