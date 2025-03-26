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
                
                /* Tab styling */
                .tabs {{ display: flex; margin: 20px 0 0 0; padding: 0; border-bottom: 1px solid #ccc; }}
                .tab {{ padding: 10px 15px; cursor: pointer; margin-right: 2px; background-color: #f1f1f1; }}
                .tab.active {{ background-color: #4CAF50; color: white; }}
                .tab-content {{ display: none; padding: 15px 0; }}
                .tab-content.active {{ display: block; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {{
                        tabcontent[i].style.display = "none";
                    }}
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {{
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
                
                // Default to open the first tab when the page loads
                document.addEventListener('DOMContentLoaded', function() {{
                    document.querySelector('.tab').click();
                }});
            </script>
        </head>
        <body>
            <div class="header">
                <h1>{company_name} Financial Analysis Report</h1>
                <p>Generated on {generation_date}</p>
            </div>
            
            <!-- Tab navigation -->
            <div class="tabs">
                <button class="tab active" onclick="openTab(event, 'summary')">Summary</button>
                <button class="tab" onclick="openTab(event, 'dcf-valuation')">DCF Valuation</button>
            </div>
            
            <!-- Summary Tab -->
            <div id="summary" class="tab-content active">
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
                            <h3>Fair Value (based on DCF valuation)</h3>
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
                    <h2>Valuation Summary</h2>
                    <div class="recommendation">
                        Investment Recommendation: {recommendation}
                    </div>
                    <p><strong>Fair Value (based on DCF valuation):</strong> {target_price}</p>
                    <p><strong>Current Price:</strong> {current_price}</p>
                    <p><strong>Upside Potential:</strong> {upside}</p>
                </div>
                
                <div class="section">
                    <h2>Financial Trends</h2>
                    {financial_trends_html}
                    {stock_performance_html}
                </div>
                
                <div class="section">
                    <h2>Financial Ratios</h2>
                    <div class="acronym-explanation">
                        <strong>ROE</strong> - Return on Equity: Net Income / Total Equity<br>
                        <strong>ROA</strong> - Return on Assets: Net Income / Total Assets<br>
                        <strong>Profit Margin</strong> - Net Income / Total Revenue<br>
                        <strong>Debt to Equity</strong> - Total Debt / Total Equity
                    </div>
                    {ratios_html}
                </div>
            </div>
            
            <!-- DCF Valuation Tab -->
            <div id="dcf-valuation" class="tab-content">
                <div class="section">
                    <h2>Sensitivity Analysis</h2>
                    <p>This heatmap shows how changes in Discount Rate (WACC) and Terminal Growth Rate affect the calculated share price.</p>
                    {sensitivity_heatmap_html}
                </div>
                
                <div class="section">
                    <h2>Equilibrium Analysis</h2>
                    <p>These visualizations show combinations of growth rates that yield the current market price.</p>
                    {equilibrium_plot_html}
                    {rev_term_growth_plot_html}
                </div>
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


def get_visualization_html_templates(ticker, has_financial_trends, has_stock_performance, 
                                    has_sensitivity_heatmap, has_equilibrium_plot, has_rev_term_growth_plot=False):
    """
    Returns HTML for visualization elements.
    
    Args:
        ticker (str): The stock ticker symbol
        has_financial_trends (bool): Whether financial trends visualization exists
        has_stock_performance (bool): Whether stock performance visualization exists
        has_sensitivity_heatmap (bool): Whether sensitivity heatmap visualization exists
        has_equilibrium_plot (bool): Whether equilibrium plot visualization exists
        has_rev_term_growth_plot (bool): Whether revenue-terminal growth equilibrium plot exists
    
    Returns:
        dict: Dictionary containing HTML for each visualization type
    """
    financial_trends_html = (
        '<div class="image-container"><h3>Financial Trends</h3>'
        f'<img src="visualizations/{ticker}_financial_trends.png" alt="Financial Trends" style="max-width:100%;">'
        '</div>'
        if has_financial_trends
        else "<p>Financial trends visualization not available</p>"
    )
    
    stock_performance_html = (
        '<div class="image-container"><h3>Stock Performance Comparison</h3>'
        f'<img src="visualizations/{ticker}_stock_performance.png" alt="Stock Performance" style="max-width:100%;">'
        '</div>'
        if has_stock_performance
        else "<p>Stock performance comparison visualization not available</p>"
    )
    
    sensitivity_heatmap_html = (
        '<div class="image-container"><h3>Sensitivity Analysis</h3>'
        f'<img src="visualizations/{ticker}_sensitivity_heatmap.png" alt="Sensitivity Analysis" style="max-width:100%;">'
        '</div>'
        if has_sensitivity_heatmap
        else "<p>Sensitivity analysis visualization not available</p>"
    )
    
    equilibrium_plot_html = (
        '<div class="image-container"><h3>Discount Rate vs Terminal Growth Equilibrium</h3>'
        '<p>This plot shows combinations of Discount Rate and Terminal Growth Rate that yield the current stock price '
        'while holding revenue growth constant, with a regression line showing the relationship and the current discount rate marked.</p>'
        f'<img src="visualizations/{ticker}_equilibrium_plot.png" alt="Equilibrium Analysis" style="max-width:100%;">'
        '</div>'
        if has_equilibrium_plot
        else "<p>Discount rate vs terminal growth equilibrium visualization not available</p>"
    )
    
    rev_term_growth_plot_html = (
        '<div class="image-container"><h3>Revenue Growth vs Terminal Growth Equilibrium</h3>'
        '<p>This plot shows combinations of Revenue Growth Rate and Terminal Growth Rate that yield the current stock price '
        'while holding the discount rate (WACC) constant, with a regression curve showing the relationship.</p>'
        f'<img src="visualizations/{ticker}_rev_term_growth_equilibrium.png" alt="Revenue-Terminal Growth Equilibrium" style="max-width:100%;">'
        '</div>'
        if has_rev_term_growth_plot
        else "<p>Revenue growth vs terminal growth equilibrium visualization not available</p>"
    )
    
    return {
        'financial_trends_html': financial_trends_html,
        'stock_performance_html': stock_performance_html,
        'sensitivity_heatmap_html': sensitivity_heatmap_html,
        'equilibrium_plot_html': equilibrium_plot_html,
        'rev_term_growth_plot_html': rev_term_growth_plot_html
    }
