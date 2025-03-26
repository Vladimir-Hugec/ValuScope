"""
Test script to generate the revenue-terminal growth equilibrium plot.
This script demonstrates the exhaustive grid search for combinations of
revenue growth and terminal growth rates that yield the current price.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create test plot output directory with absolute path
output_dir = os.path.join(os.getcwd(), "test_output")
print(f"Writing to output directory: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Set WACC (discount rate) - needed for the y-axis upper limit
wacc = 0.09  # 9% discount rate

# Create a simulated DCF model that calculates model price based on growth rates
# This is a simplified version of the real DCF model in the valuation.py module
def simplified_dcf_model(revenue_growth, terminal_growth):
    """
    Simulate a DCF model with revenue growth and terminal growth as inputs.
    
    This function mimics the behavior of a real DCF model:
    - Higher revenue growth increases the model price
    - Terminal growth has increasing impact as it approaches WACC
    - There's a non-linear relationship between revenue and terminal growth
    """
    # Parameters affecting the model price
    base_price = 100.0
    
    # Higher growth rates increase the model price
    revenue_impact = revenue_growth * 500.0
    
    # Terminal growth has high impact, especially when closer to discount rate
    # This is the key relationship - as terminal growth approaches WACC, 
    # the terminal value approaches infinity (division by near-zero)
    # Add safety check to avoid division by very small numbers
    growth_diff = max(wacc - terminal_growth, 0.001)  # Ensure at least 0.1% difference
    terminal_impact = terminal_growth / growth_diff * 200.0
    
    # Cap terminal impact to avoid unrealistic values
    terminal_impact = min(terminal_impact, 2000.0)
    
    # Add non-linear interactions between the variables
    interaction = revenue_growth * terminal_growth * 300.0
    
    # Add some randomization to simulate real-world complexity
    np.random.seed(42 + int(revenue_growth * 1000) + int(terminal_growth * 1000))  # Deterministic but varied
    noise = np.random.normal(0, 2.0)
    
    # Calculate final model price
    model_price = base_price + revenue_impact + terminal_impact + interaction + noise
    
    return model_price

# Define current market price to match against
current_price = 190.0
print(f"Current market price: ${current_price:.2f}")

# Define the ranges for our grid search
num_points_rev = 60  # Number of revenue growth points to test
num_points_term = 40  # Number of terminal growth points to test

# Define the grid search ranges
revenue_growth_rates = np.linspace(0.01, 0.20, num_points_rev)  # 1% to 20%
max_terminal_growth = wacc * 0.99  # Use 99% of WACC with no arbitrary cap
terminal_growth_rates = np.linspace(0.0, max_terminal_growth, num_points_term)

print(f"Searching grid of {num_points_rev}x{num_points_term} points")
print(f"Revenue growth range: {revenue_growth_rates[0]:.1%} to {revenue_growth_rates[-1]:.1%}")
print(f"Terminal growth range: {terminal_growth_rates[0]:.1%} to {terminal_growth_rates[-1]:.1%}")
print(f"Maximum terminal growth: {max_terminal_growth:.2%} (99% of WACC: {wacc:.2%})")

# Use a tolerance to find points close to the current price
tolerance = 0.05  # 5% tolerance
print(f"Tolerance for matching current price: {tolerance:.1%}")

# Create a matrix to store all model prices for analysis
all_results = np.full((len(revenue_growth_rates), len(terminal_growth_rates)), np.nan)

# Perform exhaustive grid search
equilibrium_points = []
np.random.seed(42)  # For reproducibility

print("Starting grid search for equilibrium points...")
for i, rev_growth in enumerate(revenue_growth_rates):
    for j, term_growth in enumerate(terminal_growth_rates):
        # Skip invalid combinations (terminal > WACC)
        if term_growth >= wacc:
            continue
            
        try:
            # Calculate model price using our simplified DCF model
            model_price = simplified_dcf_model(rev_growth, term_growth)
            
            # Store the result regardless of match
            all_results[i, j] = model_price
            
            # Calculate percentage difference from current price
            price_diff = abs(model_price - current_price) / current_price
            
            # If within tolerance, add to equilibrium points
            if price_diff <= tolerance:
                equilibrium_points.append({
                    "revenue_growth": rev_growth,
                    "terminal_growth": term_growth,
                    "model_price": model_price,
                    "price_diff": price_diff
                })
                print(f"Found equilibrium point: rev={rev_growth:.2%}, term={term_growth:.2%}, price=${model_price:.2f} (diff: {price_diff:.2%})")
        except Exception as e:
            print(f"Error calculating for rev={rev_growth:.2%}, term={term_growth:.2%}: {str(e)}")
            continue

# Analyze all results to understand why we might be missing points at lower revenue growth
min_price = np.nanmin(all_results)
max_price = np.nanmax(all_results)
print(f"\nAll model prices range: ${min_price:.2f} to ${max_price:.2f}")

# Map the landscape of prices across revenue growth values
for i, rev_growth in enumerate(revenue_growth_rates):
    if i % 10 == 0:  # Sample every 10th value to avoid excessive output
        row = all_results[i, :]
        if not np.all(np.isnan(row)):
            min_row = np.nanmin(row)
            max_row = np.nanmax(row)
            
            # Print the min/max at this revenue growth to understand the pattern
            print(f"Revenue growth {rev_growth:.2%}: price range ${min_row:.2f} to ${max_row:.2f}")
            
            # Check if it can hit the target price
            within_range = (min_row <= current_price * (1 + tolerance)) and (max_row >= current_price * (1 - tolerance))
            if not within_range:
                print(f"  Cannot reach target price ${current_price:.2f} ± {tolerance:.1%}")

# Create DataFrame from equilibrium points
df = pd.DataFrame(equilibrium_points)

# Check if we found any equilibrium points
if df.empty:
    print("No equilibrium points found within tolerance! Try increasing the tolerance.")
    exit(1)

# Sort by price difference for better visualization
df = df.sort_values("price_diff")

# Print summary statistics
print(f"\nFound {len(df)} equilibrium points within {tolerance:.1%} tolerance")
print(f"Minimum revenue growth: {df['revenue_growth'].min():.2%}")
print(f"Maximum revenue growth: {df['revenue_growth'].max():.2%}")
print(f"Minimum terminal growth: {df['terminal_growth'].min():.2%}")
print(f"Maximum terminal growth: {df['terminal_growth'].max():.2%}")

# Save the DataFrame to CSV
csv_path = os.path.join(output_dir, "test_equilibrium_points.csv")
df.to_csv(csv_path)
print(f"Saved test data to {csv_path}")

# Create a plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the points
scatter = ax.scatter(
    df['revenue_growth'], 
    df['terminal_growth'], 
    c=df['model_price'], 
    cmap='viridis', 
    alpha=0.7, 
    s=50,
    label="Equal to Current Price"
)

# Add a colorbar to show the model price differences
cbar = plt.colorbar(scatter)
cbar.set_label('Model Price ($)')

# Fit a polynomial regression curve to better capture the relationship
# Try different polynomial degrees to find the best fit
best_r2 = -1
best_degree = 1
best_model = None
coeffs = None

# Try polynomial regression with different degrees
max_degree = min(5, max(1, len(df) // 5))  # Avoid overfitting with high degrees

for degree in range(1, max_degree + 1):
    # Create and fit polynomial model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    
    X = df[['revenue_growth']].values
    y = df['terminal_growth'].values
    
    model.fit(X, y)
    
    # Calculate R² score
    from sklearn.metrics import r2_score
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"Polynomial degree {degree}: R² = {r2:.4f}")
    
    # Keep track of the best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_degree = degree

# Use the best model to generate a smooth curve
curve_x = np.linspace(min(df['revenue_growth']), max(df['revenue_growth']), 100)
curve_x_2d = curve_x.reshape(-1, 1)
curve_y = best_model.predict(curve_x_2d)

# Extract coefficients for equation display
linear_model = best_model.named_steps['linearregression']
coeffs = linear_model.coef_
intercept = linear_model.intercept_

# Create label and equation text based on degree
if best_degree == 1:
    label_text = f"Linear Fit (R² = {best_r2:.4f})"
    equation_text = f"Terminal Growth = {coeffs[1]:.4f} × Revenue Growth + {intercept:.4f}"
elif best_degree == 2:
    label_text = f"Quadratic Fit (R² = {best_r2:.4f})"
    equation_text = f"Terminal Growth = {coeffs[3]:.4f}x² + {coeffs[1]:.4f}x + {intercept:.4f}"
elif best_degree == 3:
    label_text = f"Cubic Fit (R² = {best_r2:.4f})"
    equation_text = f"Terminal Growth = {coeffs[7]:.4f}x³ + {coeffs[4]:.4f}x² + {coeffs[1]:.4f}x + {intercept:.4f}"
else:
    label_text = f"Polynomial Fit (degree={best_degree}, R² = {best_r2:.4f})"
    equation_text = f"Polynomial fit of degree {best_degree}"

# Plot the curve
ax.plot(curve_x, curve_y, 'r-', linewidth=2, label=label_text)

# Set labels and title
ax.set_xlabel("Revenue Growth Rate", fontsize=12)
ax.set_ylabel("Terminal Growth Rate", fontsize=12)
ax.set_title("Revenue Growth vs Terminal Growth Rate Equilibrium\n"
             f"(Points yielding current price: ${current_price:.2f}, WACC fixed at {wacc:.2%})", 
             fontsize=14)

# Format axes as percentages
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))

# Add grid and legend
ax.grid(True, alpha=0.3)

# Set axis limits to provide a cleaner visualization - extend to WACC
ax.set_xlim(0.0, 0.21)  # Extended to 21% for padding
y_min = 0.0
y_max = wacc * 0.99  # Extend up to 99% of WACC
ax.set_ylim(y_min, y_max)

# Add horizontal line at WACC to show the theoretical upper limit
ax.axhline(y=wacc, color='gray', linestyle='--', alpha=0.6, 
          label=f"WACC: {wacc:.2%}")

# Add the regression equation text to the plot
# Position text in the upper left
ax.text(0.05, 0.95, equation_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', 
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Add legend
ax.legend(loc='lower right')

# Save the plot
plot_path = os.path.join(output_dir, "test_rev_term_growth_equilibrium.png")
plt.tight_layout()
plt.savefig(plot_path)
plt.close(fig)

# Verify the files were created
print(f"Checking if CSV file exists: {os.path.exists(csv_path)}")
print(f"Checking if plot file exists: {os.path.exists(plot_path)}")

print(f"Saved test plot to {plot_path}")
print("Test completed successfully") 