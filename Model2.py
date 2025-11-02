import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Input data (years in millions of users) ---
years = np.arange(2016, 2031)
upi_users = np.array([1, 5, 15, 45, 90, 160, 240, 310, 380, 450, 510, 560, 600, 630, 650])
t = np.arange(1, len(years) + 1)

# --- Bass model ---
def bass_model(t, p, q, m):
    return m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

params, _ = curve_fit(bass_model, t, upi_users, p0=[0.03, 0.4, 700], maxfev=10000)
p, q, m = params

# --- Predicted cumulative users and yearly new adopters ---
predicted_cum = bass_model(t, p, q, m)        # cumulative predicted values (millions)
cum_prev = np.insert(predicted_cum[:-1], 0, 0.0)  # cumulative at start of each year
annual_new = predicted_cum - cum_prev         # new adopters during each year (millions)

# --- Category boundaries by cumulative share (Rogers cumulative cutoffs) ---   Fixed values given by Rogers
# Innovators up to 2.5% -> 0.025
# Early Adopters up to 16% -> 0.16 (0.025 + 0.135)
# Early Majority up to 50% -> 0.50
# Late Majority up to 84% -> 0.84
# Laggards up to 100% -> 1.00
cat_names = ["Innovators", "Early Adopters", "Early Majority", "Late Majority", "Laggards"]
cat_bounds_frac = np.array([0.025, 0.16, 0.50, 0.84, 1.00])   # upper bounds (cumulative)
cat_lower_frac = np.concatenate(([0.0], cat_bounds_frac[:-1])) # lower bounds

# Convert to absolute user thresholds using modelled market potential m
lower_users = cat_lower_frac * m
upper_users = cat_bounds_frac * m

# --- Allocate yearly new adopters to categories (purely modeled) ---
allocated = np.zeros(len(cat_names))  # total modeled users by category (millions)
category_year_contrib = {c: [] for c in cat_names}  # list of (year, users) contributing to that category

for i, year in enumerate(years):
    prev_cum = cum_prev[i]
    curr_cum = predicted_cum[i]
    # iterate categories and compute overlap between [prev_cum, curr_cum] & [lower_users[j], upper_users[j]]
    for j, (low_u, up_u) in enumerate(zip(lower_users, upper_users)):
        if curr_cum <= low_u:
            # this entire year's adoption finishes before this category starts
            continue
        if prev_cum >= up_u:
            # this category already completed before this year
            continue
        overlap_start = max(prev_cum, low_u)
        overlap_end = min(curr_cum, up_u)
        overlap_amount = max(0.0, overlap_end - overlap_start)  # in millions
        if overlap_amount > 0:
            allocated[j] += overlap_amount
            category_year_contrib[cat_names[j]].append((int(year), float(round(overlap_amount, 4))))

# Modeled totals and percentages
modeled_total = allocated.sum()              # sum of allocated users (should be close to predicted_cum[-1])
modeled_pct = (allocated / modeled_total) * 100  # percent share of modeled total (not theoretical)

# Build results DataFrame
df = pd.DataFrame({
    "Category": cat_names,
    "Modeled_Users_millions": np.round(allocated, 3),
    "Modeled_%_of_modeled_total": np.round(modeled_pct, 3),
    "Model_cum_lower_frac": np.round(cat_lower_frac, 3),
    "Model_cum_upper_frac": np.round(cat_bounds_frac, 3)
})

# Add totals row
totals = pd.DataFrame([{
    "Category": "Total",
    "Modeled_Users_millions": np.round(modeled_total, 3),
    "Modeled_%_of_modeled_total": np.round(modeled_pct.sum(), 3),
    "Model_cum_lower_frac": "",
    "Model_cum_upper_frac": ""
}])
df_result = pd.concat([df, totals], ignore_index=True)

# Print summary
print("Model fit summary:")
print(f"p = {p:.4f}, q = {q:.4f}, m (market potential) = {m:.3f} million")
print(f"Predicted cumulative at last year: {predicted_cum[-1]:.3f} million (modeled total used for allocation)\n")

print("Modeled allocation to adopter categories (derived from the Bass curve):")
print(df_result.to_string(index=False))

# Also show year contributions for transparency
print("\nYear contributions (category -> [(year, users_millions), ...])")
for c in cat_names:
    print(f"{c}: {category_year_contrib[c]}")

# Optional: compare to theoretical Rogers percentages if desired
theoretical_pct = np.array([0.025, 0.135, 0.34, 0.34, 0.16]) * 100
comparison = pd.DataFrame({
    "Category": cat_names,
    "Theoretical_%": np.round(theoretical_pct, 2),
    "Modeled_%": np.round(modeled_pct, 3),
    "Modeled_Users_millions": np.round(allocated, 3)
})
print("\nComparison (Theoretical vs Modeled):")
print(comparison.to_string(index=False))

# Plot modeled users per category
plt.figure(figsize=(8,4))
plt.bar(cat_names, allocated, alpha=0.85)
plt.title("Modeled users per adopter category (millions) — derived from Bass curve")
plt.ylabel("Users (millions)")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.show()
