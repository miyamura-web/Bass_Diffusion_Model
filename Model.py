import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Step 1: Data ---
years = np.arange(2016, 2031)
upi_users = np.array([1, 5, 15, 45, 90, 160, 240, 310, 380, 450, 510, 560, 600, 630, 650])
t = np.arange(1, len(years) + 1)

# --- Step 2: Bass Model ---
def bass_model(t, p, q, m):
    return m * (1 - np.exp(-(p + q) * t)) / (1 + (q / p) * np.exp(-(p + q) * t))

params, _ = curve_fit(bass_model, t, upi_users, p0=[0.03, 0.4, 700], maxfev=10000)
p, q, m = params

predicted = bass_model(t, p, q, m)
annual_adopters = np.diff(np.insert(predicted, 0, 0))  # yearly new adopters

# --- Step 3: Normalize to get percentages ---
total_users = predicted[-1]
share = (annual_adopters / total_users) * 100

# --- Step 4: Build DataFrame ---
results = pd.DataFrame({
    "Year": years,
    "Predicted_Cumulative_Users": predicted.round(1),
    "New_Adopters": annual_adopters.round(1),
    "Share_of_Total(%)": share.round(2)
})

print("\n📊 Dynamic Adopter Distribution from Bass Model:")
print(results)

# --- Step 5: Optional Plot ---
plt.figure(figsize=(6,3))
plt.plot(years, predicted, "s--", label="Predicted UPI Users (Bass Model)")
plt.bar(years, annual_adopters, alpha=0.4, label="New Adopters per Year")
plt.title("UPI Adoption Forecast (Bass Diffusion Model)")
plt.xlabel("Year")
plt.ylabel("Users (millions)")
plt.legend()
plt.grid(True)
plt.show()

print(f"\nCoefficient of Innovation (p): {p:.4f}")
print(f"Coefficient of Imitation (q): {q:.4f}")
print(f"Market Potential (m): {m:.0f} million users")
