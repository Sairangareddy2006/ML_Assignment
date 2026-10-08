import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import jarque_bera

# Configuration
ROLL = 'BT2024185'
DATA_DIR = f'{ROLL}'

train_df = pd.read_csv(f'{DATA_DIR}/{ROLL}_train_var2.csv')
test_df = pd.read_csv(f'{DATA_DIR}/{ROLL}_test_var2.csv')

features = ['x1', 'x2', 'x3']
X = train_df[features].values
y = train_df['y'].values
X_test = test_df[features].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

degrees = [1, 2, 4, 6, 8, 9, 10, 12, 14, 16]
ridge_mses = []

print("Evaluating polynomial degrees for var2 (Phase 2):")
for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=kf)
    cv_ridge = cross_validate(ridge, X_scaled, y, cv=kf, scoring={'mse': 'neg_mean_squared_error', 'r2': 'r2'})
    r_mse = -cv_ridge['test_mse'].mean()
    r_r2 = cv_ridge['test_r2'].mean()
    ridge_mses.append(r_mse)
    print(f"Degree {deg:2d} ({X_poly.shape[1]:3d} feats) -> Ridge MSE: {r_mse:.4f} | R2: {r_r2:.4f}")

# Selected Model: Degree 12 Polynomial with Ridge
selected_degree = 12
poly = PolynomialFeatures(degree=selected_degree, include_bias=False)
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

model = RidgeCV(alphas=np.logspace(-2, 3, 40), cv=kf)
model.fit(X_train_scaled, y)

train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)
oof_preds = cross_val_predict(model, X_train_scaled, y, cv=kf)

cv_mse = mean_squared_error(y, oof_preds)
cv_r2 = r2_score(y, oof_preds)
residuals = y - oof_preds
jb_stat, jb_p = jarque_bera(residuals)

print(f"\nFinal Model (Degree {selected_degree} Ridge):")
print(f"Best Alpha: {model.alpha_:.4f}")
print(f"Feature Count: {X_train_poly.shape[1]}")
print(f"Train MSE:  {mean_squared_error(y, train_preds):.4f}")
print(f"Train R2:   {r2_score(y, train_preds):.4f}")
print(f"CV MSE:     {cv_mse:.4f}")
print(f"CV R2:      {cv_r2:.4f}")
print(f"Residual Jarque-Bera p-value: {jb_p:.4f} (Gaussian noise: {jb_p > 0.05})")

# Save Predictions
pred_path = f'{DATA_DIR}/{ROLL}_pred_var2.csv'
pred_df = pd.DataFrame({'y': test_preds})
pred_df.to_csv(pred_path, index=False)
print(f"Saved predictions to {pred_path}")

# Generate and save diagnostic figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(degrees, ridge_mses, 's-', color='#167d8d', label='Ridge MSE')
ax1.axvline(selected_degree, color='r', linestyle='--', label=f'Selected (d={selected_degree})')
ax1.set_xlabel('Degree')
ax1.set_ylabel('CV MSE')
ax1.set_title('var2: Error vs Degree')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.scatter(oof_preds, y, alpha=0.4, color='#167d8d', s=18)
limits = [min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())]
ax2.plot(limits, limits, 'r--', label='1:1 Line')
ax2.set_xlabel('Predicted y')
ax2.set_ylabel('Actual y')
ax2.set_title('var2: Out-of-Fold Parity Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var2_plot.png', dpi=200)
plt.close(fig)
print("Saved plot to var2_plot.png")
