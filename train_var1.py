import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
ROLL = 'BT2024185'
DATA_DIR = f'{ROLL}'

train_df = pd.read_csv(f'{DATA_DIR}/{ROLL}_train_var1.csv')
test_df = pd.read_csv(f'{DATA_DIR}/{ROLL}_test_var1.csv')

features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X = train_df[features].values
y = train_df['y'].values
X_test = test_df[features].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)

degrees = [1, 2, 3, 4, 5, 6]
ridge_mses = []
lasso_mses = []

print("Evaluating polynomial degrees for var1 (Phase 1):")
for deg in degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    ridge = RidgeCV(alphas=np.logspace(-3, 2, 20), cv=kf)
    cv_ridge = cross_validate(ridge, X_scaled, y, cv=kf, scoring={'mse': 'neg_mean_squared_error', 'r2': 'r2'})
    r_mse = -cv_ridge['test_mse'].mean()
    r_r2 = cv_ridge['test_r2'].mean()
    ridge_mses.append(r_mse)

    lasso = LassoCV(alphas=np.logspace(-3, 1, 25), cv=kf, max_iter=10000, random_state=42)
    cv_lasso = cross_validate(lasso, X_scaled, y, cv=kf, scoring={'mse': 'neg_mean_squared_error', 'r2': 'r2'})
    l_mse = -cv_lasso['test_mse'].mean()
    l_r2 = cv_lasso['test_r2'].mean()
    lasso_mses.append(l_mse)
    print(f"Degree {deg:2d} ({X_poly.shape[1]:3d} feats) -> Ridge MSE: {r_mse:.4f} (R2: {r_r2:.4f}) | Lasso MSE: {l_mse:.4f} (R2: {l_r2:.4f})")

# Selected Model: Degree 5 Polynomial with Lasso
selected_degree = 5
poly = PolynomialFeatures(degree=selected_degree, include_bias=False)
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

model = LassoCV(alphas=np.logspace(-3, 1, 30), cv=kf, max_iter=20000, random_state=42)
model.fit(X_train_scaled, y)

train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)
oof_preds = cross_val_predict(model, X_train_scaled, y, cv=kf)

nonzero_count = int(np.sum(model.coef_ != 0))
cv_mse = mean_squared_error(y, oof_preds)
cv_r2 = r2_score(y, oof_preds)

print(f"\nFinal Model (Degree {selected_degree} Lasso):")
print(f"Best Alpha: {model.alpha_:.5f}")
print(f"Selected Non-Zero Features: {nonzero_count} / {X_train_poly.shape[1]}")
print(f"Train MSE: {mean_squared_error(y, train_preds):.4f}")
print(f"Train R2:  {r2_score(y, train_preds):.4f}")
print(f"CV MSE:    {cv_mse:.4f}")
print(f"CV R2:     {cv_r2:.4f}")

# Save Predictions
pred_path = f'{DATA_DIR}/{ROLL}_pred_var1.csv'
pred_df = pd.DataFrame({'y': test_preds})
pred_df.to_csv(pred_path, index=False)
print(f"Saved predictions to {pred_path}")

# Generate and save diagnostic figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.plot(degrees, ridge_mses, 's-', label='Ridge MSE', color='#167d8d')
ax1.plot(degrees, lasso_mses, 'o-', label='Lasso MSE', color='#b66128')
ax1.axvline(selected_degree, color='r', linestyle='--', label=f'Selected (d={selected_degree})')
ax1.set_xlabel('Degree')
ax1.set_ylabel('CV MSE')
ax1.set_title('var1: Error vs Degree')
ax1.legend()
ax1.grid(True, alpha=0.3)

residuals = y - oof_preds
ax2.scatter(oof_preds, y, alpha=0.4, s=18, color='#167d8d')
limits = [min(y.min(), oof_preds.min()), max(y.max(), oof_preds.max())]
ax2.plot(limits, limits, 'r--', label='1:1 Line')
ax2.set_xlabel('Predicted y')
ax2.set_ylabel('Actual y')
ax2.set_title('var1: Out-of-Fold Parity Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var1_plot.png', dpi=200)
plt.close(fig)
print("Saved plot to var1_plot.png")
