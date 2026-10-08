# ML Assignment 1: Polynomial Regression
Roll No: BT2024185

This repository contains the complete source code, trained models, evaluation figures, and test predictions for Assignment 1.

---

## Repository Structure

```
├── BT2024185/
│   ├── BT2024185_train_var1.csv   # Phase 1 training data
│   ├── BT2024185_test_var1.csv    # Phase 1 test data
│   ├── BT2024185_train_var2.csv   # Phase 2 training data
│   ├── BT2024185_test_var2.csv    # Phase 2 test data
│   ├── BT2024185_pred_var1.csv    # Phase 1 test predictions (Single column 'y', 1000 rows)
│   └── BT2024185_pred_var2.csv    # Phase 2 test predictions (Single column 'y', 1000 rows)
├── train_var1.py                  # Self-contained training & evaluation for Phase 1
├── train_var2.py                  # Self-contained training & evaluation for Phase 2
├── generate_report.py             # Script compiling the academic report PDF
├── var1_plot.png                  # Phase 1 degree error curve & parity plot
├── var2_plot.png                  # Phase 2 degree error curve & parity plot
├── BT2024185_report.pdf           # 4-page final report
└── README.md                      # Documentation and run instructions
```

---

## Setup & Requirements

Ensure Python 3.9+ is installed along with standard scientific libraries:
```bash
pip install numpy pandas scikit-learn matplotlib scipy reportlab
```

---

## How to Run

1. **Train Phase 1 (Steam Turbine Optimization - var1):**
   ```bash
   python train_var1.py
   ```
   - Sweeps polynomial degrees 1 to 6 using 5-fold cross-validation.
   - Standardizes polynomial terms and selects **Degree 5 Polynomial with Lasso** ($\alpha \approx 0.0092$).
   - Outputs test predictions to `BT2024185/BT2024185_pred_var1.csv` and saves `var1_plot.png`.

2. **Train Phase 2 (Subterranean Thermal Reservoir Mapping - var2):**
   ```bash
   python train_var2.py
   ```
   - Sweeps polynomial degrees 1 to 16 using 5-fold cross-validation.
   - Standardizes polynomial terms and selects **Degree 12 Polynomial with Ridge** ($\alpha \approx 2.03$).
   - Outputs test predictions to `BT2024185/BT2024185_pred_var2.csv` and saves `var2_plot.png`.

3. **Recompile the PDF Report:**
   ```bash
   python generate_report.py
   ```
   - Builds `BT2024185_report.pdf` (exactly 4 pages).

---

## Final Results Summary

| Problem | Selected Degree | Regularization | Active Terms | 5-Fold CV MSE | 5-Fold CV R² | Deliverable File |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Phase 1 (var1)** | 5 | Lasso ($\alpha \approx 0.0092$) | 114 / 461 | **0.3168** | **96.78%** | `BT2024185/BT2024185_pred_var1.csv` |
| **Phase 2 (var2)** | 12 | Ridge ($\alpha \approx 2.03$) | 454 / 454 | **0.2697** | **99.46%** | `BT2024185/BT2024185_pred_var2.csv` |

---

## Deliverables Checklist

- [x] `BT2024185_report.pdf` (Concise 4-page academic write-up detailing methodology, baselines, and diagnostics)
- [x] `BT2024185_pred_var1.csv` (1,000 predictions, single column `y`, preserving original test order)
- [x] `BT2024185_pred_var2.csv` (1,000 predictions, single column `y`, preserving original test order)
- [x] Self-contained training scripts with zero external path dependencies.
