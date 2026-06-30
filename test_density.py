import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_excel('Nature paper.xlsx', header=1)

# Find SMILES and Density columns
smiles_col = [c for c in df.columns if 'smiles' in str(c).lower()][0]
density_col = [c for c in df.columns if 'density' in str(c).lower()][0]

# Keep only rows with valid Density and valid SMILES
df_valid = df.dropna(subset=[smiles_col, density_col]).copy()

# Generate RDKit features
def get_features(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mw = Descriptors.MolWt(mol)
            mr = Descriptors.MolMR(mol)
            return mw, mr
    except:
        pass
    return np.nan, np.nan

df_valid[['MW', 'MolMR']] = df_valid[smiles_col].apply(lambda x: pd.Series(get_features(str(x))))
df_valid = df_valid.dropna(subset=['MW', 'MolMR'])

# Create theoretical empirical feature
df_valid['MW_over_MR'] = df_valid['MW'] / df_valid['MolMR']

X = df_valid[['MW', 'MolMR', 'MW_over_MR']].values
y = pd.to_numeric(df_valid[density_col], errors='coerce').values

# Drop any remaining nans in y
valid_y_mask = ~np.isnan(y)
X = X[valid_y_mask]
y = y[valid_y_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Simple Linear Regression (Empirical Equation)
lr = LinearRegression()
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, preds_lr)

# Model 2: XGBoost (Non-linear relationship)
xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)
preds_xgb = xgb.predict(X_test)
r2_xgb = r2_score(y_test, preds_xgb)

print(f"Total valid density samples: {len(y)}")
print(f"Linear Regression R2: {r2_lr:.4f}")
print(f"Empirical Equation: Density = {lr.coef_[0]:.4f}*(MW) + {lr.coef_[1]:.4f}*(MolMR) + {lr.coef_[2]:.4f}*(MW/MolMR) + {lr.intercept_:.4f}")
print(f"XGBoost Regressor R2: {r2_xgb:.4f}")

