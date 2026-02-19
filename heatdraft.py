import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


import sys
# Force UTF-8 output for Windows consoles
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust local ML pipeline for HeatDraft data.")
    parser.add_argument("input_file", nargs="?", help="Path to input .csv/.xlsx")
    parser.add_argument("--target", default="Removal rate (%)", help="Target column name")
    parser.add_argument("--trials", type=int, default=90, help="Total Optuna trials across models")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction")
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=90.0,
        help="Removal rate threshold used to define high-performance data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", default="outputs", help="Directory for reports/plots")
    
    # Feature Selection Args
    parser.add_argument("--no-feature-drop", action="store_true", help="Disable feature dropping pipeline")
    parser.add_argument("--corr-threshold", type=float, default=0.92, help="Correlation threshold for dropping")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF threshold")
    parser.add_argument("--nzv-threshold", type=float, default=0.01, help="Near-zero variance threshold")
    parser.add_argument(
        "--max-cat-cardinality",
        type=int,
        default=60,
        help="Drop categorical columns whose train-cardinality exceeds this value",
    )
    parser.add_argument(
        "--cat-unique-ratio-threshold",
        type=float,
        default=0.95,
        help="Drop categorical columns with near-unique values (ID-like)",
    )
    parser.add_argument(
        "--disable-moe",
        action="store_true",
        help="Disable KMeans-gated Mixture-of-Experts stage",
    )
    
    # Masking Args (Simplified for CLI - detailed masking usually needs code/config)
    parser.add_argument("--target-mask-range", nargs=2, type=float, help="Mask target values in [low, high] (replace with NaN)")
    
    return parser.parse_args()


def auto_pick_input_file(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = sorted(list(Path.cwd().glob("*.csv")) + list(Path.cwd().glob("*.xlsx")))
    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise SystemExit("No input_file provided and no .csv/.xlsx found in current directory.")

    names = "\n".join(f"- {c.name}" for c in candidates)
    raise SystemExit(
        "Multiple data files found. Pass one explicitly, e.g. `python heatdraft.py your_file.xlsx`\n"
        f"Candidates:\n{names}"
    )


def normalize_col_name(name: str) -> str:
    s = str(name).strip()
    if any(tok in s for tok in ["Ã", "Â", "â", "Î", "ð"]):
        s = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    replacements = {
        "\u2009": " ",
        "\u202f": " ",
        "\xa0": " ",
        "Â·": "·",
        "â€‰": " ",
        "Î³": "γ",
        "âˆ†": "∆",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return " ".join(s.split())


def load_dataframe(path: Path, target_hint: str) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, header=0)
    else:
        df = pd.read_csv(path, header=0)
    df.columns = [normalize_col_name(c) for c in df.columns]
    if target_hint in df.columns:
        return df
    inferred = infer_target_column(df.columns.tolist(), target_hint)
    if inferred:
        return df
    raise SystemExit(f"Target hint '{target_hint}' not found after strict load with header=0.")



def infer_target_column(columns: List[str], target_hint: str) -> Optional[str]:
    if target_hint in columns:
        return target_hint
    low = {c.lower(): c for c in columns}
    if target_hint.lower() in low:
        return low[target_hint.lower()]

    for c in columns:
        lc = c.lower()
        if "removal" in lc and "%" in lc:
            return c
    return None


def apply_target_mask(
    df: pd.DataFrame, target_col: str, mask_range: Tuple[float, float]
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Mask (set to NaN) target values that fall INSIDE [low, high].
    Returns the modified dataframe and the index of masked rows.
    """
    low, high = mask_range
    mask = (df[target_col] >= low) & (df[target_col] <= high)
    masked_idx = df.index[mask]
    df = df.copy()
    df.loc[mask, target_col] = np.nan
    print(f"  [Target Mask] Masked {mask.sum()} rows where "
          f"{target_col} ∈ [{low}, {high}]")
    return df, masked_idx


def add_rdkit_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    smiles_col = None
    for c in df.columns:
        if c.lower() == "smiles":
            smiles_col = c
            break

    if smiles_col is None:
        return df

    descriptor_fns = {
        "RD_MW": Descriptors.MolWt,
        "RD_LogP": Descriptors.MolLogP,
        "RD_TPSA": Descriptors.TPSA,
        "RD_HBA": Descriptors.NumHAcceptors,
        "RD_HBD": Descriptors.NumHDonors,
        "RD_Rings": Descriptors.RingCount,
        "RD_RotBonds": Descriptors.NumRotatableBonds,
        "RD_FracCSP3": Descriptors.FractionCSP3,
    }

    mols = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) and x.strip() else None)

    for feat, fn in descriptor_fns.items():
        df[feat] = mols.apply(lambda m: fn(m) if m is not None else np.nan)

    valid = mols.notna().mean() * 100.0
    print(f"RDKit descriptors added from '{smiles_col}' (valid SMILES: {valid:.1f}%).")
    return df


# ═══════════════════════════════════════════════════════════════════
#  FEATURE SIMILARITY DROPPING
# ═══════════════════════════════════════════════════════════════════

def drop_near_zero_variance(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Drop numeric features with variance below threshold (after scaling)."""
    num = X.select_dtypes(include=[np.number])
    if num.empty:
        return X, []
    # Normalize before variance check to avoid unit sensitivity
    scaled = (num - num.mean()) / (num.std().replace(0, 1))
    var = scaled.var()
    dropped = var[var < threshold].index.tolist()
    if dropped:
        print(f"  [NZV] Dropped {len(dropped)} near-zero-variance features: {dropped}")
    return X.drop(columns=dropped, errors="ignore"), dropped


def build_correlation_clusters(X: pd.DataFrame, threshold: float,
                                method: str) -> List[List[str]]:
    """
    Build clusters of highly correlated features using greedy grouping.
    Each cluster is a list of feature names.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return []

    X_filled = X[num_cols].fillna(X[num_cols].median())

    if method == "spearman":
        corr = X_filled.rank().corr(method="pearson").abs()
    elif method == "both":
        c1 = X_filled.corr(method="pearson").abs()
        c2 = X_filled.rank().corr(method="pearson").abs()
        corr = (c1 + c2) / 2
    else:  # pearson
        corr = X_filled.corr(method="pearson").abs()

    visited = set()
    clusters = []
    for col in num_cols:
        if col in visited:
            continue
        similar = corr[col][corr[col] >= threshold].index.tolist()
        similar = [c for c in similar if c != col]
        if similar:
            cluster = [col] + [c for c in similar if c not in visited]
            visited.update(cluster)
            clusters.append(cluster)
        else:
            visited.add(col)
    return clusters


def select_cluster_representative(cluster: List[str], X: pd.DataFrame,
                                   y: pd.Series, method: str) -> str:
    """
    From a cluster of correlated features, pick the most informative one.
    Methods: 'mi' (mutual information), 'pearson' (correlation with target),
             'spearman', 'variance'
    """
    num_cols = [c for c in cluster if c in X.select_dtypes(include=[np.number]).columns]
    if not num_cols:
        return cluster[0]

    X_sub = X[num_cols].fillna(X[num_cols].median())
    y_clean = y.fillna(y.median())

    if method == "mi":
        scores = mutual_info_regression(
            X_sub.values, y_clean.values, random_state=42
        )
        return num_cols[int(np.argmax(scores))]

    if method in ("pearson", "spearman"):
        corrs = [abs(X_sub[c].corr(y_clean, method=method)) for c in num_cols]
        return num_cols[int(np.argmax(corrs))]

    if method == "variance":
        return X_sub.std().idxmax()
    raise ValueError(f"Unknown cluster representative method: {method}")


def drop_correlated_features(
    X: pd.DataFrame, y: pd.Series,
    corr_threshold: float, corr_method: str,
    use_mi: bool, mi_corr_floor: float
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full feature-similarity dropping pipeline:
      1. Build correlation clusters at corr_threshold
      2. Within each cluster, keep most informative feature (by MI or correlation)
      3. Optionally form finer MI-based clusters at mi_corr_floor
    Returns reduced X and a detailed report.
    """
    report = {
        "original_features": X.shape[1],
        "clusters": [],
        "dropped": [],
        "kept": [],
    }

    clusters = build_correlation_clusters(X, corr_threshold, corr_method)

    to_drop = set()
    for cluster in clusters:
        rep = select_cluster_representative(cluster, X, y, "mi" if use_mi else corr_method)
        drop_in_cluster = [c for c in cluster if c != rep]
        to_drop.update(drop_in_cluster)
        report["clusters"].append({
            "kept": rep,
            "dropped": drop_in_cluster,
            "size": len(cluster)
        })

    if use_mi and mi_corr_floor < corr_threshold:
        mi_clusters = build_correlation_clusters(X, mi_corr_floor, corr_method)
        for cluster in mi_clusters:
            remaining = [c for c in cluster if c not in to_drop]
            if len(remaining) < 2:
                continue
            rep = select_cluster_representative(remaining, X, y, "mi")
            for c in remaining:
                if c != rep:
                    to_drop.add(c)

    X_reduced = X.drop(columns=list(to_drop), errors="ignore")
    report["dropped"] = sorted(to_drop)
    report["kept"] = sorted(X_reduced.columns.tolist())
    report["final_features"] = X_reduced.shape[1]
    report["n_dropped"] = len(to_drop)
    return X_reduced, report


def drop_high_vif_features(
    X: pd.DataFrame, vif_threshold: float
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Iteratively drop the feature with the highest VIF until all VIFs
    are below vif_threshold. Returns reduced X, dropped feature list,
    and the final VIF table.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[num_cols].fillna(X[num_cols].median())
    # Drop constant cols to avoid division by zero in VIF
    X_num = X_num.loc[:, X_num.std() > 0]
    cols = X_num.columns.tolist()
    dropped_vif = []

    while True:
        if len(cols) < 2:
            break
        data = X_num[cols].values
        vif_values = [
            variance_inflation_factor(data, i)
            for i in range(data.shape[1])
        ]
        max_vif = max(vif_values)
        if max_vif <= vif_threshold:
            break
        worst = cols[int(np.argmax(vif_values))]
        print(f'  [VIF] Dropping "{worst}" (VIF={max_vif:.2f} > {vif_threshold})')
        dropped_vif.append(worst)
        cols.remove(worst)

    vif_data = []
    if len(cols) >= 2:
        data = X_num[cols].values
        for i, c in enumerate(cols):
            vif_data.append({"feature": c, "VIF": round(variance_inflation_factor(data, i), 3)})

    vif_table = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
    X_out = X.drop(columns=dropped_vif, errors="ignore")
    return X_out, dropped_vif, vif_table


def full_feature_selection(
    X: pd.DataFrame, y: pd.Series,
    nzv_threshold: float,
    sparsity_threshold: float,
    corr_threshold: float,
    corr_method: str,
    enable_vif: bool,
    vif_threshold: float,
    enable_mi: bool,
    mi_corr_floor: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Orchestrates the complete feature selection pipeline:
      Step 1: Drop sparse features (< sparsity_threshold non-null)
      Step 2: Drop near-zero-variance features
      Step 3: Drop correlated features (Pearson/Spearman clusters)
      Step 4: Drop high-VIF features (multicollinearity)
    """
    full_report = {"steps": {}}
    n_start = X.shape[1]

    # Step 1: Sparsity
    sparse_dropped = [c for c in X.columns if X[c].notna().mean() < sparsity_threshold]
    X = X.drop(columns=sparse_dropped, errors="ignore")
    full_report["steps"]["sparsity"] = {
        "dropped": sparse_dropped, "remaining": X.shape[1]
    }
    if sparse_dropped:
        print(f"  [Sparsity] Dropped {len(sparse_dropped)} sparse features: {sparse_dropped}")

    # Step 2: Near-zero variance
    X, nzv_dropped = drop_near_zero_variance(X, nzv_threshold)
    full_report["steps"]["near_zero_variance"] = {
        "dropped": nzv_dropped, "remaining": X.shape[1]
    }

    # Step 3: Correlation clusters
    print(f"  [Corr] Building correlation clusters (threshold={corr_threshold}, method={corr_method})...")
    X, corr_report = drop_correlated_features(
        X, y, corr_threshold, corr_method, enable_mi, mi_corr_floor
    )
    full_report["steps"]["correlation"] = corr_report
    print(f'  [Corr] {corr_report["n_dropped"]} features dropped, '
          f'{corr_report["final_features"]} remaining')

    # Step 4: VIF
    vif_table = pd.DataFrame()
    if enable_vif:
        print(f"  [VIF] Running VIF analysis (threshold={vif_threshold})...")
        X, vif_dropped, vif_table = drop_high_vif_features(X, vif_threshold)
        full_report["steps"]["vif"] = {
            "dropped": vif_dropped, "remaining": X.shape[1]
        }
        print(f"  [VIF] {len(vif_dropped)} features dropped, {X.shape[1]} remaining")

    full_report["n_start"] = n_start
    full_report["n_final"] = X.shape[1]
    full_report["total_dropped"] = n_start - X.shape[1]
    full_report["vif_table"] = vif_table
    return X, full_report


def drop_high_cardinality_categoricals(
    X_train: pd.DataFrame,
    frames_to_filter: List[pd.DataFrame],
    max_cardinality: int,
    unique_ratio_threshold: float,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    dropped: List[str] = []
    details: List[Dict[str, Any]] = []
    n_rows = max(1, X_train.shape[0])

    for col in cat_cols:
        non_null = X_train[col].notna().sum()
        if non_null == 0:
            continue
        uniq = int(X_train[col].nunique(dropna=True))
        uniq_ratio = uniq / n_rows
        lc = col.lower()
        id_like = any(tok in lc for tok in ["id", "name", "smiles", "inchi", "uuid"])
        if uniq > max_cardinality or uniq_ratio >= unique_ratio_threshold or (id_like and uniq > 20):
            dropped.append(col)
            details.append(
                {
                    "column": col,
                    "n_unique_train": uniq,
                    "unique_ratio_train": round(float(uniq_ratio), 4),
                    "id_like": id_like,
                }
            )

    filtered = [f.drop(columns=dropped, errors="ignore") for f in frames_to_filter]
    report = {"dropped_columns": dropped, "details": details}
    return filtered, report


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def model_search_space(model_name: str, trial: optuna.Trial, seed: int):
    if model_name == "xgboost":
        return xgb.XGBRegressor(
            objective="reg:absoluteerror",  # Changed to MAE for robustness
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 15.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )

    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 300, 1400),
            max_depth=trial.suggest_int("max_depth", 4, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            random_state=seed,
            n_jobs=-1,
            criterion="absolute_error", # Experimenting with MAE criterion if possible, else default squared_error
        )

    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 300, 1400),
            max_depth=trial.suggest_int("max_depth", 4, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            random_state=seed,
            n_jobs=-1,
            criterion="absolute_error", # Try MAE
        )

    if model_name == "hist_gb":
        return HistGradientBoostingRegressor(
            loss="absolute_error", # MAE
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 16),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 120),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 60),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            random_state=seed,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    trials: int,
    seed: int,
    high_threshold: float,
    weight_ratio: float,
) -> Tuple[Pipeline, Dict[str, float]]:
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        reg = model_search_space(model_name, trial, seed)
        
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Dynamic Weighting
            w_tr = np.where(y_tr >= high_threshold, weight_ratio, 1.0)
            w_val = np.where(y_val >= high_threshold, weight_ratio, 1.0)
            
            # Preprocess
            X_tr_trans = preprocessor.fit_transform(X_tr, y_tr)
            X_val_trans = preprocessor.transform(X_val)
            
            # Fit with weights - fail if not supported
            reg.fit(X_tr_trans, y_tr, sample_weight=w_tr)
                
            preds = reg.predict(X_val_trans)
            # Optimize for MAE now since we switched objectives
            mae = mean_absolute_error(y_val, preds, sample_weight=w_val)
            fold_scores.append(mae)

        return -1.0 * np.mean(fold_scores) # Optuna minimizes, so we negate MAE

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max(8, trials))

    best_reg = model_search_space(model_name, optuna.trial.FixedTrial(study.best_trial.params), seed)
    best_pipe = Pipeline(steps=[("prep", preprocessor), ("model", best_reg)])
    
    # Final fit on all training data with weights
    w_train = np.where(y_train >= high_threshold, weight_ratio, 1.0)
    
    # Fail if weight not accepted
    best_pipe.fit(X_train, y_train, model__sample_weight=w_train)

    return best_pipe, study.best_trial.params



def target_transform(y: pd.Series | np.ndarray) -> np.ndarray:
    # Stable bounded transform for percentage targets in [0, 100].
    # 1) scale to [0,1], 2) clip off exact edges, 3) logit
    eps = 1e-4
    v = np.array(y, dtype=float)
    p = np.clip(v / 100.0, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def target_inverse_transform(y_trans: pd.Series | np.ndarray) -> np.ndarray:
    # inverse-logit, then rescale back to [0, 100]
    v = np.array(y_trans, dtype=float)
    p = 1.0 / (1.0 + np.exp(-np.clip(v, -30.0, 30.0)))
    return 100.0 * p


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_high_focus(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_threshold: float,
) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    high_hit_rate = float(np.mean(y_pred >= high_threshold))
    return {
        "model": name,
        "rmse_high": rmse,
        "mae_high": mae,
        "r2_high": r2,
        "high_hit_rate": high_hit_rate,
    }


def build_low_failure_report(
    best_model,
    X_low: pd.DataFrame,
    y_low: pd.Series,
    X_high_train: pd.DataFrame,
    high_threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if len(X_low) == 0:
        return {
            "low_rows": 0,
            "false_high_rate": float("nan"),
            "mean_overprediction": float("nan"),
            "low_rmse": float("nan"),
            "low_mae": float("nan"),
        }, pd.DataFrame()

    # Model predicts transformed.
    preds_trans = best_model.predict(X_low)
    preds_low = pd.Series(target_inverse_transform(preds_trans), index=X_low.index)
    
    residual = preds_low - y_low
    false_high_mask = preds_low >= high_threshold

    summary = {
        "low_rows": int(len(X_low)),
        "false_high_rate": float(false_high_mask.mean()),
        "mean_overprediction": float(residual.mean()),
        "low_rmse": float(np.sqrt(mean_squared_error(y_low, preds_low))),
        "low_mae": float(mean_absolute_error(y_low, preds_low)),
    }

    numeric_cols = X_low.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return summary, pd.DataFrame()

    high_profile = X_high_train[numeric_cols].median(numeric_only=True)
    low_profile = X_low[numeric_cols].median(numeric_only=True)

    prof = pd.DataFrame(
        {
            "high_train_median": high_profile,
            "low_median": low_profile,
        }
    ).dropna()
    prof["abs_gap"] = (prof["low_median"] - prof["high_train_median"]).abs()
    prof = prof.sort_values("abs_gap", ascending=False).head(12)
    return summary, prof.reset_index().rename(columns={"index": "feature"})


class KMeansMoERegressor:
    def __init__(
        self,
        gate_cols: List[str],
        gate_imputer: SimpleImputer,
        gate_scaler: RobustScaler,
        gate_pca: PCA,
        gate_model: KMeans,
        experts: Dict[int, Pipeline],
    ) -> None:
        self.gate_cols = gate_cols
        self.gate_imputer = gate_imputer
        self.gate_scaler = gate_scaler
        self.gate_pca = gate_pca
        self.gate_model = gate_model
        self.experts = experts

    def _gate_labels(self, X: pd.DataFrame) -> np.ndarray:
        X_gate = X[self.gate_cols]
        z = self.gate_imputer.transform(X_gate)
        z = self.gate_scaler.transform(z)
        z = self.gate_pca.transform(z)
        return self.gate_model.predict(z)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        labels = self._gate_labels(X)
        preds = np.zeros(X.shape[0], dtype=float)
        for cluster_id in np.unique(labels):
            if int(cluster_id) not in self.experts:
                raise RuntimeError(f"No trained expert for predicted cluster {cluster_id}.")
            idx = np.where(labels == cluster_id)[0]
            preds[idx] = self.experts[int(cluster_id)].predict(X.iloc[idx])
        return preds


def build_kmeans_moe(
    expert_template: Pipeline,
    X_train: pd.DataFrame,
    y_train_trans: pd.Series,
    high_threshold_trans: float,
    weight_ratio: float,
    seed: int,
    n_clusters: int = 2,
    min_cluster_rows: int = 25,
) -> Tuple[KMeansMoERegressor, Dict[str, Any]]:
    gate_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(gate_cols) < 2:
        raise RuntimeError("MoE requires at least 2 numeric columns for gating.")

    gate_imputer = SimpleImputer(strategy="median")
    gate_scaler = RobustScaler()
    X_gate = gate_imputer.fit_transform(X_train[gate_cols])
    X_gate = gate_scaler.fit_transform(X_gate)

    n_comp = min(6, X_gate.shape[1], X_gate.shape[0] - 1)
    if n_comp < 1:
        raise RuntimeError("Insufficient rows for PCA/KMeans gating.")
    gate_pca = PCA(n_components=n_comp, random_state=seed)
    X_gate_pca = gate_pca.fit_transform(X_gate)

    gate_model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = gate_model.fit_predict(X_gate_pca)

    experts: Dict[int, Pipeline] = {}
    cluster_rows: Dict[str, int] = {}

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_cluster = int(cluster_mask.sum())
        cluster_rows[f"cluster_{cluster_id}"] = n_cluster
        if n_cluster < min_cluster_rows:
            raise RuntimeError(
                f"Cluster {cluster_id} has {n_cluster} rows, below minimum {min_cluster_rows}."
            )

        expert = clone(expert_template)
        X_c = X_train.iloc[cluster_mask]
        y_c = y_train_trans.iloc[cluster_mask]
        w_c = np.where(y_c >= high_threshold_trans, weight_ratio, 1.0)
        expert.fit(X_c, y_c, model__sample_weight=w_c)
        experts[cluster_id] = expert

    moe = KMeansMoERegressor(
        gate_cols=gate_cols,
        gate_imputer=gate_imputer,
        gate_scaler=gate_scaler,
        gate_pca=gate_pca,
        gate_model=gate_model,
        experts=experts,
    )
    report = {
        "enabled": True,
        "n_clusters": n_clusters,
        "gate_numeric_features": len(gate_cols),
        "cluster_rows": cluster_rows,
        "experts_trained": sorted(int(k) for k in experts.keys()),
    }
    return moe, report


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = auto_pick_input_file(args.input_file)
    print(f"Using input file: {input_path}")

    df = load_dataframe(input_path, args.target)
    df = add_rdkit_descriptors(df)

    target_col = infer_target_column(df.columns.tolist(), args.target)
    if not target_col:
        raise SystemExit(
            f"Target column '{args.target}' not found. Available columns include: {df.columns[:12].tolist()} ..."
        )

    # --- Step 2: Masking ---
    masking_report = {}

    if args.target_mask_range:
        df, masked_idx = apply_target_mask(df, target_col, tuple(args.target_mask_range))
        masking_report['target_mask'] = {
            'range': args.target_mask_range,
            'masked_rows': len(masked_idx)
        }

    # --- Step 3: Prepare X, y ---
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()

    drop_like = ["number", "index", "unnamed: 0"]
    leak_cols = [c for c in df.columns if c.lower() in drop_like]

    X_raw = df.drop(columns=[target_col] + leak_cols, errors="ignore")
    y = df[target_col]  # Original y
    high_mask_all = y >= args.high_threshold

    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X_raw,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=high_mask_all if high_mask_all.sum() > 10 else None,
    )

    # --- Step 4: Drop categorical time-bombs using TRAIN only ---
    [X_train_pre, X_test_pre, X_full_pre], cat_report = drop_high_cardinality_categoricals(
        X_train=X_train_raw,
        frames_to_filter=[X_train_raw, X_test_raw, X_raw],
        max_cardinality=args.max_cat_cardinality,
        unique_ratio_threshold=args.cat_unique_ratio_threshold,
    )
    if cat_report["dropped_columns"]:
        print(f"Dropped {len(cat_report['dropped_columns'])} high-cardinality/id-like categorical columns.")

    # --- Step 5: Feature selection fit on TRAIN only, then apply ---
    feature_selection_report = {}
    selected_columns = X_train_pre.columns.tolist()

    if not args.no_feature_drop:
        print("\nRunning feature selection pipeline on TRAIN split only...")
        X_train_sel, feature_selection_report = full_feature_selection(
            X=X_train_pre,
            y=y_train_orig,
            nzv_threshold=args.nzv_threshold,
            sparsity_threshold=0.05,
            corr_threshold=args.corr_threshold,
            corr_method="pearson",
            enable_vif=True,
            vif_threshold=args.vif_threshold,
            enable_mi=True,
            mi_corr_floor=0.85,
        )
        selected_columns = X_train_sel.columns.tolist()
        print(f"  Final TRAIN features: {len(selected_columns)} (dropped {feature_selection_report['total_dropped']})")
    else:
        selected_columns = [c for c in X_train_pre.columns if X_train_pre[c].notna().mean() >= 0.05]
        print(f"  Feature dropping DISABLED. Kept {len(selected_columns)} features (train-sparsity check only).")

    X_train = X_train_pre[selected_columns].copy()
    X_test = X_test_pre[selected_columns].copy()
    X_full = X_full_pre[selected_columns].copy()

    print("Applying stable logit target transform on percentage target...")
    y_train_trans = pd.Series(target_transform(y_train_orig), index=y_train_orig.index)
    high_threshold_trans = target_transform([args.high_threshold])[0]

    high_train_count = int((y_train_orig >= args.high_threshold).sum())
    low_train_count = int((y_train_orig < args.high_threshold).sum())
    if high_train_count < 10:
        print(f"Warning: Only {high_train_count} high-removal rows in TRAIN split. High-zone metrics may be noisy.")

    preprocessor = build_preprocessor(X_train)

    model_names = ["xgboost", "extra_trees", "random_forest", "hist_gb"]
    per_model_trials = max(8, args.trials // len(model_names))

    fitted_models = {}
    best_params_report = {}
    metrics_high = []

    print(
        f"\nTraining objective: maximize performance on HIGH removal rows (>= {args.high_threshold}) with weighted MAE in transformed space."
    )
    print(f"Rows: total={len(df)}, train_high={high_train_count}, train_low={low_train_count}")

    if high_train_count > 0:
        raw_ratio = low_train_count / high_train_count
        weight_ratio = min(max(raw_ratio * 1.5, 10.0), 1000.0)
    else:
        weight_ratio = 10.0
        
    print(f"Dynamic High-Sample Weight: {weight_ratio:.2f}")
    print("\nTuning base models with Optuna (Weighted MAE on Transformed Target)...")

    test_high_mask = y_test_orig >= args.high_threshold
    
    X_test_high = X_test[test_high_mask]
    y_test_high_orig = y_test_orig[test_high_mask]
    
    if len(X_test_high) == 0:
        print("Warning: No high-performance rows in test set. Evaluation metrics for High-Zone will be NaN.")

    for name in model_names:
        print(f"- Tuning {name} ({per_model_trials} trials)")
        model, params = tune_model(
            name,
            X_train,
            y_train_trans,
            preprocessor,
            per_model_trials,
            args.seed,
            high_threshold_trans,
            weight_ratio,
        )
        fitted_models[name] = model
        best_params_report[name] = params
        
        # Predict (Transformed)
        preds_trans = model.predict(X_test)
        preds_orig = target_inverse_transform(preds_trans)
        
        # Evaluate GLOBAL Performance
        global_metrics = evaluate(name, y_test_orig, preds_orig)
        
        # Evaluate on High subset
        if len(X_test_high) > 0:
            # Predict only high subset
            preds_high_trans = model.predict(X_test_high)
            preds_high_orig = target_inverse_transform(preds_high_trans)
            h_metrics = evaluate_high_focus(name, y_test_high_orig, preds_high_orig, args.high_threshold)
        else:
            h_metrics = {
                "model": name, "rmse_high": float("nan"), "mae_high": float("nan"), 
                "r2_high": float("nan"), "high_hit_rate": 0.0
            }
        
        # Combine metrics
        combined = {**h_metrics, "r2_global": global_metrics["r2"], "rmse_global": global_metrics["rmse"]}
        metrics_high.append(combined)

    if not metrics_high:
        raise SystemExit("All models failed to tune. Check errors.")

    metrics_df = pd.DataFrame(metrics_high).sort_values(
        by=["r2_high", "rmse_high", "high_hit_rate"], ascending=[False, True, False]
    ).reset_index(drop=True)
    print("\nHigh-zone model leaderboard (evaluated on High subset of Test data):")
    print(metrics_df.to_string(index=False))
    
    if metrics_df["r2_high"].iloc[0] < 0:
        print("\nNote: Negative High-Zone R2 means predictions variance is larger than the (small) variance of the high-zone subset.")

    print("\nFitting Stacking Ensemble (on transformed data)...")
    
    prep_stack = clone(preprocessor)
    # Fit preprocessor on X_train (y can be anything for fit_transform if not target encoded, but passing y_train_trans is fine)
    X_train_trans = prep_stack.fit_transform(X_train, y_train_trans)
    
    top3 = metrics_df.head(3)["model"].tolist()
    base_estimators = []
    for n in top3:
        reg = fitted_models[n].named_steps["model"]
        base_estimators.append((n, reg))
        
    stack = StackingRegressor(
        estimators=base_estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 20)),
        passthrough=False,
        n_jobs=-1,
    )
    
    w_train = np.where(y_train_trans >= high_threshold_trans, weight_ratio, 1.0)
    
    stack.fit(X_train_trans, y_train_trans, sample_weight=w_train)
    
    final_stack_pipe = Pipeline(steps=[("prep", prep_stack), ("model", stack)])
    
    # Eval stack
    stack_preds_trans = final_stack_pipe.predict(X_test)
    stack_preds_orig = target_inverse_transform(stack_preds_trans)
    
    global_metrics_stack = evaluate("stacking_top3", y_test_orig, stack_preds_orig)
    
    if len(X_test_high) > 0:
        stack_preds_high_trans = final_stack_pipe.predict(X_test_high)
        stack_preds_high_orig = target_inverse_transform(stack_preds_high_trans)
        stack_metrics = evaluate_high_focus("stacking_top3", y_test_high_orig, stack_preds_high_orig, args.high_threshold)
    else:
        stack_metrics = {
             "model": "stacking_top3", "rmse_high": float("nan"), "mae_high": float("nan"), 
             "r2_high": float("nan"), "high_hit_rate": 0.0
        }
    stack_metrics["r2_global"] = global_metrics_stack["r2"]
    stack_metrics["rmse_global"] = global_metrics_stack["rmse"]
    metrics_high.append(stack_metrics)
    best_params_report["stacking_top3"] = {"top3": top3}

    moe_report: Dict[str, Any] = {"enabled": False, "reason": "Disabled by user"}
    moe_model: Optional[KMeansMoERegressor] = None
    if not args.disable_moe:
        moe_model, moe_report = build_kmeans_moe(
            expert_template=final_stack_pipe,
            X_train=X_train,
            y_train_trans=y_train_trans,
            high_threshold_trans=high_threshold_trans,
            weight_ratio=weight_ratio,
            seed=args.seed,
            n_clusters=2,
            min_cluster_rows=25,
        )
        moe_preds_trans = moe_model.predict(X_test)
        moe_preds_orig = target_inverse_transform(moe_preds_trans)
        moe_global = evaluate("moe_kmeans2", y_test_orig, moe_preds_orig)

        if len(X_test_high) > 0:
            moe_preds_high_trans = moe_model.predict(X_test_high)
            moe_preds_high_orig = target_inverse_transform(moe_preds_high_trans)
            moe_high = evaluate_high_focus("moe_kmeans2", y_test_high_orig, moe_preds_high_orig, args.high_threshold)
        else:
            moe_high = {
                "model": "moe_kmeans2", "rmse_high": float("nan"), "mae_high": float("nan"),
                "r2_high": float("nan"), "high_hit_rate": 0.0
            }
        moe_high["r2_global"] = moe_global["r2"]
        moe_high["rmse_global"] = moe_global["rmse"]
        metrics_high.append(moe_high)

    final_metrics = pd.DataFrame(metrics_high).sort_values(
        by=["r2_high", "rmse_high", "high_hit_rate"], ascending=[False, True, False]
    ).reset_index(drop=True)
    print("\nFinal high-zone leaderboard (including ensemble + MoE):")
    print(final_metrics.to_string(index=False))

    winner = final_metrics.iloc[0]["model"]
    if winner == "moe_kmeans2":
        if moe_model is None:
            raise RuntimeError("Winner is MoE but MoE model is not available.")
        best_model = moe_model
    elif winner == "stacking_top3":
        best_model = final_stack_pipe
    else:
        best_model = fitted_models[winner]

    import matplotlib.gridspec as gridspec

    # Predict High for dashboard
    preds_plot_trans = best_model.predict(X_test_high if len(X_test_high) > 0 else X_test)
    preds_plot = target_inverse_transform(preds_plot_trans)
    y_plot = y_test_high_orig if len(X_test_high) > 0 else y_test_orig

    # ── Figure 1: 2×2 dashboard ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # Plot A: Predicted vs Actual
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(y_plot, preds_plot, alpha=0.75, edgecolors="black",
                linewidths=0.25, color="#1976D2", s=60)
    lo = min(float(y_plot.min()), float(preds_plot.min()))
    hi = max(float(y_plot.max()), float(preds_plot.max()))
    ax0.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Ideal")
    r2v = r2_score(y_plot, preds_plot) if len(y_plot) > 1 else 0
    maev = mean_absolute_error(y_plot, preds_plot)
    ax0.set_xlabel("Actual Removal Rate (%)")
    ax0.set_ylabel("Predicted")
    ax0.set_title(f"Predicted vs Actual — High Zone\n{winner}  |  R2={r2v:.3f}  MAE={maev:.2f}")
    ax0.legend(fontsize=9)
    ax0.grid(alpha=0.3)

    # Plot B: Residuals
    ax1 = fig.add_subplot(gs[0, 1])
    residuals = preds_plot - y_plot
    ax1.scatter(y_plot, residuals, alpha=0.7, edgecolors="black",
                linewidths=0.25, color="#F57C00", s=60)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Actual Removal Rate (%)")
    ax1.set_ylabel("Residual (Predicted - Actual)")
    ax1.set_title("Residual Plot — High Zone")
    ax1.grid(alpha=0.3)

    # Plot C: Model comparison bar
    ax2 = fig.add_subplot(gs[1, 0])
    m_plot = final_metrics.dropna(subset=["r2_high"]).copy()
    colors = ["#2E7D32" if m == winner else "#90CAF9" for m in m_plot["model"]]
    bars = ax2.barh(m_plot["model"], m_plot["r2_high"],
                    color=colors, edgecolor="black", linewidth=0.5)
    ax2.axvline(0, color="red", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, m_plot["r2_high"]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)
    ax2.set_xlabel("R2 (High Zone)")
    ax2.set_title("Model R2 Comparison (High Zone)")
    ax2.grid(axis="x", alpha=0.3)

    # Plot D: MAE comparison
    ax3 = fig.add_subplot(gs[1, 1])
    colors_mae = ["#2E7D32" if m == winner else "#EF9A9A" for m in m_plot["model"]]
    bars_mae = ax3.barh(m_plot["model"], m_plot["mae_high"],
                        color=colors_mae, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars_mae, m_plot["mae_high"]):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=9)
    ax3.set_xlabel("MAE (High Zone)")
    ax3.set_title("Model MAE Comparison (High Zone)\n(lower is better)")
    ax3.grid(axis="x", alpha=0.3)

    plt.suptitle(f"HeatDraft ML Pipeline — Best Model: {winner}",
                 fontsize=14, fontweight="bold", y=1.01)
    
    dashboard_path = outdir / "dashboard.png"
    plt.savefig(dashboard_path, dpi=160, bbox_inches="tight")
    plt.close()

    # ── Figure 2: Correlation heatmap ────────────────────────────────────────────
    heatmap_path = None
    if not args.no_feature_drop and X_train_pre.shape[1] <= 80:
        num_before = X_train_pre.select_dtypes(include=[np.number]).columns.tolist()
        num_after = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        fig2, axes2 = plt.subplots(1, 2, figsize=(max(12, len(num_before)//2),
                                                  max(9, len(num_before)//2)))
        
        for ax_corr, cols, title in [
            (axes2[0], num_before, f"Before Dropping ({len(num_before)} features)"),
            (axes2[1], num_after, f"After Dropping ({len(num_after)} features)"),
        ]:
            if len(cols) > 0:
                data = X_train_pre[cols].fillna(X_train_pre[cols].median()) if "Before" in title \
                       else X_train[cols].fillna(X_train[cols].median())
                corr_mat = data.corr(method="pearson")
                mask_tri = np.triu(np.ones_like(corr_mat, dtype=bool))
                sns.heatmap(corr_mat, mask=mask_tri, ax=ax_corr, cmap="coolwarm",
                            center=0, vmin=-1, vmax=1, square=True,
                            linewidths=0.3, annot=len(cols)<=20,
                            fmt=".1f", annot_kws={"size": 7})
                ax_corr.set_title(title, fontsize=11, fontweight="bold")
                ax_corr.tick_params(axis="x", rotation=45, labelsize=7)
                ax_corr.tick_params(axis="y", rotation=0, labelsize=7)
        
        plt.suptitle("Pearson Correlation: Before vs After Feature Dropping",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        heatmap_path = outdir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Low-zone diagnostics
    full_low_mask = y < args.high_threshold
    X_low = X_full.loc[full_low_mask]
    y_low = y.loc[full_low_mask]
    X_high_train = X_train.loc[y_train_orig >= args.high_threshold]

    low_summary, low_feature_gaps = build_low_failure_report(
        best_model=best_model,
        X_low=X_low,
        y_low=y_low,
        X_high_train=X_high_train,
        high_threshold=args.high_threshold,
    )
    
    low_summary_path = outdir / "low_zone_diagnostics.json"
    low_summary_path.write_text(json.dumps(low_summary, indent=2), encoding="utf-8")
    
    low_gap_path = outdir / "low_zone_feature_gaps.csv"
    gap_plot_path = None
    
    if not low_feature_gaps.empty:
        low_feature_gaps.to_csv(low_gap_path, index=False)
        print("\nTop feature gaps (low median vs high-train median):")
        print(low_feature_gaps.to_string(index=False))

        # --- Visualization of Feature Gaps ---
        plt.figure(figsize=(10, 8))
        sorted_gaps = low_feature_gaps.sort_values("abs_gap", ascending=True)
        y_pos = np.arange(len(sorted_gaps))

        plt.barh(
            y_pos - 0.2,
            sorted_gaps["high_train_median"],
            height=0.4,
            align="center",
            label="High Zone Median",
            color="#4CAF50",
            alpha=0.8,
            edgecolor="black", linewidth=0.4
        )
        plt.barh(
            y_pos + 0.2,
            sorted_gaps["low_median"],
            height=0.4,
            align="center",
            label="Low Zone Median",
            color="#F44336",
            alpha=0.8,
            edgecolor="black", linewidth=0.4
        )

        plt.yticks(y_pos, sorted_gaps["feature"])
        plt.xlabel("Median Feature Value")
        plt.ylabel("Feature")
        plt.title("Feature Comparison: High vs. Low Performance Zones")
        plt.legend()
        plt.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        gap_plot_path = outdir / "feature_gaps.png"
        plt.savefig(gap_plot_path, dpi=160)
        plt.close()
    else:
        pd.DataFrame(columns=["feature", "high_train_median", "low_median", "abs_gap"]).to_csv(
            low_gap_path, index=False
        )

    # Save artifact paths
    artifacts = {
        "dashboard": str(dashboard_path),
        "low_zone_diagnostics": str(low_summary_path),
        "low_zone_feature_gaps": str(low_gap_path),
    }
    if heatmap_path:
        artifacts["correlation_heatmap"] = str(heatmap_path)
    if gap_plot_path:
        artifacts["feature_gaps"] = str(gap_plot_path)

    # Final JSON Report
    report = {
        "input_file": str(input_path),
        "target": target_col,
        "high_threshold": args.high_threshold,
        "rows_total": int(df.shape[0]),
        "high_rows": int((y >= args.high_threshold).sum()),
        "low_rows": int(len(X_low)),
        "features_raw": int(X_raw.shape[1]),
        "features_final": int(X_train.shape[1]),
        "winner": winner,
        "masking_report": masking_report,
        "categorical_filter": cat_report,
        "feature_selection": {
            k: v for k, v in feature_selection_report.items()
            if k != "vif_table"
        },
        "moe_report": moe_report,
        "metrics": final_metrics.to_dict(orient="records"),
        "low_zone_diagnostics": low_summary,
        "best_params": best_params_report,
        "artifacts": artifacts,
    }

    report_path = outdir / "model_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    
    # Save VIF table if present
    vif_table = feature_selection_report.get("vif_table", pd.DataFrame())
    if not vif_table.empty:
        vif_table.to_csv(outdir / "vif_table.csv", index=False)

    print(f"\nSaved report: {report_path}")
    print(f"Saved dashboard: {dashboard_path}")
    if heatmap_path:
        print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
