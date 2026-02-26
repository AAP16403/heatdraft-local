import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_TARGET_ALIASES = [
    "Removal rate (%)",
    "RemovalRate___",
    "RemovalRate",
    "Removal_rate___",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust local ML pipeline for HeatDraft data.")
    parser.add_argument("input_file", nargs="?", help="Path to input .csv/.xlsx")
    parser.add_argument("--target", default="Removal rate (%)", help="Primary target column name")
    parser.add_argument(
        "--target-aliases",
        nargs="*",
        default=[],
        help="Optional extra target aliases to resolve the target column robustly.",
    )
    parser.add_argument(
        "--header",
        type=int,
        default=-1,
        help="Header row index (0-indexed). Use -1 to auto-search common rows [0..3].",
    )
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
    parser.add_argument("--no-feature-drop", action="store_true", help="Disable feature dropping pipeline")
    parser.add_argument("--corr-threshold", type=float, default=0.92, help="Correlation threshold for dropping")
    parser.add_argument("--vif-threshold", type=float, default=10.0, help="VIF threshold")
    parser.add_argument("--nzv-threshold", type=float, default=0.01, help="Near-zero variance threshold")
    parser.add_argument("--knn-k", type=int, default=5, help="K for KNN-based numeric imputation")
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
    parser.add_argument(
        "--moe-clusters",
        type=int,
        default=2,
        help="Number of MoE clusters.",
    )
    parser.add_argument(
        "--moe-min-cluster-rows",
        type=int,
        default=20,
        help="Hard minimum rows required per MoE cluster.",
    )
    parser.add_argument(
        "--disable-inverse",
        action="store_true",
        help="Disable inverse-design stage (target removal -> suggested filter properties).",
    )
    parser.add_argument(
        "--inverse-targets",
        nargs="+",
        type=float,
        default=[90.0, 95.0, 98.0],
        help="Target removal rates for inverse design recommendations.",
    )
    parser.add_argument(
        "--inverse-samples",
        type=int,
        default=12000,
        help="Number of candidate designs sampled for inverse search.",
    )
    parser.add_argument(
        "--inverse-topk",
        type=int,
        default=12,
        help="Top recommendations saved per target removal rate.",
    )
    parser.add_argument(
        "--inverse-pollutant-col",
        default="TypesOfContaminants",
        help="Feature column used as pollutant condition for inverse design.",
    )
    parser.add_argument(
        "--inverse-pollutants",
        nargs="+",
        default=[],
        help="Pollutant values to condition inverse design on. If omitted, most frequent seen in train are used.",
    )
    parser.add_argument(
        "--inverse-pollutants-file",
        default="",
        help="Optional path (.txt/.csv/.json) with pollutant values (combined with --inverse-pollutants).",
    )
    parser.add_argument(
        "--inverse-controllable-cols",
        nargs="+",
        default=[],
        help="Optional explicit list of controllable filter-property columns for inverse design.",
    )
    parser.add_argument(
        "--inverse-confidence",
        type=float,
        default=0.80,
        help="Confidence level for suggested parameter ranges (e.g., 0.80 gives 10th-90th percentiles).",
    )
    parser.add_argument("--target-mask-range", nargs=2, type=float, help="Mask target values in [low, high] (replace with NaN)")
    parser.add_argument("--dry-run", action="store_true", help="Run data prep only and print sanity checks, then exit")
    return parser.parse_args()


def validate_and_normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.header < -1:
        raise RuntimeError("--header must be -1 (auto) or >= 0.")
    if not (0.0 < float(args.test_size) < 1.0):
        raise RuntimeError("--test-size must be between 0 and 1.")
    if int(args.trials) < 1:
        raise RuntimeError("--trials must be >= 1.")
    if int(args.seed) < 0:
        raise RuntimeError("--seed must be >= 0.")
    if int(args.knn_k) < 1:
        raise RuntimeError("--knn-k must be >= 1.")
    if int(args.moe_clusters) < 2:
        raise RuntimeError("--moe-clusters must be >= 2.")
    if int(args.moe_min_cluster_rows) < 1:
        raise RuntimeError("--moe-min-cluster-rows must be >= 1.")
    if args.target_mask_range:
        lo, hi = float(args.target_mask_range[0]), float(args.target_mask_range[1])
        if lo > hi:
            raise RuntimeError("--target-mask-range must satisfy low <= high.")
        if lo < 0.0 or hi > 100.0:
            raise RuntimeError("--target-mask-range must stay within [0, 100].")
    if float(args.high_threshold) < 0.0 or float(args.high_threshold) > 100.0:
        raise RuntimeError("--high-threshold must be within [0, 100].")
    if int(args.inverse_samples) < 100:
        raise RuntimeError("--inverse-samples must be >= 100.")
    if int(args.inverse_topk) < 1:
        raise RuntimeError("--inverse-topk must be >= 1.")
    if not (0.5 <= float(args.inverse_confidence) < 1.0):
        raise RuntimeError("--inverse-confidence must be in [0.5, 1.0).")


    dedup_targets: List[float] = []
    for t in [float(v) for v in args.inverse_targets]:
        if t not in dedup_targets:
            dedup_targets.append(t)
    args.inverse_targets = dedup_targets
    return args


def auto_pick_input_file(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p

    candidates = sorted(
        list(Path.cwd().glob("*.csv")) +
        list(Path.cwd().glob("*.xlsx")) +
        list(Path.cwd().glob("*.xls"))
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise SystemExit("No input file found. Pass one explicitly: python heatdraft.py your_file.xlsx")

    names = "\n".join(f"- {c.name}" for c in candidates)
    raise SystemExit(
        "Multiple input files found. Pass one explicitly, e.g.:\n"
        "python heatdraft.py your_file.xlsx\n"
        f"Candidates:\n{names}"
    )


def normalize_col_name(name: str) -> str:
    s = str(name).strip()
    if any(tok in s for tok in ["Ã", "Â", "â", "Î", "ð"]):
        s = s.encode("latin1").decode("utf-8")
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


def canonical_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column_name(columns: List[str], candidates: List[str]) -> Optional[str]:
    if not columns:
        return None
    cols = [str(c) for c in columns]
    cand = [str(c).strip() for c in candidates if str(c).strip()]
    if not cand:
        return None


    for t in cand:
        if t in cols:
            return t


    lower_map = {c.lower(): c for c in cols}
    for t in cand:
        if t.lower() in lower_map:
            return lower_map[t.lower()]


    canon_map: Dict[str, str] = {}
    for c in cols:
        canon_map.setdefault(canonical_col_name(c), c)
    for t in cand:
        key = canonical_col_name(t)
        if key in canon_map:
            return canon_map[key]
    return None


def _read_table_with_header(path: Path, header_idx: int) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, header=header_idx)
    else:
        df = pd.read_csv(path, header=header_idx)
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def load_dataframe(
    path: Path,
    target_hint: str,
    target_aliases: List[str],
    header_idx: int,
) -> Tuple[pd.DataFrame, str, int]:
    candidates: List[str] = []
    for c in [target_hint] + list(target_aliases) + DEFAULT_TARGET_ALIASES:
        c2 = str(c).strip()
        if c2 and c2 not in candidates:
            candidates.append(c2)

    search_headers = [header_idx] if header_idx >= 0 else [0, 1, 2, 3]

    last_cols: List[str] = []
    load_errors: List[str] = []
    for h in search_headers:
        try:
            df = _read_table_with_header(path, h)
        except Exception as ex:
            load_errors.append(f"header={h}: {ex}")
            continue

        target_col = resolve_column_name(df.columns.tolist(), candidates)
        if target_col is not None:
            return df, target_col, h
        last_cols = [str(c) for c in df.columns.tolist()]

    cols_preview = ", ".join(last_cols[:30]) if last_cols else "<no columns>"
    extra = f"\nLoad attempts failed:\n- " + "\n- ".join(load_errors) if load_errors else ""
    raise SystemExit(
        "Could not resolve target column from provided aliases.\n"
        f"Tried targets: {candidates}\n"
        f"Tried headers: {search_headers}\n"
        f"Columns preview: {cols_preview}{extra}"
    )


def parse_inverse_pollutant_values(cli_values: List[str], file_path: str) -> List[str]:
    values: List[str] = []

    def _push_token(tok: str) -> None:
        t = str(tok).strip()
        if t and t not in values:
            values.append(t)

    for raw in cli_values or []:
        parts = [p.strip() for p in str(raw).split(",")]
        for p in parts:
            _push_token(p)

    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise RuntimeError(f"Inverse pollutants file not found: {p}")
        suffix = p.suffix.lower()
        if suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and isinstance(data.get("pollutants"), list):
                items = data["pollutants"]
            else:
                raise RuntimeError("JSON pollutant file must be a list or a dict with key 'pollutants'.")
            for it in items:
                _push_token(str(it))
        else:
            text = p.read_text(encoding="utf-8")
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                for part in s.split(","):
                    _push_token(part)

    return values


def apply_target_mask(
    df: pd.DataFrame, target_col: str, mask_range: Tuple[float, float]
) -> Tuple[pd.DataFrame, pd.Index]:
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


def drop_near_zero_variance(X: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    num = X.select_dtypes(include=[np.number])
    if num.empty:
        return X, []
    scaled = (num - num.mean()) / (num.std().replace(0, 1))
    var = scaled.var()
    dropped = var[var < threshold].index.tolist()
    if dropped:
        print(f"  [NZV] Dropped {len(dropped)} near-zero-variance features: {dropped}")
    return X.drop(columns=dropped), dropped


def knn_impute_numeric_frame(
    X_num: pd.DataFrame,
    n_neighbors: int,
    context: str,
) -> pd.DataFrame:
    if X_num.empty:
        return X_num.copy()
    all_nan_cols = [c for c in X_num.columns if X_num[c].notna().sum() == 0]
    if all_nan_cols:
        raise RuntimeError(
            f"KNN imputation failed in {context}: all values are missing in columns {all_nan_cols}."
        )
    imputer = KNNImputer(n_neighbors=max(1, int(n_neighbors)), weights="distance")
    arr = imputer.fit_transform(X_num)
    out = pd.DataFrame(arr, columns=X_num.columns, index=X_num.index)
    return out


def build_correlation_clusters(
    X: pd.DataFrame,
    threshold: float,
    method: str,
    knn_k: int,
) -> List[List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return []

    X_filled = knn_impute_numeric_frame(X[num_cols], n_neighbors=knn_k, context="correlation clustering")

    if method == "spearman":
        corr = X_filled.rank().corr(method="pearson").abs()
    elif method == "both":
        c1 = X_filled.corr(method="pearson").abs()
        c2 = X_filled.rank().corr(method="pearson").abs()
        corr = (c1 + c2) / 2
    else:
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


def select_cluster_representative(
    cluster: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    knn_k: int,
) -> str:
    num_cols = [c for c in cluster if c in X.select_dtypes(include=[np.number]).columns]
    if not num_cols:
        raise RuntimeError("Cluster representative selection requires numeric features.")

    X_sub = knn_impute_numeric_frame(X[num_cols], n_neighbors=knn_k, context="cluster representative")
    y_clean = pd.to_numeric(y, errors="coerce")
    valid = y_clean.notna()
    X_sub = X_sub.loc[valid]
    y_clean = y_clean.loc[valid]
    if len(y_clean) < 3:
        raise RuntimeError("Insufficient finite target values for cluster representative selection.")

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
    use_mi: bool, mi_corr_floor: float,
    knn_k: int,
) -> Tuple[pd.DataFrame, Dict]:
    report = {
        "original_features": X.shape[1],
        "clusters": [],
        "dropped": [],
        "kept": [],
    }

    clusters = build_correlation_clusters(X, corr_threshold, corr_method, knn_k=knn_k)

    to_drop = set()
    for cluster in clusters:
        rep = select_cluster_representative(
            cluster,
            X,
            y,
            "mi" if use_mi else corr_method,
            knn_k=knn_k,
        )
        drop_in_cluster = [c for c in cluster if c != rep]
        to_drop.update(drop_in_cluster)
        report["clusters"].append({
            "kept": rep,
            "dropped": drop_in_cluster,
            "size": len(cluster)
        })

    if use_mi and mi_corr_floor < corr_threshold:
        mi_clusters = build_correlation_clusters(X, mi_corr_floor, corr_method, knn_k=knn_k)
        for cluster in mi_clusters:
            remaining = [c for c in cluster if c not in to_drop]
            if len(remaining) < 2:
                continue
            rep = select_cluster_representative(remaining, X, y, "mi", knn_k=knn_k)
            for c in remaining:
                if c != rep:
                    to_drop.add(c)

    X_reduced = X.drop(columns=list(to_drop))
    report["dropped"] = sorted(to_drop)
    report["kept"] = sorted(X_reduced.columns.tolist())
    report["final_features"] = X_reduced.shape[1]
    report["n_dropped"] = len(to_drop)
    return X_reduced, report


def drop_high_vif_features(
    X: pd.DataFrame, vif_threshold: float, knn_k: int
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = knn_impute_numeric_frame(X[num_cols], n_neighbors=knn_k, context="VIF filtering")
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
    else:
        vif_table = pd.DataFrame(columns=["feature", "VIF"])
    X_out = X.drop(columns=dropped_vif)
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
    knn_k: int,
) -> Tuple[pd.DataFrame, Dict]:
    full_report = {"steps": {}}
    n_start = X.shape[1]

    sparse_dropped = [c for c in X.columns if X[c].notna().mean() < sparsity_threshold]
    X = X.drop(columns=sparse_dropped)
    full_report["steps"]["sparsity"] = {
        "dropped": sparse_dropped, "remaining": X.shape[1]
    }
    if sparse_dropped:
        print(f"  [Sparsity] Dropped {len(sparse_dropped)} sparse features: {sparse_dropped}")

    X, nzv_dropped = drop_near_zero_variance(X, nzv_threshold)
    full_report["steps"]["near_zero_variance"] = {
        "dropped": nzv_dropped, "remaining": X.shape[1]
    }

    print(f"  [Corr] Building correlation clusters (threshold={corr_threshold}, method={corr_method})...")
    X, corr_report = drop_correlated_features(
        X, y, corr_threshold, corr_method, enable_mi, mi_corr_floor, knn_k
    )
    full_report["steps"]["correlation"] = corr_report
    print(f'  [Corr] {corr_report["n_dropped"]} features dropped, '
          f'{corr_report["final_features"]} remaining')

    vif_table = pd.DataFrame()
    if enable_vif:
        print(f"  [VIF] Running VIF analysis (threshold={vif_threshold})...")
        X, vif_dropped, vif_table = drop_high_vif_features(X, vif_threshold, knn_k)
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

    filtered = []
    for f in frames_to_filter:
        missing = [c for c in dropped if c not in f.columns]
        if missing:
            raise KeyError(f"Columns marked for drop are missing in frame: {missing}")
        filtered.append(f.drop(columns=dropped))
    report = {"dropped_columns": dropped, "details": details}
    return filtered, report


def build_preprocessor(X: pd.DataFrame, knn_k: int) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=max(1, int(knn_k)), weights="distance")),
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
            objective="reg:absoluteerror",
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
            criterion="absolute_error",
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
            criterion="absolute_error",
        )

    if model_name == "hist_gb":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
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
            w_tr = np.where(y_tr >= high_threshold, weight_ratio, 1.0)
            w_val = np.where(y_val >= high_threshold, weight_ratio, 1.0)
            X_tr_trans = preprocessor.fit_transform(X_tr, y_tr)
            X_val_trans = preprocessor.transform(X_val)
            reg.fit(X_tr_trans, y_tr, sample_weight=w_tr)
            preds = reg.predict(X_val_trans)
            mae = mean_absolute_error(y_val, preds, sample_weight=w_val)
            fold_scores.append(mae)

        return -1.0 * np.mean(fold_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max(8, trials))

    best_reg = model_search_space(model_name, optuna.trial.FixedTrial(study.best_trial.params), seed)
    best_pipe = Pipeline(steps=[("prep", preprocessor), ("model", best_reg)])
    w_train = np.where(y_train >= high_threshold, weight_ratio, 1.0)
    best_pipe.fit(X_train, y_train, model__sample_weight=w_train)

    return best_pipe, study.best_trial.params


def target_transform(y: Union[pd.Series, np.ndarray]) -> np.ndarray:
    eps = 1e-4
    v = np.array(y, dtype=float)
    if np.isnan(v).any():
        raise ValueError("Target contains NaN values before transform.")
    if ((v < 0.0) | (v > 100.0)).any():
        bad_min = float(np.min(v))
        bad_max = float(np.max(v))
        raise ValueError(
            f"Target values must be within [0, 100]. Found range [{bad_min:.4f}, {bad_max:.4f}]."
        )
    p = np.clip(v / 100.0, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def target_inverse_transform(y_trans: Union[pd.Series, np.ndarray]) -> np.ndarray:
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


def _is_integer_like_series(s: pd.Series) -> bool:
    s_num = pd.to_numeric(s, errors="coerce").dropna()
    if s_num.empty:
        return False
    frac = np.abs(s_num - np.round(s_num))
    return bool((frac < 1e-9).mean() > 0.98)


def _resolve_controllable_columns(
    X_train: pd.DataFrame,
    pollutant_col: str,
    user_cols: List[str],
) -> List[str]:
    if not user_cols:
        raise RuntimeError("Provide --inverse-controllable-cols explicitly. Auto fallback is disabled.")

    cols_resolved: List[str] = []
    missing: List[str] = []
    for c in user_cols:
        resolved = resolve_column_name(X_train.columns.tolist(), [str(c)])
        if resolved is None:
            missing.append(str(c))
        elif resolved not in cols_resolved:
            cols_resolved.append(resolved)
    if missing:
        raise RuntimeError(f"Inverse controllable columns not found in selected features: {missing}")
    cols = [c for c in cols_resolved if c != pollutant_col]
    if not cols:
        raise RuntimeError("No controllable columns remain after removing pollutant column.")
    return cols


def _sample_inverse_candidates(
    X_ref: pd.DataFrame,
    n_samples: int,
    seed: int,
    pool_mask: pd.Series,
    controllable_cols: List[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=np.arange(n_samples), columns=X_ref.columns)

    num_cols = X_ref.select_dtypes(include=[np.number]).columns.tolist()
    if not isinstance(pool_mask, pd.Series):
        raise RuntimeError("Inverse candidate generation failed: pool_mask must be a pandas Series.")
    pool_mask = pool_mask.reindex(X_ref.index)
    if pool_mask.isna().any():
        raise RuntimeError("Inverse candidate generation failed: pollutant mask does not align with feature index.")
    pool_mask = pool_mask.astype(bool)
    if int(pool_mask.sum()) <= 5:
        raise RuntimeError(
            "Not enough conditioning rows in training data for requested pollutant (need > 5)."
        )
    base_pool = X_ref.loc[pool_mask].copy()

    for col in X_ref.columns:
        if col not in controllable_cols:
            s_fix = base_pool[col]
            if pd.api.types.is_numeric_dtype(s_fix):
                finite_vals = pd.to_numeric(s_fix, errors="coerce").dropna().to_numpy(dtype=float)
                if finite_vals.size == 0:
                    raise RuntimeError(
                        f"Inverse candidate generation failed: no finite values available for '{col}'."
                    )
                out[col] = rng.choice(finite_vals, size=n_samples, replace=True)
            else:
                choices = s_fix.dropna().astype(str).unique()
                if choices.size == 0:
                    raise RuntimeError(
                        f"Inverse candidate generation failed: no categorical values available for '{col}'."
                    )
                out[col] = rng.choice(choices, size=n_samples, replace=True)
            continue

        if col in num_cols:
            s = pd.to_numeric(base_pool[col], errors="coerce")
            q05 = float(s.quantile(0.05))
            q95 = float(s.quantile(0.95))
            if np.isfinite(q05) and np.isfinite(q95) and q95 > q05:
                vals = rng.uniform(q05, q95, size=n_samples)
            else:
                raise RuntimeError(
                    f"Inverse candidate generation failed: invalid numeric range for controllable '{col}'."
                )
            if _is_integer_like_series(s):
                vals = np.round(vals)
            out[col] = vals
        else:
            s = base_pool[col].dropna().astype(str)
            vc = s.value_counts(normalize=True)
            choices = vc.index.to_list()
            probs = vc.values
            if len(choices) == 0:
                raise RuntimeError(
                    f"Inverse candidate generation failed: no categorical values for controllable '{col}'."
                )
            out[col] = rng.choice(choices, size=n_samples, replace=True, p=probs)

    return out[X_ref.columns]


def run_inverse_design(
    best_model,
    X_train: pd.DataFrame,
    X_condition: pd.DataFrame,
    y_train_orig: pd.Series,
    target_rates: List[float],
    pollutant_col: str,
    pollutant_values: List[str],
    controllable_cols: List[str],
    n_samples: int,
    topk: int,
    confidence_level: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if X_condition.empty:
        raise RuntimeError("Inverse conditioning frame is empty.")
    if not X_train.index.isin(X_condition.index).all():
        raise RuntimeError("Inverse conditioning frame does not fully cover selected-train index.")

    resolved_pollutant_col = resolve_column_name(X_condition.columns.tolist(), [pollutant_col])
    if resolved_pollutant_col is None:
        raise RuntimeError(
            f"Pollutant column '{pollutant_col}' is not available in raw training features. "
            "Use --inverse-pollutant-col with an available raw feature."
        )
    pollutant_col = resolved_pollutant_col

    if not pollutant_values:
        raise RuntimeError(
            "Provide pollutant inputs explicitly using --inverse-pollutants or --inverse-pollutants-file."
        )
    pollutant_values = [str(v) for v in pollutant_values]
    controllable_cols = _resolve_controllable_columns(X_train, pollutant_col, controllable_cols)
    pollutant_series = X_condition.loc[X_train.index, pollutant_col]
    if pollutant_series.notna().sum() == 0:
        raise RuntimeError("Resolved pollutant column has no usable values in training rows.")
    pollutant_series = pollutant_series.astype(str)

    conf = float(confidence_level)
    if not (0.5 <= conf < 1.0):
        raise RuntimeError("inverse-confidence must be in [0.5, 1.0).")
    q_lo = (1.0 - conf) / 2.0
    q_hi = 1.0 - q_lo

    rows: List[pd.DataFrame] = []
    ranges_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    med = X_train[controllable_cols].median(numeric_only=True)
    iqr = (X_train[controllable_cols].quantile(0.75) - X_train[controllable_cols].quantile(0.25)).replace(0.0, 1.0)

    for p_idx, pol in enumerate(pollutant_values):
        pol_mask = pollutant_series == str(pol)
        if int(pol_mask.sum()) <= 5:
            raise RuntimeError(
                f"Not enough rows for pollutant '{pol}' in training data (need > 5)."
            )
        candidates = _sample_inverse_candidates(
            X_ref=X_train,
            n_samples=n_samples,
            seed=seed + 97 * (p_idx + 1),
            pool_mask=pol_mask,
            controllable_cols=controllable_cols,
        )
        if pollutant_col in candidates.columns:
            candidates[pollutant_col] = str(pol)

        preds_trans = best_model.predict(candidates)
        preds_orig = target_inverse_transform(preds_trans)

        z = (candidates[controllable_cols] - med).abs().div(iqr, axis=1)
        plausibility = np.exp(-np.clip(z.mean(axis=1).to_numpy(dtype=float), 0.0, 8.0))

        for target in target_rates:
            tmp = candidates.copy()
            tmp["pollutant_input"] = pol
            tmp["target_removal_rate"] = float(target)
            tmp["predicted_removal_rate"] = preds_orig
            tmp["abs_error_to_target"] = np.abs(tmp["predicted_removal_rate"] - float(target))
            tmp["plausibility_score"] = plausibility
            tmp = tmp.sort_values(
                by=["abs_error_to_target", "plausibility_score"],
                ascending=[True, False],
            ).head(max(1, int(topk))).copy()
            tmp["rank"] = np.arange(1, len(tmp) + 1)
            rows.append(tmp)

            b = tmp.iloc[0]
            best_rows.append(
                {
                    "pollutant_input": pol,
                    "target_removal_rate": float(target),
                    "predicted_removal_rate": float(b["predicted_removal_rate"]),
                    "abs_error_to_target": float(b["abs_error_to_target"]),
                    "plausibility_score": float(b["plausibility_score"]),
                }
            )

            pred_q = tmp["predicted_removal_rate"].quantile([q_lo, 0.5, q_hi])
            for col in controllable_cols:
                s_col = pd.to_numeric(tmp[col], errors="coerce")
                if s_col.notna().sum() == 0:
                    continue
                col_q = s_col.quantile([q_lo, 0.5, q_hi])
                ranges_rows.append(
                    {
                        "pollutant_input": pol,
                        "target_removal_rate": float(target),
                        "parameter": col,
                        "value_low": float(col_q.loc[q_lo]),
                        "value_median": float(col_q.loc[0.5]),
                        "value_high": float(col_q.loc[q_hi]),
                        "confidence_level": conf,
                        "achievable_removal_low": float(pred_q.loc[q_lo]),
                        "achievable_removal_median": float(pred_q.loc[0.5]),
                        "achievable_removal_high": float(pred_q.loc[q_hi]),
                    }
                )

    inverse_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    col_order = [
        "pollutant_input",
        "target_removal_rate",
        "rank",
        "predicted_removal_rate",
        "abs_error_to_target",
        "plausibility_score",
        pollutant_col,
    ] + controllable_cols
    if not inverse_table.empty:
        keep_cols = [c for c in col_order if c in inverse_table.columns]
        inverse_table = inverse_table[keep_cols]

    ranges_table = pd.DataFrame(ranges_rows)
    summary: Dict[str, Any] = {
        "enabled": True,
        "pollutant_column": pollutant_col,
        "pollutants_requested": pollutant_values,
        "controllable_columns": controllable_cols,
        "targets": [float(t) for t in target_rates],
        "n_candidates_sampled_per_pollutant": int(n_samples),
        "topk_per_target": int(topk),
        "confidence_level": conf,
        "best_recommendations": best_rows,
        "training_target_range": [
            float(np.nanmin(y_train_orig.to_numpy(dtype=float))),
            float(np.nanmax(y_train_orig.to_numpy(dtype=float))),
        ],
    }
    return inverse_table, ranges_table, summary


def build_low_failure_report(
    best_model,
    X_low: pd.DataFrame,
    y_low: pd.Series,
    X_high_train: pd.DataFrame,
    high_threshold: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if len(X_low) == 0:
        raise RuntimeError("No low-performance rows available for low-zone diagnostics.")

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


def validate_pipeline_integrity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    y_train_orig: pd.Series,
    y_test_orig: pd.Series,
    y_train_trans: pd.Series,
) -> None:
    if X_train.empty:
        raise RuntimeError("Integrity check failed: X_train is empty.")
    if X_train.shape[1] == 0:
        raise RuntimeError("Integrity check failed: no features remain after preprocessing.")

    for name, frame in [("X_train", X_train), ("X_test", X_test), ("X_full", X_full)]:
        if frame.columns.duplicated().any():
            dup = frame.columns[frame.columns.duplicated()].tolist()
            raise RuntimeError(f"Integrity check failed: duplicate columns in {name}: {dup}")

    expected_cols = X_train.columns.tolist()
    if X_test.columns.tolist() != expected_cols:
        raise RuntimeError("Integrity check failed: X_test columns/order do not match X_train.")
    if X_full.columns.tolist() != expected_cols:
        raise RuntimeError("Integrity check failed: X_full columns/order do not match X_train.")

    if not X_train.index.equals(y_train_orig.index):
        raise RuntimeError("Integrity check failed: X_train index does not match y_train_orig index.")
    if not X_test.index.equals(y_test_orig.index):
        raise RuntimeError("Integrity check failed: X_test index does not match y_test_orig index.")
    if not y_train_trans.index.equals(y_train_orig.index):
        raise RuntimeError("Integrity check failed: y_train_trans index does not match y_train_orig index.")

    overlap = X_train.index.intersection(X_test.index)
    if len(overlap) > 0:
        raise RuntimeError(
            f"Integrity check failed: train/test index overlap detected ({len(overlap)} rows)."
        )


class KMeansMoERegressor:
    def __init__(
        self,
        gate_cols: List[str],
        gate_imputer: KNNImputer,
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
    knn_k: int,
    n_clusters: int = 2,
    min_cluster_rows: int = 25,
) -> Tuple[KMeansMoERegressor, Dict[str, Any]]:
    gate_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(gate_cols) < 2:
        raise RuntimeError("MoE requires at least 2 numeric columns for gating.")
    if X_train.shape[0] < n_clusters * min_cluster_rows:
        raise RuntimeError(
            f"MoE infeasible: train rows={X_train.shape[0]} < "
            f"n_clusters*min_cluster_rows={n_clusters * min_cluster_rows}. "
            f"Adjust --moe-clusters or --moe-min-cluster-rows."
        )

    gate_imputer = KNNImputer(n_neighbors=max(1, int(knn_k)), weights="distance")
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
    args = validate_and_normalize_args(parse_args())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = auto_pick_input_file(args.input_file)
    print(f"Using input file: {input_path}")

    df, target_col, used_header = load_dataframe(
        path=input_path,
        target_hint=args.target,
        target_aliases=list(args.target_aliases),
        header_idx=int(args.header),
    )
    if int(args.header) < 0:
        print(f"Auto header detection selected header={used_header}.")
    else:
        print(f"Using header={used_header}.")
    print(f"Resolved target column: {target_col}")

    df = add_rdkit_descriptors(df)

    masking_report = {}

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    if args.target_mask_range:
        df, masked_idx = apply_target_mask(df, target_col, tuple(args.target_mask_range))
        masking_report['target_mask'] = {
            'range': args.target_mask_range,
            'masked_rows': len(masked_idx)
        }
    df = df.dropna(subset=[target_col]).copy()

    drop_like = ["number", "index", "unnamed: 0"]
    leak_cols = [c for c in df.columns if c.lower() in drop_like]

    X_raw = df.drop(columns=[target_col] + leak_cols)
    y = df[target_col]
    high_mask_all = y >= args.high_threshold
    low_mask_all = ~high_mask_all
    if high_mask_all.sum() < 2 or low_mask_all.sum() < 2:
        raise RuntimeError(
            "Need at least 2 high-threshold and 2 low-threshold rows for a valid split."
        )

    X_train_raw, X_test_raw, y_train_orig, y_test_orig = train_test_split(
        X_raw,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=high_mask_all if high_mask_all.sum() > 10 else None,
    )

    [X_train_pre, X_test_pre, X_full_pre], cat_report = drop_high_cardinality_categoricals(
        X_train=X_train_raw,
        frames_to_filter=[X_train_raw, X_test_raw, X_raw],
        max_cardinality=args.max_cat_cardinality,
        unique_ratio_threshold=args.cat_unique_ratio_threshold,
    )
    if cat_report["dropped_columns"]:
        print(f"Dropped {len(cat_report['dropped_columns'])} high-cardinality/id-like categorical columns.")

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
            knn_k=args.knn_k,
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
    validate_pipeline_integrity(
        X_train=X_train,
        X_test=X_test,
        X_full=X_full,
        y_train_orig=y_train_orig,
        y_test_orig=y_test_orig,
        y_train_trans=y_train_trans,
    )

    high_train_count = int((y_train_orig >= args.high_threshold).sum())
    low_train_count = int((y_train_orig < args.high_threshold).sum())
    if high_train_count < 10:
        print(f"Warning: Only {high_train_count} high-removal rows in TRAIN split. High-zone metrics may be noisy.")

    if high_train_count > 0:
        raw_ratio = low_train_count / high_train_count
        weight_ratio = min(max(raw_ratio * 1.5, 10.0), 1000.0)
    else:
        weight_ratio = 10.0

    if args.dry_run:
        print("\n--- DRY RUN: DATA PIPELINE CHECK ---")
        print(f"Dataframe initial shape: {df.shape}")
        print(f"Target column: {target_col}")
        print(f"Rows after target cleanup: {len(df)}")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train_orig shape: {y_train_orig.shape}")
        print(f"y_test_orig shape: {y_test_orig.shape}")
        print(f"High threshold (orig): {args.high_threshold}")
        print(f"High threshold (trans): {high_threshold_trans:.4f}")
        print(f"Weight ratio for high samples: {weight_ratio:.2f}")
        print(f"Categorical drop count: {len(cat_report['dropped_columns'])}")
        if not args.no_feature_drop:
            print(f"Feature selection dropped: {feature_selection_report.get('total_dropped', 0)}")
        print("First 5 features:", X_train.columns.tolist()[:5])
        print("All data prep steps completed successfully! Exiting (--dry-run).")
        sys.exit(0)

    preprocessor = build_preprocessor(X_train, knn_k=args.knn_k)

    model_names = ["xgboost", "extra_trees", "random_forest", "hist_gb"]
    per_model_trials = max(8, args.trials // len(model_names))

    fitted_models = {}
    best_params_report = {}
    metrics_high = []

    print(
        f"\nTraining objective: maximize performance on HIGH removal rows (>= {args.high_threshold}) with weighted MAE in transformed space."
    )
    print(f"Rows: total={len(df)}, train_high={high_train_count}, train_low={low_train_count}")

    print(f"Dynamic High-Sample Weight: {weight_ratio:.2f}")
    print("\nTuning base models with Optuna (Weighted MAE on Transformed Target)...")

    test_high_mask = y_test_orig >= args.high_threshold
    X_test_high = X_test[test_high_mask]
    y_test_high_orig = y_test_orig[test_high_mask]
    if len(X_test_high) == 0:
        raise RuntimeError(
            "No high-performance rows in test split. Cannot compute high-zone metrics."
        )

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
        preds_trans = model.predict(X_test)
        preds_orig = target_inverse_transform(preds_trans)
        global_metrics = evaluate(name, y_test_orig, preds_orig)
        preds_high_trans = model.predict(X_test_high)
        preds_high_orig = target_inverse_transform(preds_high_trans)
        h_metrics = evaluate_high_focus(name, y_test_high_orig, preds_high_orig, args.high_threshold)
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
    stack_preds_trans = final_stack_pipe.predict(X_test)
    stack_preds_orig = target_inverse_transform(stack_preds_trans)
    global_metrics_stack = evaluate("stacking_top3", y_test_orig, stack_preds_orig)
    stack_preds_high_trans = final_stack_pipe.predict(X_test_high)
    stack_preds_high_orig = target_inverse_transform(stack_preds_high_trans)
    stack_metrics = evaluate_high_focus("stacking_top3", y_test_high_orig, stack_preds_high_orig, args.high_threshold)
    stack_metrics["r2_global"] = global_metrics_stack["r2"]
    stack_metrics["rmse_global"] = global_metrics_stack["rmse"]
    metrics_high.append(stack_metrics)
    best_params_report["stacking_top3"] = {"top3": top3}

    moe_report: Dict[str, Any] = {"enabled": False, "reason": "Disabled by user"}
    moe_model: Optional[KMeansMoERegressor] = None
    if not args.disable_moe:
        print(
            f"Building MoE with n_clusters={args.moe_clusters}, "
            f"min_cluster_rows={args.moe_min_cluster_rows}..."
        )
        moe_model, moe_report = build_kmeans_moe(
            expert_template=final_stack_pipe,
            X_train=X_train,
            y_train_trans=y_train_trans,
            high_threshold_trans=high_threshold_trans,
            weight_ratio=weight_ratio,
            seed=args.seed,
            knn_k=args.knn_k,
            n_clusters=args.moe_clusters,
            min_cluster_rows=args.moe_min_cluster_rows,
        )
        moe_preds_trans = moe_model.predict(X_test)
        moe_preds_orig = target_inverse_transform(moe_preds_trans)
        moe_global = evaluate("moe_kmeans2", y_test_orig, moe_preds_orig)

        moe_preds_high_trans = moe_model.predict(X_test_high)
        moe_preds_high_orig = target_inverse_transform(moe_preds_high_trans)
        moe_high = evaluate_high_focus("moe_kmeans2", y_test_high_orig, moe_preds_high_orig, args.high_threshold)
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

    figures_dir = outdir / "figures"
    data_dir = outdir / "data"
    reports_dir = outdir / "reports"
    for p in [figures_dir, data_dir, reports_dir]:
        p.mkdir(parents=True, exist_ok=True)

    preds_test_trans = best_model.predict(X_test)
    preds_test_orig = target_inverse_transform(preds_test_trans)
    test_predictions = pd.DataFrame(
        {
            "row_id": X_test.index.map(str),
            "actual": y_test_orig.loc[X_test.index].astype(float).values,
            "predicted": preds_test_orig,
        },
        index=X_test.index,
    )
    test_predictions["residual"] = test_predictions["predicted"] - test_predictions["actual"]
    test_predictions["abs_error"] = test_predictions["residual"].abs()
    test_predictions["zone"] = np.where(
        test_predictions["actual"] >= args.high_threshold, "high", "low"
    )

    if not (test_predictions["zone"] == "high").any():
        raise RuntimeError("No high-zone rows available for visualization.")

    sns.set_theme(style="whitegrid", context="notebook")
    zone_palette = {"high": "#0B7285", "low": "#C92A2A"}

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax0 = axes[0, 0]
    sns.scatterplot(
        data=test_predictions,
        x="actual",
        y="predicted",
        hue="zone",
        palette=zone_palette,
        s=70,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.3,
        ax=ax0,
    )
    lo = min(float(test_predictions["actual"].min()), float(test_predictions["predicted"].min()))
    hi = max(float(test_predictions["actual"].max()), float(test_predictions["predicted"].max()))
    ax0.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, color="#1F2937", label="Ideal")
    ax0.axvline(args.high_threshold, linestyle=":", linewidth=1.2, color="#6B7280")
    ax0.axhline(args.high_threshold, linestyle=":", linewidth=1.2, color="#6B7280")
    ax0.set_title(f"Predicted vs Actual (Test) | Winner: {winner}")
    ax0.set_xlabel("Actual Removal Rate (%)")
    ax0.set_ylabel("Predicted Removal Rate (%)")
    ax0.grid(alpha=0.2)
    ax0.legend(loc="upper left", fontsize=9)

    ax1 = axes[0, 1]
    sns.histplot(
        data=test_predictions,
        x="residual",
        hue="zone",
        bins=24,
        stat="density",
        common_norm=False,
        element="step",
        palette=zone_palette,
        ax=ax1,
    )
    ax1.axvline(0, linestyle="--", linewidth=1.4, color="#1F2937")
    ax1.set_title("Residual Distribution (Test)")
    ax1.set_xlabel("Residual (Predicted - Actual)")
    ax1.set_ylabel("Density")
    ax1.grid(alpha=0.2)

    ax2 = axes[1, 0]
    for _, row in final_metrics.iterrows():
        marker_color = "#2B8A3E" if row["model"] == winner else "#1C7ED6"
        ax2.scatter(
            row["rmse_high"],
            row["r2_high"],
            s=140,
            color=marker_color,
            edgecolor="black",
            linewidth=0.4,
        )
        ax2.text(
            row["rmse_high"] + 0.02,
            row["r2_high"],
            str(row["model"]),
            fontsize=9,
            va="center",
        )
    ax2.axhline(0, linestyle="--", linewidth=1.0, color="#9CA3AF")
    ax2.set_title("Model Frontier (High-Zone Metrics)")
    ax2.set_xlabel("RMSE (High Zone)")
    ax2.set_ylabel("R2 (High Zone)")
    ax2.grid(alpha=0.2)

    ax3 = axes[1, 1]
    worst = test_predictions.nlargest(min(12, len(test_predictions)), "abs_error").sort_values("abs_error")
    ax3.barh(
        worst["row_id"],
        worst["abs_error"],
        color="#F08C00",
        edgecolor="black",
        linewidth=0.4,
    )
    ax3.set_title("Largest Absolute Errors (Test)")
    ax3.set_xlabel("Absolute Error")
    ax3.set_ylabel("Row ID")
    ax3.grid(axis="x", alpha=0.2)

    plt.suptitle("HeatDraft Evaluation Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    dashboard_path = figures_dir / "performance_dashboard.png"
    plt.savefig(dashboard_path, dpi=170, bbox_inches="tight")
    plt.close()

    n_bins = min(8, int(test_predictions["actual"].nunique()))
    if n_bins >= 2:
        calib_df = test_predictions.copy()
        calib_df["actual_bin"] = pd.qcut(calib_df["actual"], q=n_bins, duplicates="drop")
        calibration = (
            calib_df.groupby("actual_bin", observed=False)
            .agg(actual_mean=("actual", "mean"), predicted_mean=("predicted", "mean"), count=("actual", "size"))
            .reset_index(drop=True)
        )
    else:
        calibration = pd.DataFrame(
            {
                "actual_mean": [float(test_predictions["actual"].mean())],
                "predicted_mean": [float(test_predictions["predicted"].mean())],
                "count": [int(len(test_predictions))],
            }
        )

    fig_cal, ax_cal = plt.subplots(figsize=(8.5, 7))
    ax_cal.plot(
        calibration["actual_mean"],
        calibration["predicted_mean"],
        marker="o",
        linewidth=2.0,
        color="#1971C2",
    )
    cal_lo = min(float(test_predictions["actual"].min()), float(test_predictions["predicted"].min()))
    cal_hi = max(float(test_predictions["actual"].max()), float(test_predictions["predicted"].max()))
    ax_cal.plot([cal_lo, cal_hi], [cal_lo, cal_hi], linestyle="--", linewidth=1.5, color="#1F2937")
    for _, r in calibration.iterrows():
        ax_cal.text(float(r["actual_mean"]) + 0.1, float(r["predicted_mean"]), f"n={int(r['count'])}", fontsize=8)
    ax_cal.set_title("Calibration by Actual-Value Quantiles (Test)")
    ax_cal.set_xlabel("Actual Mean")
    ax_cal.set_ylabel("Predicted Mean")
    ax_cal.grid(alpha=0.25)
    plt.tight_layout()
    calibration_path = figures_dir / "calibration_curve.png"
    plt.savefig(calibration_path, dpi=170, bbox_inches="tight")
    plt.close()

    heatmap_path = None
    if not args.no_feature_drop and X_train_pre.shape[1] <= 80:
        num_before = X_train_pre.select_dtypes(include=[np.number]).columns.tolist()
        num_after = X_train.select_dtypes(include=[np.number]).columns.tolist()
        fig2, axes2 = plt.subplots(1, 2, figsize=(max(12, len(num_before) // 2), max(9, len(num_before) // 2)))
        for ax_corr, cols, title in [
            (axes2[0], num_before, f"Before Dropping ({len(num_before)} features)"),
            (axes2[1], num_after, f"After Dropping ({len(num_after)} features)"),
        ]:
            if len(cols) > 0:
                src = X_train_pre if "Before" in title else X_train
                data = knn_impute_numeric_frame(
                    src[cols],
                    n_neighbors=args.knn_k,
                    context=f"heatmap {title}",
                )
                corr_mat = data.corr(method="pearson")
                mask_tri = np.triu(np.ones_like(corr_mat, dtype=bool))
                sns.heatmap(
                    corr_mat,
                    mask=mask_tri,
                    ax=ax_corr,
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.3,
                    annot=len(cols) <= 20,
                    fmt=".1f",
                    annot_kws={"size": 7},
                )
                ax_corr.set_title(title, fontsize=11, fontweight="bold")
                ax_corr.tick_params(axis="x", rotation=45, labelsize=7)
                ax_corr.tick_params(axis="y", rotation=0, labelsize=7)
        plt.suptitle("Pearson Correlation: Before vs After Feature Dropping", fontsize=13, fontweight="bold")
        plt.tight_layout()
        heatmap_path = figures_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=160, bbox_inches="tight")
        plt.close()

    test_low_mask = y_test_orig < args.high_threshold
    X_low = X_test.loc[test_low_mask]
    y_low = y_test_orig.loc[test_low_mask]
    X_high_train = X_train.loc[y_train_orig >= args.high_threshold]

    low_summary, low_feature_gaps = build_low_failure_report(
        best_model=best_model,
        X_low=X_low,
        y_low=y_low,
        X_high_train=X_high_train,
        high_threshold=args.high_threshold,
    )
    low_summary_path = reports_dir / "low_zone_diagnostics.json"
    low_summary_path.write_text(json.dumps(low_summary, indent=2), encoding="utf-8")
    low_gap_path = data_dir / "low_zone_feature_gaps.csv"
    gap_plot_path = None
    if not low_feature_gaps.empty:
        low_feature_gaps.to_csv(low_gap_path, index=False)
        print("\nTop feature gaps (low median vs high-train median):")
        print(low_feature_gaps.to_string(index=False))

        sorted_gaps = low_feature_gaps.sort_values("abs_gap", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(sorted_gaps))
        fig_gap, ax_gap = plt.subplots(figsize=(10, 8))
        ax_gap.hlines(
            y=y_pos,
            xmin=sorted_gaps["high_train_median"],
            xmax=sorted_gaps["low_median"],
            color="#ADB5BD",
            linewidth=2.0,
        )
        ax_gap.scatter(
            sorted_gaps["high_train_median"],
            y_pos,
            color="#2B8A3E",
            edgecolor="black",
            linewidth=0.3,
            s=80,
            label="High Zone Median",
        )
        ax_gap.scatter(
            sorted_gaps["low_median"],
            y_pos,
            color="#C92A2A",
            edgecolor="black",
            linewidth=0.3,
            s=80,
            label="Low Zone Median",
        )
        ax_gap.set_yticks(y_pos)
        ax_gap.set_yticklabels(sorted_gaps["feature"])
        ax_gap.set_xlabel("Median Feature Value")
        ax_gap.set_ylabel("Feature")
        ax_gap.set_title("Low-vs-High Feature Gap (Dumbbell View)")
        ax_gap.legend(loc="lower right")
        ax_gap.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        gap_plot_path = figures_dir / "feature_gaps.png"
        plt.savefig(gap_plot_path, dpi=170, bbox_inches="tight")
        plt.close()
    else:
        pd.DataFrame(columns=["feature", "high_train_median", "low_median", "abs_gap"]).to_csv(low_gap_path, index=False)

    inverse_design_path = None
    inverse_ranges_path = None
    inverse_summary_path = None
    inverse_plot_path = None
    inverse_summary: Dict[str, Any] = {"enabled": False, "reason": "Disabled by user"}
    if not args.disable_inverse:
        targets = [float(t) for t in args.inverse_targets]
        if any((t < 0.0 or t > 100.0) for t in targets):
            raise RuntimeError("All inverse targets must be within [0, 100].")
        pollutant_values = parse_inverse_pollutant_values(
            cli_values=[str(v) for v in args.inverse_pollutants],
            file_path=str(args.inverse_pollutants_file),
        )

        inverse_table, inverse_ranges, inverse_summary = run_inverse_design(
            best_model=best_model,
            X_train=X_train,
            X_condition=X_train_raw,
            y_train_orig=y_train_orig,
            target_rates=targets,
            pollutant_col=str(args.inverse_pollutant_col),
            pollutant_values=pollutant_values,
            controllable_cols=[str(c) for c in args.inverse_controllable_cols],
            n_samples=max(2000, int(args.inverse_samples)),
            topk=max(1, int(args.inverse_topk)),
            confidence_level=float(args.inverse_confidence),
            seed=args.seed + 313,
        )
        inverse_design_path = data_dir / "inverse_design_recommendations.csv"
        inverse_table.to_csv(inverse_design_path, index=False)
        inverse_ranges_path = data_dir / "inverse_parameter_ranges.csv"
        inverse_ranges.to_csv(inverse_ranges_path, index=False)

        inverse_summary_path = reports_dir / "inverse_design_summary.json"
        inverse_summary_path.write_text(json.dumps(inverse_summary, indent=2), encoding="utf-8")

        fig_inv, ax_inv = plt.subplots(figsize=(10, 7))
        pollutant_vals = inverse_table["pollutant_input"].astype(str).unique().tolist()
        cmap = sns.color_palette("viridis", n_colors=max(3, len(pollutant_vals)))
        for i, pol in enumerate(pollutant_vals):
            s = inverse_table[inverse_table["pollutant_input"].astype(str) == str(pol)]
            if s.empty:
                continue
            ax_inv.scatter(
                s["predicted_removal_rate"],
                s["plausibility_score"],
                s=42 + 10 * (s["rank"].max() - s["rank"] + 1),
                alpha=0.8,
                color=cmap[i],
                label=f"{pol}",
                edgecolor="black",
                linewidth=0.3,
            )
            best = s.nsmallest(1, "abs_error_to_target").iloc[0]
            ax_inv.text(
                float(best["predicted_removal_rate"]) + 0.05,
                float(best["plausibility_score"]),
                f"best t={best['target_removal_rate']:.1f}%",
                fontsize=8,
            )
        ax_inv.set_title("Inverse Design Candidates by Pollutant")
        ax_inv.set_xlabel("Predicted Removal Rate (%)")
        ax_inv.set_ylabel("Plausibility Score (higher is closer to training manifold)")
        ax_inv.grid(alpha=0.25)
        ax_inv.legend(title="Pollutant", loc="best", fontsize=9)
        plt.tight_layout()
        inverse_plot_path = figures_dir / "inverse_design_candidates.png"
        plt.savefig(inverse_plot_path, dpi=170, bbox_inches="tight")
        plt.close()

    predictions_path = data_dir / "test_predictions.csv"
    test_predictions.reset_index(drop=True).to_csv(predictions_path, index=False)
    metrics_path = data_dir / "metrics_leaderboard.csv"
    final_metrics.to_csv(metrics_path, index=False)
    calibration_table_path = data_dir / "calibration_table.csv"
    calibration.to_csv(calibration_table_path, index=False)
    selected_features_path = data_dir / "selected_features.csv"
    pd.DataFrame({"feature": selected_columns}).to_csv(selected_features_path, index=False)

    best_params_path = reports_dir / "best_params.json"
    best_params_path.write_text(json.dumps(best_params_report, indent=2, default=str), encoding="utf-8")
    split_summary = {
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_high_rows": int((y_train_orig >= args.high_threshold).sum()),
        "train_low_rows": int((y_train_orig < args.high_threshold).sum()),
        "test_high_rows": int((y_test_orig >= args.high_threshold).sum()),
        "test_low_rows": int((y_test_orig < args.high_threshold).sum()),
        "high_threshold": float(args.high_threshold),
    }
    split_summary_path = reports_dir / "split_summary.json"
    split_summary_path.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    vif_table = feature_selection_report.get("vif_table", pd.DataFrame())
    vif_table_path = None
    if not vif_table.empty:
        vif_table_path = data_dir / "vif_table.csv"
        vif_table.to_csv(vif_table_path, index=False)

    artifacts = {
        "figures": {
            "performance_dashboard": str(dashboard_path),
            "calibration_curve": str(calibration_path),
        },
        "data": {
            "metrics_leaderboard": str(metrics_path),
            "test_predictions": str(predictions_path),
            "selected_features": str(selected_features_path),
            "calibration_table": str(calibration_table_path),
            "low_zone_feature_gaps": str(low_gap_path),
        },
        "reports": {
            "low_zone_diagnostics": str(low_summary_path),
            "best_params": str(best_params_path),
            "split_summary": str(split_summary_path),
        },
    }
    if heatmap_path:
        artifacts["figures"]["correlation_heatmap"] = str(heatmap_path)
    if gap_plot_path:
        artifacts["figures"]["feature_gaps"] = str(gap_plot_path)
    if vif_table_path:
        artifacts["data"]["vif_table"] = str(vif_table_path)
    if inverse_design_path:
        artifacts["data"]["inverse_design_recommendations"] = str(inverse_design_path)
    if inverse_ranges_path:
        artifacts["data"]["inverse_parameter_ranges"] = str(inverse_ranges_path)
    if inverse_plot_path:
        artifacts["figures"]["inverse_design_candidates"] = str(inverse_plot_path)
    if inverse_summary_path:
        artifacts["reports"]["inverse_design_summary"] = str(inverse_summary_path)

    report = {
        "input_file": str(input_path),
        "target": target_col,
        "high_threshold": args.high_threshold,
        "rows_total": int(df.shape[0]),
        "rows_train": int(X_train.shape[0]),
        "rows_test": int(X_test.shape[0]),
        "high_rows": int((y >= args.high_threshold).sum()),
        "low_rows": int((y < args.high_threshold).sum()),
        "high_rows_test": int((y_test_orig >= args.high_threshold).sum()),
        "low_rows_test": int((y_test_orig < args.high_threshold).sum()),
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
        "inverse_design": inverse_summary,
        "metrics": final_metrics.to_dict(orient="records"),
        "low_zone_diagnostics": low_summary,
        "artifacts": artifacts,
    }

    report_path = reports_dir / "model_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nSaved report: {report_path}")
    print(f"Saved dashboard: {dashboard_path}")
    print(f"Saved predictions table: {predictions_path}")
    print(f"Saved metrics table: {metrics_path}")
    if inverse_design_path:
        print(f"Saved inverse recommendations: {inverse_design_path}")
    if inverse_ranges_path:
        print(f"Saved inverse parameter ranges: {inverse_ranges_path}")


if __name__ == "__main__":
    main()
