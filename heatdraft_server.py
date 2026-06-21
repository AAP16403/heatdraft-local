"""
HeatDraft Inverse-Design — Web Dashboard Server

Lightweight Flask backend that loads a trained app_bundle.pkl and exposes
REST endpoints consumed by the interactive single-page dashboard.

Usage:
    python heatdraft_server.py                               # default bundle
    python heatdraft_server.py path/to/app_bundle.pkl        # custom bundle
    python heatdraft_server.py --port 8080                   # custom port
"""
import argparse
import io
import json
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response

# Import core functions from the pipeline — no ML logic duplicated
from heatdraft import (
    build_low_risk_model,
    build_preprocessor,
    load_app_bundle,
    resolve_column_name,
    run_inverse_design,
    target_inverse_transform,
)

if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Globals (populated at startup) ──────────────────────────────────────────
_bundle: Dict[str, Any] = {}
_config_cache: Dict[str, Any] = {}
_last_results: Dict[str, Any] = {}

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ── JSON serialiser that handles numpy / pandas types ───────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = NumpyEncoder


def _safe_float(v) -> float:
    """Convert value to a JSON-safe float (handles NaN/Inf)."""
    f = float(v)
    if np.isnan(f) or np.isinf(f):
        return None
    return round(f, 6)


def _df_to_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of dicts with JSON-safe values."""
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.floating, float)):
                rec[col] = _safe_float(val)
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            else:
                rec[col] = str(val) if pd.notna(val) else None
        records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config", methods=["GET"])
def api_config():
    """Return all metadata the frontend needs to initialise controls."""
    return jsonify(_config_cache)


@app.route("/api/run", methods=["POST"])
def api_run():
    """Execute inverse design with the submitted parameters."""
    global _last_results
    try:
        body = request.get_json(force=True)

        pollutant_col = str(body.get("pollutant_col", _config_cache["defaults"]["pollutant_col"]))
        pollutant_values = [str(v) for v in body.get("pollutant_values", [])]
        target_rates = [float(t) for t in body.get("target_rates", _config_cache["defaults"]["target_rates"])]
        controllable_cols = [str(c) for c in body.get("controllable_cols", [])]
        n_samples = int(body.get("n_samples", 12000))
        topk = int(body.get("topk", 12))
        confidence = float(body.get("confidence", 0.80))
        risk_weight = float(body.get("risk_weight", _config_cache["defaults"]["risk_weight"]))

        # Validate
        if not pollutant_values:
            return jsonify({"error": "No pollutant values selected."}), 400
        if not target_rates:
            return jsonify({"error": "No target removal rates specified."}), 400
        if not controllable_cols:
            return jsonify({"error": "No controllable columns selected."}), 400
        for t in target_rates:
            if t < 0 or t > 100:
                return jsonify({"error": f"Target rate {t} out of [0, 100] range."}), 400
        if not (0.5 <= confidence < 1.0):
            return jsonify({"error": "Confidence must be in [0.5, 1.0)."}), 400

        X_train = _bundle["X_train"]
        X_train_raw = _bundle["X_train_raw"]
        y_train_orig = _bundle["y_train_orig"]
        best_model = _bundle["best_model"]
        high_threshold = float(_bundle["high_threshold"])
        risk_model = _bundle.get("low_risk_model")
        risk_k = int(_bundle.get("inverse_low_risk_k", 25))

        # Build risk model on first call if missing
        if risk_model is None and risk_weight > 0:
            risk_model = build_low_risk_model(
                X_train=X_train,
                y_train_orig=y_train_orig,
                preprocessor=build_preprocessor(X_train, knn_k=risk_k),
                high_threshold=high_threshold,
                n_neighbors=risk_k,
            )
            _bundle["low_risk_model"] = risk_model

        inverse_table, ranges_table, summary = run_inverse_design(
            best_model=best_model,
            X_train=X_train,
            X_condition=X_train_raw,
            y_train_orig=y_train_orig,
            target_rates=target_rates,
            pollutant_col=pollutant_col,
            pollutant_values=pollutant_values,
            controllable_cols=controllable_cols,
            n_samples=max(2000, n_samples),
            topk=max(1, topk),
            confidence_level=confidence,
            risk_model=risk_model,
            risk_weight=risk_weight,
            seed=42,
        )

        recommendations = _df_to_records(inverse_table) if not inverse_table.empty else []
        ranges = _df_to_records(ranges_table) if not ranges_table.empty else []

        # Sanitise summary best_recommendations
        best_recs = []
        for rec in summary.get("best_recommendations", []):
            best_recs.append({
                k: (_safe_float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in rec.items()
            })

        result = {
            "recommendations": recommendations,
            "ranges": ranges,
            "summary": {
                "pollutant_column": summary.get("pollutant_column"),
                "pollutants_requested": summary.get("pollutants_requested"),
                "controllable_columns": summary.get("controllable_columns"),
                "targets": summary.get("targets"),
                "n_candidates_sampled_per_pollutant": summary.get("n_candidates_sampled_per_pollutant"),
                "topk_per_target": summary.get("topk_per_target"),
                "confidence_level": summary.get("confidence_level"),
                "low_risk_enabled": summary.get("low_risk_enabled"),
                "best_recommendations": best_recs,
                "training_target_range": summary.get("training_target_range"),
            },
        }

        _last_results = {
            "inverse_table": inverse_table,
            "ranges_table": ranges_table,
        }

        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/export/csv", methods=["POST"])
def api_export_csv():
    """Export last results as a downloadable CSV."""
    try:
        body = request.get_json(force=True)
        export_type = body.get("type", "recommendations")

        if export_type == "ranges":
            df = _last_results.get("ranges_table", pd.DataFrame())
            filename = "inverse_parameter_ranges.csv"
        else:
            df = _last_results.get("inverse_table", pd.DataFrame())
            filename = "inverse_design_recommendations.csv"

        if df is None or df.empty:
            return jsonify({"error": "No results to export. Run an inverse design first."}), 400

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════

def _build_config_cache(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-compute all metadata the frontend needs."""
    X_train = bundle["X_train"]
    X_train_raw = bundle["X_train_raw"]
    y_train_orig = bundle["y_train_orig"]
    target_col = bundle["target_col"]
    high_threshold = float(bundle["high_threshold"])
    default_targets = bundle.get("default_inverse_targets", [90.0, 95.0, 98.0])
    default_pollutant_col = bundle.get("pollutant_col_default", "TypesOfContaminants")
    default_risk_weight = float(bundle.get("inverse_low_risk_weight", 15.0))

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    all_raw_cols = X_train_raw.columns.tolist()
    cat_cols = [c for c in all_raw_cols if c not in num_cols]

    # Pollutant value counts for each categorical column
    cat_value_counts = {}
    for col in cat_cols:
        vc = X_train_raw[col].astype(str).value_counts().head(30)
        cat_value_counts[col] = [
            {"value": str(v), "count": int(c)} for v, c in vc.items()
        ]

    # Numeric column statistics for display
    num_stats = {}
    for col in num_cols:
        s = pd.to_numeric(X_train[col], errors="coerce")
        num_stats[col] = {
            "min": _safe_float(s.min()),
            "max": _safe_float(s.max()),
            "mean": _safe_float(s.mean()),
            "median": _safe_float(s.median()),
        }

    y_arr = y_train_orig.to_numpy(dtype=float)

    return {
        "target_col": target_col,
        "high_threshold": high_threshold,
        "train_rows": int(X_train.shape[0]),
        "selected_features": int(X_train.shape[1]),
        "target_range": [_safe_float(np.nanmin(y_arr)), _safe_float(np.nanmax(y_arr))],
        "target_mean": _safe_float(np.nanmean(y_arr)),
        "high_rows": int((y_arr >= high_threshold).sum()),
        "low_rows": int((y_arr < high_threshold).sum()),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "cat_value_counts": cat_value_counts,
        "num_stats": num_stats,
        "defaults": {
            "pollutant_col": default_pollutant_col,
            "target_rates": default_targets,
            "risk_weight": default_risk_weight,
            "n_samples": 12000,
            "topk": 12,
            "confidence": 0.80,
        },
    }


def create_app(bundle_path: str) -> Flask:
    """Load the bundle and initialise the Flask app."""
    global _bundle, _config_cache

    p = Path(bundle_path)
    if not p.exists():
        print(f"\n[ERROR] App bundle not found: {p}")
        print("Run the full pipeline first:  python heatdraft.py <data_file>")
        sys.exit(1)

    print(f"Loading app bundle: {p}")
    _bundle = load_app_bundle(p)
    _config_cache = _build_config_cache(_bundle)

    print(f"  Target column       : {_config_cache['target_col']}")
    print(f"  Training rows       : {_config_cache['train_rows']}")
    print(f"  Selected features   : {_config_cache['selected_features']}")
    print(f"  Target range        : {_config_cache['target_range']}")
    print(f"  High/Low split      : {_config_cache['high_rows']} / {_config_cache['low_rows']}")
    print(f"  Numeric columns     : {len(_config_cache['numeric_cols'])}")
    print(f"  Categorical columns : {len(_config_cache['categorical_cols'])}")

    return app


def main():
    parser = argparse.ArgumentParser(
        description="HeatDraft Inverse-Design Web Dashboard"
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        default="outputs/reports/app_bundle.pkl",
        help="Path to app_bundle.pkl (default: outputs/reports/app_bundle.pkl)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to serve on (default: 5000)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open the browser"
    )
    args = parser.parse_args()

    create_app(args.bundle)

    url = f"http://{args.host}:{args.port}"
    print(f"\n{'='*60}")
    print(f"  HeatDraft Dashboard running at: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    if not args.no_browser:
        import threading
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
