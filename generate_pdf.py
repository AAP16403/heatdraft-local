from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, List

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def clean_text(text: str) -> str:
    # Keep output robust across fonts and PDF encoders.
    return str(text).encode("ascii", "ignore").decode("ascii")


class DetailedPDF(FPDF):
    def __init__(self, title: str, subtitle: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_title = clean_text(title)
        self.doc_subtitle = clean_text(subtitle)
        self.alias_nb_pages()
        self.set_margins(15, 16, 15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(24, 37, 64)
        self.cell(0, 5, self.doc_title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 4, self.doc_subtitle, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.set_draw_color(190, 200, 220)
        y = self.get_y() + 1
        self.line(15, y, 195, y)
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

    def section_title(self, index: int, title: str) -> None:
        self.set_fill_color(233, 240, 255)
        self.set_text_color(20, 37, 70)
        self.set_font("Helvetica", "B", 12)
        self.cell(
            0,
            8,
            clean_text(f"{index}. {title}"),
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
            align="L",
            fill=True,
        )
        self.ln(1.5)

    def block_title(self, label: str) -> None:
        self.set_text_color(30, 55, 95)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 5.5, clean_text(label), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")

    def paragraph(self, text: str) -> None:
        self.set_text_color(20, 20, 20)
        self.set_font("Helvetica", "", 10)
        usable_w = self.w - self.l_margin - self.r_margin
        self.set_x(self.l_margin)
        self.multi_cell(usable_w, 5.0, clean_text(text), border=0, align="L")
        self.ln(1)

    def bullets(self, items: List[str]) -> None:
        self.set_text_color(20, 20, 20)
        self.set_font("Helvetica", "", 10)
        usable_w = self.w - self.l_margin - self.r_margin
        for item in items:
            self.set_x(self.l_margin)
            self.multi_cell(usable_w, 4.8, clean_text(f"- {item}"), border=0, align="L")
        self.ln(1)


SECTIONS: List[Dict[str, object]] = [
    {
        "title": "Input Discovery and File Loading",
        "how": [
            "The pipeline resolves the input file from CLI argument first; if missing, it auto-detects a single CSV/XLSX in the working directory.",
            "It reads data using pandas with the configured header row and immediately checks that the target column exists.",
            "If target column is missing or file resolution is ambiguous, execution stops with an explicit error instead of guessing.",
        ],
        "why": [
            "Early, deterministic input checks avoid silent schema drift. Most model failures downstream originate from bad assumptions at load time.",
            "Failing before training starts protects compute budget and ensures every report corresponds to a known schema contract.",
        ],
        "checks": [
            "File existence and uniqueness are enforced.",
            "Target column presence is enforced before any transformation.",
        ],
        "outputs": [
            "A normalized in-memory dataframe with a guaranteed target column.",
        ],
    },
    {
        "title": "Column Normalization and Optional RDKit Enrichment",
        "how": [
            "Column names are normalized to reduce malformed characters and whitespace noise introduced by mixed encodings.",
            "If a SMILES column exists, RDKit descriptors are computed (molecular weight, logP, TPSA, hydrogen bond descriptors, ring count, rotatable bonds, and fraction sp3).",
            "Invalid or empty SMILES values remain NaN for descriptor fields and are handled later by the preprocessing stack.",
        ],
        "why": [
            "Model logic depends on exact column names. Encoding and whitespace variation can break downstream selectors or leak features unexpectedly.",
            "RDKit converts symbolic chemical strings into physically meaningful numeric descriptors that tree models can exploit.",
        ],
        "checks": [
            "SMILES enrichment is optional and data-driven; no SMILES means no descriptor branch.",
            "Descriptor generation does not mutate target semantics.",
        ],
        "outputs": [
            "Expanded feature space when SMILES is present, otherwise baseline schema preserved.",
        ],
    },
    {
        "title": "Target Conditioning and Hard Range Enforcement",
        "how": [
            "Optional target masking can replace a user-defined target interval with NaN to intentionally exclude uncertain ranges.",
            "Target is cast to numeric, and rows with missing target are dropped.",
            "Before logit transform, strict checks enforce no NaN and value range in [0, 100]. Out-of-range targets raise immediately.",
        ],
        "why": [
            "Bounded target semantics are physically meaningful (percentage). Silent clipping hides data quality problems and can corrupt conclusions.",
            "Explicit failure for invalid targets prevents producing polished reports from invalid scientific assumptions.",
        ],
        "checks": [
            "Target mask report includes masked row count.",
            "Transform guard enforces percentage domain assumptions exactly.",
        ],
        "outputs": [
            "Clean target vector compatible with logit/inverse-logit pipeline.",
        ],
    },
    {
        "title": "Leakage Pruning and Split Preconditions",
        "how": [
            "Known leakage-like columns (number, index, unnamed: 0) are removed before splitting.",
            "A strict gate checks that both high-threshold and low-threshold classes have at least two rows before train/test split.",
            "Split is stratified on high/low class when enough high rows exist; otherwise, split proceeds unstratified but still under class-count guard.",
        ],
        "why": [
            "Identifier columns can let trees memorize row identity instead of learning chemistry/process relationships.",
            "Class preconditions avoid impossible evaluation setups where high-zone metrics cannot be computed.",
        ],
        "checks": [
            "RuntimeError when high/low counts are insufficient.",
            "Deterministic random seed controls split reproducibility.",
        ],
        "outputs": [
            "X_train_raw, X_test_raw, y_train_orig, y_test_orig with leakage columns removed.",
        ],
    },
    {
        "title": "Category Filtering and Feature Selection Protocol",
        "how": [
            "High-cardinality and near-unique categoricals are identified from train data only, then dropped consistently from train/test/full frames.",
            "Feature selection pipeline (train only) applies sparse-feature drop, near-zero-variance filter, correlation clustering with MI representative selection, and recursive VIF pruning.",
            "If feature-drop is disabled, a light sparsity keep-rule is still applied to avoid pathological columns.",
        ],
        "why": [
            "Train-only feature selection avoids test leakage while preserving consistent schema propagation.",
            "Correlation and VIF controls reduce collinearity instability and improve interpretability of feature importance patterns.",
        ],
        "checks": [
            "Dropped categorical columns are validated to exist across all filtered frames.",
            "Selected column list is propagated in identical order to train/test/full views.",
        ],
        "outputs": [
            "Final selected feature matrix and detailed feature-selection report object.",
        ],
    },
    {
        "title": "Data Structure Integrity Gate",
        "how": [
            "A dedicated validator enforces: non-empty train matrix, no duplicate columns, exact train-test-full column order equality, and index alignment between feature and target objects.",
            "The validator also enforces zero index overlap between train and test rows.",
        ],
        "why": [
            "Most silent ML bugs are shape/index bugs, not algorithm bugs. Guarding structure immediately after preprocessing catches misuse early.",
            "Column order mismatches are particularly dangerous with transformed pipelines and can invert model semantics without obvious errors.",
        ],
        "checks": [
            "Runtime errors for any mismatch in columns, order, or index alignment.",
            "Hard failure on train/test overlap.",
        ],
        "outputs": [
            "Verified, contract-compliant train/test/full feature tensors for modeling.",
        ],
    },
    {
        "title": "Target Transform, Weighting, and Objective Design",
        "how": [
            "The percentage target is mapped to logit space to respect [0, 100] boundaries after inverse transform.",
            "A dynamic high-zone weight ratio is computed from class imbalance and applied during model fitting and CV scoring.",
            "Objective optimization uses weighted MAE in transformed space with high-zone emphasis.",
        ],
        "why": [
            "The transform stabilizes optimization near hard bounds and prevents impossible >100 or <0 behavior after inverse mapping.",
            "Weighting aligns optimization with practical goals when high-performing rows are minority but mission-critical.",
        ],
        "checks": [
            "High-zone test rows are mandatory; run fails if absent.",
            "Weight ratio and transformed threshold are logged for auditability.",
        ],
        "outputs": [
            "y_train_trans and weighted objective context for all model training stages.",
        ],
    },
    {
        "title": "Model Tuning, Leaderboard, and Ensemble Construction",
        "how": [
            "Optuna tunes xgboost, extra_trees, random_forest, and hist_gb under KFold CV using the weighted transformed objective.",
            "Each tuned model is evaluated on global test and high-zone test subsets.",
            "Top models feed a StackingRegressor with RidgeCV meta-learner; optional MoE clones this template per KMeans cluster gate.",
        ],
        "why": [
            "Model diversity plus meta-learning usually outperforms single-architecture assumptions in heterogeneous process datasets.",
            "KMeans MoE can specialize experts for different process regimes, but remains optional and strict (no silent fallback).",
        ],
        "checks": [
            "Winner selection enforces that a chosen MoE model must exist.",
            "Metrics table keeps both global and high-zone scores for balanced interpretation.",
        ],
        "outputs": [
            "Final metrics leaderboard, winner model object, and best-parameter records.",
        ],
    },
    {
        "title": "Visualization Overhaul and Diagnostic Depth",
        "how": [
            "The test-set visualization suite now includes: predicted-vs-actual scatter with threshold guides, residual distribution by zone, high-zone model frontier, and largest absolute errors.",
            "A dedicated calibration curve and calibration table are computed from actual-value quantile bins.",
            "Low-vs-high feature-gap diagnostics are rendered using a dumbbell view for direct median comparison.",
        ],
        "why": [
            "Single scatter dashboards hide calibration and tail-risk behavior. Multi-view diagnostics make failure modes visible.",
            "Error concentration and calibration matter more for deployment confidence than a single summary metric.",
        ],
        "checks": [
            "Visualization stage requires high-zone rows and low-zone rows in test where relevant diagnostics are computed.",
            "Optional plots are generated only when their data prerequisites are satisfied.",
        ],
        "outputs": [
            "Consistent figure set under the figures directory for fast model review.",
        ],
    },
    {
        "title": "Structured Storage Contract and Reporting",
        "how": [
            "Artifacts are written into three stable subdirectories under outdir: figures, data, and reports.",
            "Data exports include test_predictions.csv, metrics_leaderboard.csv, calibration_table.csv, selected_features.csv, low_zone_feature_gaps.csv, and optional vif_table.csv.",
            "Reports include model_report.json, best_params.json, split_summary.json, and low_zone_diagnostics.json.",
            "model_report.json contains a nested artifact map so downstream tools can resolve all generated paths without hard-coding.",
        ],
        "why": [
            "Flat output directories become unmaintainable as diagnostics grow. Grouping by artifact type improves reproducibility and automation.",
            "Explicit path maps reduce manual hunting and make integration into dashboards or CI straightforward.",
        ],
        "checks": [
            "Directory creation is idempotent.",
            "Artifact references in reports match actual written files.",
        ],
        "outputs": [
            "A machine-readable, automation-friendly output bundle.",
        ],
    },
    {
        "title": "Operational Modes and Failure Semantics",
        "how": [
            "Dry-run mode executes data-prep and integrity checks, then exits before expensive tuning.",
            "Full mode trains, evaluates, visualizes, and exports all artifacts.",
            "The pipeline is intentionally fail-fast: class gates, target bounds, and zone-availability constraints stop execution instead of producing partial misleading reports.",
        ],
        "why": [
            "In scientific/engineering pipelines, partial success can be dangerous if interpreted as valid evidence.",
            "Fail-fast behavior makes failure explicit and actionable, reducing hidden debt.",
        ],
        "checks": [
            "Dry-run confirms schema and preprocessing viability before long compute jobs.",
            "Strict runtime errors enforce contract integrity from ingestion to diagnostics.",
        ],
        "outputs": [
            "Either a complete trusted artifact package or a clear stopping error with cause.",
        ],
    },
]


COMMAND_REFERENCE = [
    "Run full pipeline: python heatdraft.py your_file.xlsx --header 0 --outdir outputs",
    "Run dry-run only: python heatdraft.py your_file.xlsx --header 0 --dry-run",
    "Disable MoE: python heatdraft.py your_file.xlsx --disable-moe",
    "Regenerate flowchart: python generate_flowchart.py",
    "Regenerate PDFs: python generate_pdf.py",
]


ARTIFACT_REFERENCE = [
    "figures/performance_dashboard.png",
    "figures/calibration_curve.png",
    "figures/correlation_heatmap.png (optional)",
    "figures/feature_gaps.png (optional)",
    "data/test_predictions.csv",
    "data/metrics_leaderboard.csv",
    "data/calibration_table.csv",
    "data/selected_features.csv",
    "data/low_zone_feature_gaps.csv",
    "data/vif_table.csv (optional)",
    "reports/model_report.json",
    "reports/best_params.json",
    "reports/split_summary.json",
    "reports/low_zone_diagnostics.json",
]


def render_document(filename: str, doc_title: str, mode: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"Detailed architecture rationale and implementation notes | Generated {stamp}"
    pdf = DetailedPDF(doc_title, subtitle)
    pdf.add_page()

    pdf.block_title("Scope")
    if mode == "why":
        pdf.paragraph(
            "This document explains why each major stage exists, what risk it mitigates, and why strict failure behavior is preferred over silent fallback."
        )
    elif mode == "how":
        pdf.paragraph(
            "This document explains how each stage is implemented, which checks are enforced, and what outputs are produced for downstream analysis."
        )
    else:
        pdf.paragraph(
            "This document combines the WHY and HOW perspectives for the current HeatDraft pipeline, including strict integrity guards, updated visuals, and structured storage."
        )

    section_titles = [str(section["title"]) for section in SECTIONS]
    pdf.block_title("Navigation")
    pdf.bullets([f"{i}. {title}" for i, title in enumerate(section_titles, 1)])

    for i, section in enumerate(SECTIONS, 1):
        pdf.section_title(i, str(section["title"]))
        if mode in {"how", "full"}:
            pdf.block_title("How")
            pdf.bullets([str(x) for x in section["how"]])  # type: ignore[index]
        if mode in {"why", "full"}:
            pdf.block_title("Why")
            pdf.bullets([str(x) for x in section["why"]])  # type: ignore[index]
        if mode == "full":
            pdf.block_title("Integrity / Validation Checks")
            pdf.bullets([str(x) for x in section["checks"]])  # type: ignore[index]
            pdf.block_title("Key Outputs")
            pdf.bullets([str(x) for x in section["outputs"]])  # type: ignore[index]

    pdf.section_title(len(SECTIONS) + 1, "Operational Command Reference")
    pdf.bullets(COMMAND_REFERENCE)
    pdf.section_title(len(SECTIONS) + 2, "Output Artifact Reference")
    pdf.bullets(ARTIFACT_REFERENCE)

    pdf.output(filename)
    print(f"Generated: {filename}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HeatDraft pipeline documentation PDFs."
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also generate separate WHY and HOW PDFs in addition to merged master PDF.",
    )
    parser.add_argument(
        "--master-name",
        default="HeatDraft_Pipeline_Documentation_Merged.pdf",
        help="Output filename for the merged cohesive PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Always generate a single cohesive document first.
    render_document(
        filename=args.master_name,
        doc_title="HeatDraft Pipeline - Cohesive Merged Documentation",
        mode="full",
    )

    # Keep legacy combined filename updated for compatibility with existing references.
    render_document(
        filename="HeatDraft_Pipeline_Theory_Why_And_How.pdf",
        doc_title="HeatDraft Pipeline - WHY + HOW (Detailed)",
        mode="full",
    )
    render_document(
        filename="HeatDraft_Pipeline_Documentation_Detailed.pdf",
        doc_title="HeatDraft Pipeline - Full Documentation (Detailed)",
        mode="full",
    )

    if args.split:
        render_document(
            filename="HeatDraft_Pipeline_Why_Detailed.pdf",
            doc_title="HeatDraft Pipeline - WHY (Detailed)",
            mode="why",
        )
        render_document(
            filename="HeatDraft_Pipeline_How_Detailed.pdf",
            doc_title="HeatDraft Pipeline - HOW (Detailed)",
            mode="how",
        )


if __name__ == "__main__":
    main()
