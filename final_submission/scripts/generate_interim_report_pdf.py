"""Generate interim report PDF with embedded, labeled EDA figures."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "interim_report_revised.pdf"
PLOTS = ROOT / "analysis_outputs" / "task2"
IMG_WIDTH = 15.5 * cm


def _img(path: Path, caption: str, cap_style, body) -> list:
    flow: list = []
    if path.is_file():
        im = Image(str(path), width=IMG_WIDTH, height=IMG_WIDTH * 0.48)
        im.hAlign = "CENTER"
        flow.append(Spacer(1, 0.12 * cm))
        flow.append(im)
    else:
        flow.append(Paragraph(f"<i>[Missing figure: {path.name}]</i>", body))
    flow.append(Spacer(1, 0.08 * cm))
    flow.append(Paragraph(f"<b>{caption}</b>", cap_style))
    return flow


def _table(data: list[list], col_widths: list[float] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_story() -> list:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontSize=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=4,
    )
    sub = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
    )
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor("#1a365d"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=10,
        spaceBefore=6,
        spaceAfter=3,
        textColor=colors.HexColor("#2b6cb0"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12.5,
        alignment=TA_JUSTIFY,
    )
    cap = ParagraphStyle(
        "Cap",
        parent=body,
        fontSize=8.5,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6,
    )

    s: list = []

    # Cover
    s.append(Paragraph("Interim Report", title))
    s.append(Paragraph("Credit Risk Probability Model — Week 4", title))
    s.append(
        Paragraph(
            "Analytics Engineer · Bati Bank × Xente Alternative Data<br/>"
            "Interim Submission (Tasks 1 &amp; 2) · 2 June 2026",
            sub,
        )
    )
    s.append(Spacer(1, 0.3 * cm))

    s.append(Paragraph("Executive Summary", h1))
    s.append(
        Paragraph(
            "This interim deliverable establishes the business and analytical foundation for a "
            "<b>buy-now-pay-later (BNPL)</b> credit scoring product. <b>Task 1</b> documents Basel II "
            "governance requirements, the need for a proxy default target, and interpretability trade-offs. "
            "<b>Task 2</b> explores <b>95,662 transactions</b> from <b>3,742 customers</b> (Nov 2018 – Feb 2019). "
            "EDA reveals severe value skew, category-driven fraud variation, and extreme class imbalance "
            "(fraud ~0.20%). All key findings are supported with <b>labeled visualizations</b> in Section 2. "
            "Section 4 outlines concrete next steps for WoE/IV feature engineering, RFM proxy labeling, "
            "and model evaluation metrics.",
            body,
        )
    )

    # Section 1 — Task 1
    s.append(Paragraph("1. Business Understanding (Task 1)", h1))

    s.append(Paragraph("1.1 Business objective", h2))
    s.append(
        Paragraph(
            "Bati Bank is partnering with Xente to score BNPL applicants using eCommerce transaction behavior. "
            "The model must output a <b>risk probability</b> to inform approvals, credit limits, and loan terms "
            "under regulatory expectations aligned with <b>Basel II</b>.",
            body,
        )
    )

    s.append(Paragraph("1.2 Basel II and interpretability", h2))
    s.append(
        Paragraph(
            "Basel II requires banks to <b>measure, document, and govern</b> credit risk. Practically, this means: "
            "(1) models must be explainable to risk and compliance teams; (2) pipelines must be auditable and "
            "reproducible; (3) performance must be monitored with documented limitations. These requirements favor "
            "transparent scorecard-style models unless higher-capacity models demonstrate material gains with "
            "added explainability tooling.",
            body,
        )
    )

    s.append(Paragraph("1.3 Proxy target necessity and business risks", h2))
    s.append(
        Paragraph(
            "The dataset contains <b>no direct loan default label</b>. A supervised model therefore requires a "
            "<b>proxy target</b> (planned: RFM-based high-risk segment in Task 4). Key business risks include: "
            "<b>label risk</b> (proxy may not equal true default), <b>fairness risk</b> (low-activity customers "
            "penalized), and <b>drift risk</b> (platform/product mix changes over time).",
            body,
        )
    )

    s.append(Paragraph("1.4 Model trade-offs", h2))
    s.append(
        _table(
            [
                ["Approach", "Strengths", "Regulated-context risk"],
                [
                    "Logistic Regression + WoE",
                    "Transparent coefficients; scorecard-friendly; easy to audit",
                    "May underfit nonlinear patterns",
                ],
                [
                    "Gradient Boosting (XGBoost/LightGBM)",
                    "Strong predictive power on tabular data",
                    "Harder to explain; higher governance burden",
                ],
            ],
            [4.2 * cm, 5.5 * cm, 5.5 * cm],
        )
    )
    s.append(Spacer(1, 0.1 * cm))
    s.append(
        Paragraph(
            "<b>Recommendation:</b> Build an interpretable WoE + logistic baseline first; promote complex models "
            "only if they materially outperform on PR-AUC/F1 with SHAP-based documentation.",
            body,
        )
    )

    # Section 2 — EDA with figures
    s.append(PageBreak())
    s.append(Paragraph("2. EDA Findings (Task 2)", h1))

    s.append(Paragraph("2.1 Data scope and quality", h2))
    s.append(
        _table(
            [
                ["Attribute", "Finding"],
                ["Records", "95,662 transactions"],
                ["Customers", "3,742 unique CustomerId"],
                ["Features", "16 columns (IDs, product, channel, amount, time, fraud flag)"],
                ["Period", "2018-11-15 to 2019-02-13"],
                ["Missing values", "0% in core fields"],
                ["Currency / country", "UGX; country code 256 (Uganda)"],
            ],
            [4 * cm, 11 * cm],
        )
    )

    s.append(Paragraph("2.2 Summary statistics", h2))
    s.append(
        _table(
            [
                ["Statistic", "Value (UGX)", "Amount (UGX)"],
                ["Mean", "9,901", "6,718"],
                ["Median", "1,000", "1,000"],
                ["Std dev", "123,122", "123,307"],
                ["Maximum", "9,880,000", "9,880,000"],
            ],
            [4 * cm, 5.5 * cm, 5.7 * cm],
        )
    )
    s.append(
        Paragraph(
            "Typical transactions are small (median 1,000 UGX), but rare extreme values inflate means and "
            "standard deviations — a strong signal for winsorization or log transforms in feature engineering.",
            body,
        )
    )

    figures = [
        (
            "Figure 1 — Transaction value distribution (99th percentile cap)",
            "value_distribution.png",
            "The distribution is strongly right-skewed. Most transactions cluster at low values, while a "
            "small number of extreme transactions drive the mean upward. Customer-level aggregates should use "
            "robust statistics (median, IQR) or log-scaled features.",
        ),
        (
            "Figure 2 — Fraud rate by product category (n ≥ 100)",
            "fraud_rate_by_category.png",
            "Fraud rates vary materially by category: transport (8.0%) and utility_bill (0.63%) exceed "
            "airtime (0.04%). ProductCategory will be a high-information feature; minority categories need "
            "careful validation during modeling.",
        ),
        (
            "Figure 3 — Daily transaction volume over time",
            "daily_transaction_volume.png",
            "Daily volume is relatively stable across the ~3-month window, supporting temporal features "
            "(hour, day-of-week, month) and a fixed snapshot date for RFM recency calculation in Task 4.",
        ),
        (
            "Figure 4 — Correlation heatmap (numeric features)",
            "correlation_heatmap.png",
            "Value and Amount are highly correlated by construction. FraudResult shows weak linear correlation "
            "with raw numerics, indicating that nonlinear transforms, aggregates, and categorical encodings are "
            "required rather than raw transaction amounts alone.",
        ),
        (
            "Figure 5 — Outlier detection (box plots: Value and Amount)",
            "outlier_boxplots.png",
            "Both variables exhibit long upper tails and extreme outliers. Negative Amount values represent "
            "credits/refunds and must be handled explicitly when computing customer-level debit totals.",
        ),
    ]
    for cap_text, fname, interp in figures:
        s.extend(_img(PLOTS / fname, cap_text, cap, body))
        s.append(Paragraph(f"<i>Interpretation:</i> {interp}", cap))

    s.append(Paragraph("2.3 Categorical and fraud patterns", h2))
    s.append(
        _table(
            [
                ["ProductCategory", "Count", "Share"],
                ["financial_services", "45,405", "47.5%"],
                ["airtime", "45,027", "47.1%"],
                ["utility_bill", "1,920", "2.0%"],
                ["Other categories", "4,310", "4.5%"],
            ],
            [5 * cm, 3 * cm, 3 * cm],
        )
    )
    s.append(Spacer(1, 0.1 * cm))
    s.append(
        _table(
            [
                ["FraudResult", "Count", "Rate"],
                ["0 (legitimate)", "95,469", "99.80%"],
                ["1 (fraud)", "193", "0.20%"],
            ],
            [4 * cm, 3.5 * cm, 3.5 * cm],
        )
    )

    s.append(Paragraph("2.4 Top 5 EDA insights (modeling implications)", h2))
    insights = [
        "<b>1. Severe class imbalance (~0.2% fraud):</b> Optimize PR-AUC, F1, and recall@k; use class weights or SMOTE.",
        "<b>2. Heavy skew/outliers:</b> Winsorize at 99th percentile; log1p transforms on customer aggregates.",
        "<b>3. Category-driven risk:</b> One-hot or target-encode ProductCategory; monitor performance by segment.",
        "<b>4. Dominant categories:</b> Validate models on utility_bill and transport, not only airtime/financial_services.",
        "<b>5. Customer-level modeling:</b> All features and RFM must be computed per CustomerId (3,742 units).",
    ]
    for item in insights:
        s.append(Paragraph(item, body))
        s.append(Spacer(1, 0.05 * cm))

    # Section 3 — Next steps (detailed)
    s.append(PageBreak())
    s.append(Paragraph("3. Next Steps — Tasks 3 to 6 (Detailed Plan)", h1))

    s.append(Paragraph("3.1 Task 3 — Feature engineering pipeline", h2))
    s.append(
        Paragraph(
            "Build a single <font name='Courier' size='8'>sklearn.pipeline.Pipeline</font> in "
            "<font name='Courier' size='8'>src/data_processing.py</font> with deterministic transforms "
            "(fixed random_state=42):",
            body,
        )
    )
    s.append(
        _table(
            [
                ["Step", "Technique", "Details"],
                [
                    "Aggregates",
                    "Per CustomerId",
                    "sum/mean/count/std of Amount and Value; debit vs credit split",
                ],
                [
                    "Time features",
                    "From TransactionStartTime",
                    "hour, day, month, year; days since last transaction",
                ],
                [
                    "Encoding",
                    "One-hot or label",
                    "ProductCategory, ChannelId, PricingStrategy",
                ],
                [
                    "Missing values",
                    "Imputation",
                    "median (numeric), mode (categorical); document strategy",
                ],
                [
                    "Scaling",
                    "StandardScaler",
                    "zero mean, unit variance for numeric model inputs",
                ],
                [
                    "WoE / IV",
                    "xverse or woe package",
                    "Bin continuous vars; compute WoE per bin; rank features by IV "
                    "(IV &gt; 0.3 strong, 0.1–0.3 medium, &lt; 0.1 weak); apply WoE transform for scorecard",
                ],
            ],
            [2.5 * cm, 3.5 * cm, 9.2 * cm],
        )
    )

    s.append(Paragraph("3.2 Task 4 — Proxy target (RFM + clustering)", h2))
    s.append(
        _table(
            [
                ["Step", "Method", "Specification"],
                ["Recency", "Days since last txn", "Snapshot date = max(TransactionStartTime)"],
                ["Frequency", "Transaction count", "Per CustomerId over observation window"],
                ["Monetary", "Total Value", "Sum of transaction values per customer"],
                ["Clustering", "K-Means (k=3)", "Scale RFM with StandardScaler; random_state=42"],
                ["Label", "is_high_risk", "1 = least engaged cluster (low F, low M); 0 otherwise"],
            ],
            [2.8 * cm, 4 * cm, 8.4 * cm],
        )
    )

    s.append(Paragraph("3.3 Task 5 — Model training and evaluation", h2))
    s.append(
        _table(
            [
                ["Model", "Purpose", "Hyperparameter tuning"],
                ["Logistic Regression (WoE)", "Interpretable baseline", "C, penalty via GridSearchCV"],
                ["Random Forest", "Nonlinear benchmark", "max_depth, n_estimators via RandomizedSearchCV"],
                ["XGBoost / LightGBM", "High-performance challenger", "learning_rate, depth, subsample"],
            ],
            [3.5 * cm, 4.5 * cm, 7.2 * cm],
        )
    )
    s.append(Spacer(1, 0.1 * cm))
    s.append(
        _table(
            [
                ["Metric", "Formula / use", "Why it matters (imbalanced data)"],
                ["Accuracy", "Correct / total", "Misleading when fraud is 0.2%"],
                ["Precision", "TP / (TP+FP)", "Cost of false approvals"],
                ["Recall", "TP / (TP+FN)", "Capture of high-risk customers"],
                ["F1", "Harmonic mean P & R", "Balance precision and recall"],
                ["ROC-AUC", "Area under ROC", "Ranking quality across thresholds"],
                ["PR-AUC", "Area under PR curve", "Preferred for rare positive class"],
            ],
            [2.5 * cm, 4.5 * cm, 8.2 * cm],
        )
    )
    s.append(
        Paragraph(
            "Track all runs in <b>MLflow</b> (parameters, metrics, artifacts). Register the best model by "
            "<b>PR-AUC</b> on hold-out test set (80/20 split, stratified, random_state=42). Write unit tests in "
            "<font name='Courier' size='8'>tests/test_data_processing.py</font> (≥2 tests).",
            body,
        )
    )

    s.append(Paragraph("3.4 Task 6 — Deployment and CI/CD", h2))
    s.append(
        Paragraph(
            "Serve the registered model via <b>FastAPI</b> (<font name='Courier' size='8'>/predict</font> endpoint), "
            "containerize with <b>Docker</b>, and automate linting (flake8) + pytest in "
            "<font name='Courier' size='8'>.github/workflows/ci.yml</font> on every push to main.",
            body,
        )
    )

    s.append(Paragraph("4. Limitations", h1))
    s.append(
        Paragraph(
            "Fraud label ≠ credit default; proxy design remains an explicit modeling assumption. The observation "
            "window is short (~3 months), limiting seasonality analysis. Data is single-country (Uganda) and "
            "single-currency (UGX), constraining external validity.",
            body,
        )
    )

    s.append(Spacer(1, 0.25 * cm))
    s.append(
        Paragraph(
            "<i>Regenerate: python scripts/run_task2_eda.py then "
            "python scripts/generate_interim_report_pdf.py</i>",
            ParagraphStyle(
                "foot",
                parent=body,
                fontSize=7.5,
                alignment=TA_CENTER,
                textColor=colors.grey,
            ),
        )
    )

    return s


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    doc.build(build_story())
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
