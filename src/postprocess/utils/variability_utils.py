import numpy as np
import re
from pathlib import Path
try:
    from scipy.stats import mannwhitneyu, norm
except ImportError as exc:
    raise ImportError(
        "This script requires scipy for Mann-Whitney tests. Install it with: pip install scipy"
    ) from exc


EPS = 1e-12
# -----------------------------------------------------------------------------
# Robust 1D statistics
# -----------------------------------------------------------------------------


def finite_1d(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def iqr_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    q25 = np.nanpercentile(x, 25)
    q75 = np.nanpercentile(x, 75)
    return float(q75 - q25)


def mad_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def cv_1d(x, eps=EPS):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1) if x.size > 1 else 0.0
    return float(sd / (np.abs(mu) + eps))


def median_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.nanmedian(x))


def std_1d(x):
    x = finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.nanstd(x, ddof=1) if x.size > 1 else 0.0)


def nanmedian_or_nan(x):
    x = np.asarray(x, dtype=float)
    if np.any(np.isfinite(x)):
        return float(np.nanmedian(x))
    return np.nan


def clean_values(values):
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]

# -----------------------------------------------------------------------------
# Format
# -----------------------------------------------------------------------------

def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def format_pvalue_latex(value, sig_digits=3, threshold=1e-3):

    if value is None or not np.isfinite(value):
        return "NA"

    value = float(value)

    if value == 0.0:
        return r"$<10^{-300}$"

    if abs(value) < threshold:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exponent)
        return rf"${mantissa:.{sig_digits}g} \times 10^{{{exponent}}}$"

    return f"{value:.{sig_digits}g}"


def latex_escape_text(value):

    if value is None:
        return ""

    s = str(value)

    # Already math-mode or already a LaTeX command/table fragment: leave unchanged.
    if "$" in s or s.startswith("\\"):
        return s

    replacements = {
        "\\": r"	extbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"	extasciitilde{}",
        "^": r"	extasciicircum{}",
    }

    return "".join(replacements.get(ch, ch) for ch in s)

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

def save_table(df, csv_path, tex_path, caption, label, digits=3):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    latex = dataframe_to_latex_table(
        df, caption=caption, label=label, font_size=r"\scriptsize"
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    return [csv_path, tex_path]

def dataframe_to_latex_table(
    df,
    caption=None,
    label=None,
    font_size=r"\scriptsize",
):
    r"""
    Requires in Overleaf preamble:
        \usepackage{float}
        \usepackage{booktabs}
    """
    latex_tabular = df.to_latex(
        index=False,
        escape=False,
        longtable=False,
        column_format="l" + "c" * (df.shape[1] - 1),
    )

    lines = [
        r"\begin{table}[H]",
        r"\raggedright",
        font_size,
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.9}",
    ]

    if caption:
        lines.append(f"\\caption{{{caption}}}")

    if label:
        lines.append(f"\\label{{{label}}}")

    lines.append(latex_tabular)
    lines.append(r"\end{table}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Stats group
# -----------------------------------------------------------------------------

def mann_whitney_pvalue(control_values, group_values):
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan

    try:
        res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except ValueError:
        return np.nan
    

def cohen_d(control_values, group_values):
    """
    Cohen's d using pooled standard deviation.

    Positive values mean that the compared group has a larger mean than control.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size < 2 or y.size < 2:
        return np.nan

    sx = np.nanstd(x, ddof=1)
    sy = np.nanstd(y, ddof=1)
    pooled_var = ((x.size - 1) * sx**2 + (y.size - 1) * sy**2) / (x.size + y.size - 2)

    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return np.nan

    return float((np.nanmean(y) - np.nanmean(x)) / np.sqrt(pooled_var))

def auc_from_scores(control_values, group_values):
    """
    ROC AUC computed from Mann-Whitney ranks.

    AUC is oriented so that higher scores predict the compared group.
    If AUC < 0.5, the separability is in the opposite direction; for practical
    discrimination strength, use max(AUC, 1 - AUC).
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan

    try:
        u = mannwhitneyu(y, x, alternative="two-sided", method="auto").statistic
        return float(u / (x.size * y.size))
    except ValueError:
        return np.nan
    

def mean_difference_ci95(control_values, group_values):
    """
    Approximate 95% CI for the mean difference group - control.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size < 2 or y.size < 2:
        return np.nan, np.nan, np.nan

    diff = float(np.nanmean(y) - np.nanmean(x))
    se = np.sqrt(np.nanvar(x, ddof=1) / x.size + np.nanvar(y, ddof=1) / y.size)

    if not np.isfinite(se):
        return diff, np.nan, np.nan

    return diff, float(diff - 1.96 * se), float(diff + 1.96 * se)

def best_threshold_sensitivity_specificity(control_values, group_values):
    """
    Finds the threshold maximizing Youden's index.

    The function automatically chooses the direction of classification:
    - disease/group positive if score >= threshold when group tends to be higher;
    - disease/group positive if score <= threshold when group tends to be lower.
    """
    x = clean_values(control_values)
    y = clean_values(group_values)

    if x.size == 0 or y.size == 0:
        return np.nan, np.nan, np.nan, "NA"

    values = np.unique(np.concatenate([x, y]))
    if values.size == 1:
        return float(values[0]), np.nan, np.nan, "NA"

    thresholds = (values[:-1] + values[1:]) / 2.0
    group_higher = np.nanmedian(y) >= np.nanmedian(x)

    best = None
    for threshold in thresholds:
        if group_higher:
            tp = np.sum(y >= threshold)
            fn = np.sum(y < threshold)
            tn = np.sum(x < threshold)
            fp = np.sum(x >= threshold)
            direction = ">="
        else:
            tp = np.sum(y <= threshold)
            fn = np.sum(y > threshold)
            tn = np.sum(x > threshold)
            fp = np.sum(x <= threshold)
            direction = "<="

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        youden = sensitivity + specificity - 1.0

        candidate = (youden, threshold, sensitivity, specificity, direction)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return np.nan, np.nan, np.nan, "NA"

    _, threshold, sensitivity, specificity, direction = best
    return float(threshold), float(sensitivity), float(specificity), direction


def overlap_from_cohen_d(d):
    """
    Gaussian equal-variance overlap approximation: OVL = 2 Phi(-|d|/2).
    """
    if d is None or not np.isfinite(d):
        return np.nan
    return float(2.0 * norm.cdf(-abs(float(d)) / 2.0))

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_")

def extract_sort_key(filename):
    name = Path(filename).name

    date_match = re.search(r"(\d{6})", name)
    date = int(date_match.group(1)) if date_match else 0

    hd_match = re.search(r"_(\d+)_HD", name)
    hd_index = int(hd_match.group(1)) if hd_match else 0

    return date, hd_index