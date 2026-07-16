import numpy as np

from math_utils import clean_values, nanmean, nanstd

from .constants import METRIC_LABELS


def format_mean_std(values, digits=3):
    values = clean_values(values)
    if values.size == 0:
        return "NA"
    mean = nanmean(values)
    std = nanstd(values, ddof=1) if values.size > 1 else 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


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
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"${mantissa:.{sig_digits}g} \times 10^{{{exponent}}}$"


def latex_escape_text(value):
    if value is None:
        return ""
    text = str(value)
    if "$" in text or text.startswith("\\"):
        return text
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def metric_label(metric_name):
    return METRIC_LABELS.get(metric_name, latex_escape_text(metric_name))


def format_decision_rule(threshold, direction, group_name, digits=4):
    if direction == "NA" or threshold is None or not np.isfinite(threshold):
        return "NA"
    backslash = chr(92)
    operator = (
        "$" + backslash + "geq$"
        if direction == ">="
        else "$" + backslash + "leq$"
    )
    threshold_text = format_float(threshold, digits=digits)
    group_text = latex_escape_text(group_name)
    return (
        f"score {operator} {threshold_text} $"
        + backslash
        + f"rightarrow$ {group_text}"
    )
