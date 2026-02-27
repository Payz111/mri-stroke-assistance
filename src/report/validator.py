"""Report validation -- cross-check prose against structured findings.

Ensures the generated report text is consistent with the underlying
findings data and flags potential hallucinations or omissions.
"""
from __future__ import annotations

import re
from typing import Any


def validate_report(
    report_text: str,
    findings: dict[str, Any],
) -> dict[str, Any]:
    """Validate a generated report against its source findings.

    Checks:
    1. Lesion count mentioned matches findings
    2. Volume mentioned matches findings
    3. Each lesion's laterality appears in text
    4. Impression class appears in text
    5. No numbers in report that don't come from findings

    Parameters
    ----------
    report_text:
        The generated radiology report text.
    findings:
        The V1Findings dictionary that the report was derived from.

    Returns
    -------
    dict
        Validation result with keys:
            - ``valid`` (bool): overall pass/fail
            - ``issues`` (list[str]): descriptions of any discrepancies
            - ``checks`` (dict): per-check pass/fail detail
    """
    issues = []
    checks = {}
    text_lower = report_text.lower()

    # Check 1: Lesion count
    n_lesions = findings.get("total_lesion_count", 0)
    if n_lesions == 0:
        count_ok = "no acute" in text_lower or "no ischemic" in text_lower
        if not count_ok:
            issues.append(
                f"Report should indicate no lesions (found {n_lesions})"
            )
    else:
        count_ok = str(n_lesions) in report_text
        if not count_ok:
            issues.append(
                f"Lesion count {n_lesions} not found in report text"
            )
    checks["lesion_count"] = count_ok

    # Check 2: Total volume
    total_vol = findings.get("total_lesion_volume_ml", 0.0)
    vol_str = f"{total_vol:.1f}"
    vol_ok = vol_str in report_text or n_lesions == 0
    if not vol_ok:
        issues.append(f"Total volume {vol_str} mL not found in report")
    checks["total_volume"] = vol_ok

    # Check 3: Each lesion's laterality mentioned
    lesions = findings.get("lesions", [])
    lat_ok = True
    for lesion in lesions:
        lat = lesion.get("laterality", "")
        if lat and lat.lower() not in text_lower:
            lat_ok = False
            issues.append(
                f"Lesion {lesion.get('lesion_id')}: "
                f"laterality '{lat}' not in report"
            )
    checks["laterality_mentioned"] = lat_ok

    # Check 4: Impression class
    impression = findings.get("overall_impression", "")
    impression_keywords = {
        "acute": ["acute"],
        "early_subacute": ["subacute"],
        "no_acute_infarct": ["no acute"],
        "indeterminate": ["indeterminate"],
    }
    keywords = impression_keywords.get(impression, [impression])
    impression_ok = any(kw in text_lower for kw in keywords)
    if not impression_ok:
        issues.append(f"Impression '{impression}' not reflected in report")
    checks["impression_present"] = impression_ok

    # Check 5: No hallucinated numbers (volumes)
    # Extract decimal numbers, but exclude timestamps and template constants
    # Remove timestamp line before scanning for numbers
    lines_to_check = []
    for line in report_text.split("\n"):
        # Skip header lines (timestamp, study ID, model version)
        if line.startswith("Generated:") or line.startswith("Study ID:"):
            continue
        lines_to_check.append(line)
    scannable_text = "\n".join(lines_to_check)

    report_numbers = set(re.findall(r"\d+\.\d+", scannable_text))

    # Known template constants (not from findings data)
    template_constants = {"4.5"}  # "<4.5h onset" in mismatch text

    allowed_numbers = set()
    allowed_numbers.update(template_constants)
    allowed_numbers.add(f"{total_vol:.1f}")
    for lesion in lesions:
        allowed_numbers.add(f"{lesion.get('volume_ml', 0):.1f}")
        allowed_numbers.add(f"{lesion.get('max_diameter_mm', 0):.1f}")
        allowed_numbers.add(f"{lesion.get('territory_confidence', 0):.2f}")
    # Confidence percentages
    conf = findings.get("overall_confidence", 0)
    allowed_numbers.add(f"{conf:.2f}")
    allowed_numbers.add(f"{conf:.0%}".replace("%", ""))

    suspicious = []
    for num in report_numbers:
        val = float(num)
        if val < 0.01 or num in allowed_numbers:
            continue
        close_match = any(
            abs(val - float(a)) < 0.15 for a in allowed_numbers if a
        )
        if not close_match and val > 1.0:
            suspicious.append(num)

    numbers_ok = len(suspicious) == 0
    if not numbers_ok:
        issues.append(f"Unverified numbers in report: {suspicious}")
    checks["no_hallucinated_numbers"] = numbers_ok

    valid = len(issues) == 0
    return {"valid": valid, "issues": issues, "checks": checks}
