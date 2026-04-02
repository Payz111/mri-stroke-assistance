"""Report generation -- structured findings to prose.

Converts the V1Findings dictionary into a human-readable radiology
report string using the templates defined in :mod:`src.report.templates`.

Key principle: ZERO hallucinations -- every sentence is deterministically
produced from the structured JSON fields. No LLM generation.
"""
from __future__ import annotations

from typing import Any

from src.report.templates import (
    ADC_TEXT,
    FLAIR_TEXT,
    HEADER_TEMPLATE,
    IMPRESSION_TEMPLATE,
    IMPRESSION_TEXT,
    LATERALITY_TEXT,
    LESION_DETAIL_TEMPLATE,
    LESION_SUMMARY_TEMPLATE,
    MISMATCH_TEXT,
    NO_LESION_FINDINGS,
    PERFUSION_CORE_TEMPLATE,
    PERFUSION_HEADER_TEMPLATE,
    PERFUSION_HYPOPERFUSION_TEMPLATE,
    PERFUSION_IMPRESSION_TEMPLATE,
    PERFUSION_MISMATCH_TEMPLATE,
    PERFUSION_PENUMBRA_TEMPLATE,
    PERFUSION_QC_FAIL_TEMPLATE,
    PROTOCOL_TEMPLATE,
    QC_FAIL_TEMPLATE,
    TARGET_MISMATCH_TEXT,
)


def generate_report(findings: dict[str, Any]) -> str:
    """Generate a textual radiology report from structured findings.

    Parameters
    ----------
    findings:
        V1Findings dictionary as produced by
        :func:`src.findings.builder.build_findings`.

    Returns
    -------
    str
        Multi-section radiology report text (Header + Technique +
        Findings + Impression).
    """
    sections = []

    # Header
    sections.append(HEADER_TEMPLATE.format(
        study_id=findings.get("study_id", "unknown"),
        model_version=findings.get("model_version", "unknown"),
        timestamp=findings.get("timestamp", ""),
    ))

    # Quality gate warning
    qg = findings.get("quality_gate", {})
    if not qg.get("passed", True):
        reasons = ", ".join(qg.get("reasons", ["unspecified"]))
        sections.append(QC_FAIL_TEMPLATE.format(reasons=reasons))

    # Protocol / Technique
    protocol = findings.get("protocol_present", {})
    sequences = []
    if protocol.get("dwi"):
        sequences.append("DWI")
    if protocol.get("adc"):
        sequences.append("ADC")
    if protocol.get("flair"):
        sequences.append("FLAIR")
    if protocol.get("gre_swi"):
        sequences.append("GRE/SWI")
    if protocol.get("tof_mra"):
        sequences.append("TOF-MRA")
    sections.append(PROTOCOL_TEMPLATE.format(
        sequences=", ".join(sequences) if sequences else "none documented"
    ))

    # Findings
    lesions = findings.get("lesions", [])
    if not lesions:
        sections.append(NO_LESION_FINDINGS)
    else:
        total_vol = findings.get("total_lesion_volume_ml", 0.0)
        sections.append(LESION_SUMMARY_TEMPLATE.format(
            n_lesions=len(lesions),
            total_volume_ml=f"{total_vol:.1f}",
        ))

        for lesion in lesions:
            sections.append(_render_lesion(lesion))

    # V2 Perfusion section (if present)
    perfusion = findings.get("perfusion")
    if perfusion:
        sections.append(_render_perfusion(perfusion))

    # Impression
    impression = findings.get("overall_impression", "indeterminate")
    impression_text = IMPRESSION_TEXT.get(impression, impression)
    confidence = findings.get("overall_confidence", 0.0)

    impression_body = impression_text
    if lesions:
        vol_text = f"{findings.get('total_lesion_volume_ml', 0):.1f} mL"
        lateralities = set(l.get("laterality", "") for l in lesions)
        lat_str = ", ".join(sorted(lateralities))
        impression_body += f", {lat_str}, total volume {vol_text}"
    impression_body += f" (confidence: {confidence:.0%})."

    # Add perfusion impression if present
    if perfusion:
        perf_impression = _render_perfusion_impression(perfusion)
        if perf_impression:
            impression_body += "\n" + perf_impression

    sections.append(IMPRESSION_TEMPLATE.format(impression_body=impression_body))

    return "\n".join(sections)


def _render_perfusion(perfusion: dict[str, Any]) -> str:
    """Render the CT perfusion analysis section."""
    parts = []

    # Perfusion QC
    qc = perfusion.get("perfusion_quality_gate", {})
    if not qc.get("passed", True):
        reasons = ", ".join(qc.get("reasons", ["unspecified"]))
        parts.append(PERFUSION_QC_FAIL_TEMPLATE.format(reasons=reasons))

    # Header
    thresholds = perfusion.get("thresholds_used", {})
    tmax_t = thresholds.get("hypoperfusion_tmax_sec", 6.0)
    rcbf_t = thresholds.get("core_rcbf_threshold", 0.30)
    maps = perfusion.get("maps_present", {})
    available = [k.upper() for k, v in maps.items() if v]

    parts.append(PERFUSION_HEADER_TEMPLATE.format(
        source=perfusion.get("perfusion_source", "CTP"),
        maps_available=", ".join(available) if available else "none",
        tmax_threshold=tmax_t,
        rcbf_pct=int(rcbf_t * 100),
    ))

    # Core
    core = perfusion.get("core", {})
    parts.append(PERFUSION_CORE_TEMPLATE.format(
        rcbf_pct=int(rcbf_t * 100),
        core_volume_ml=f"{core.get('volume_ml', 0):.1f}",
    ))

    # Hypoperfusion
    hypo = perfusion.get("hypoperfusion", {})
    parts.append(PERFUSION_HYPOPERFUSION_TEMPLATE.format(
        tmax_threshold=tmax_t,
        hypo_volume_ml=f"{hypo.get('volume_ml', 0):.1f}",
    ))

    # Penumbra
    penumbra = perfusion.get("penumbra", {})
    if penumbra:
        parts.append(PERFUSION_PENUMBRA_TEMPLATE.format(
            penumbra_volume_ml=f"{penumbra.get('volume_ml', 0):.1f}",
        ))

    # Mismatch
    mismatch = perfusion.get("mismatch", {})
    status = mismatch.get("target_mismatch_status", "indeterminate")
    criteria = mismatch.get("criteria_used", {})

    status_template = TARGET_MISMATCH_TEXT.get(status, TARGET_MISMATCH_TEXT["indeterminate"])
    if status == "target_mismatch":
        status_text = status_template.format(
            core_max=criteria.get("core_max_ml", 70),
            ratio_min=criteria.get("mismatch_ratio_min", 1.8),
            vol_min=criteria.get("mismatch_volume_min_ml", 15),
        )
    else:
        status_text = status_template

    parts.append(PERFUSION_MISMATCH_TEMPLATE.format(
        ratio=mismatch.get("mismatch_ratio", 0),
        status_text=status_text,
    ))

    return "\n".join(parts)


def _render_perfusion_impression(perfusion: dict[str, Any]) -> str:
    """Render perfusion summary for the impression section."""
    core = perfusion.get("core", {})
    penumbra = perfusion.get("penumbra", {})
    mismatch = perfusion.get("mismatch", {})

    status = mismatch.get("target_mismatch_status", "indeterminate")
    status_short = {
        "target_mismatch": "target mismatch present",
        "no_mismatch": "no target mismatch",
        "indeterminate": "mismatch indeterminate",
    }.get(status, status)

    return PERFUSION_IMPRESSION_TEMPLATE.format(
        core_vol=f"{core.get('volume_ml', 0):.1f}",
        penumbra_vol=f"{penumbra.get('volume_ml', 0):.1f}",
        ratio=mismatch.get("mismatch_ratio", 0),
        status=status_short,
    )


def _render_lesion(lesion: dict[str, Any]) -> str:
    """Render a single lesion detail paragraph."""
    lesion_id = lesion.get("lesion_id", 0)
    volume_ml = lesion.get("volume_ml", 0.0)
    max_diam = lesion.get("max_diameter_mm", 0.0)

    laterality = LATERALITY_TEXT.get(
        lesion.get("laterality", "bilateral"), "bilateral"
    )

    locations = lesion.get("anatomic_location", ["multiple"])
    location = "/".join(locations)

    territory = lesion.get("vascular_territory", "uncertain")
    terr_conf = lesion.get("territory_confidence", 0.0)
    territory_note = ""
    if terr_conf < 0.5:
        territory_note = " (low confidence)"

    adc_confirmed = lesion.get("adc_low_confirmed", False)
    adc_status = ADC_TEXT.get(adc_confirmed, "unknown")

    flair_sig = lesion.get("flair_hyperintensity", "none")
    flair_note = ". " + FLAIR_TEXT.get(flair_sig, "")

    mismatch = lesion.get("dwi_flair_mismatch", "indeterminate")
    mismatch_status = MISMATCH_TEXT.get(mismatch, "indeterminate")

    return LESION_DETAIL_TEMPLATE.format(
        lesion_id=lesion_id,
        volume_ml=f"{volume_ml:.1f}",
        max_diameter_mm=f"{max_diam:.0f}",
        laterality=laterality,
        location=location,
        territory=territory,
        territory_note=territory_note,
        adc_status=adc_status,
        flair_note=flair_note,
        mismatch_status=mismatch_status,
    )
