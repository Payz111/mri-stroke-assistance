"""Report template strings.

Contains the text templates used by the report generator to convert
structured findings into human-readable radiology-style prose.
"""
from __future__ import annotations

FINDINGS_TEMPLATE: str = """\
FINDINGS:
{findings_body}
"""
"""Template for the Findings section of the radiology report.

Placeholders
------------
findings_body : str
    Rendered per-lesion findings paragraphs.
"""

IMPRESSION_TEMPLATE: str = """\
IMPRESSION:
{impression_body}
"""
"""Template for the Impression section of the radiology report.

Placeholders
------------
impression_body : str
    One-line clinical summary.
"""
