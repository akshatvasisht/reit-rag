"""Hardcoded seed list for the corpus registry.

This module holds the authoritative bootstrap list of known documents.  It is
the fallback when the database is unreachable and the single source of truth
for unit tests that do not have a live DB connection.
"""

from __future__ import annotations

from typing import Any

CORPUS_REGISTRY_SEED: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Digital Realty — two versions, version-conflict test case
    # ------------------------------------------------------------------
    {
        "keywords": ["dlr", "digital realty", "digital_realty"],
        "company": "Digital Realty",
        "ticker": "DLR",
        "doc_type": "investor_presentation",
        "versions": {
            "dec-2025": {
                "version_keywords": ["dec", "december", "2025-12", "2025_12"],
                "report_date": "2025-12",
                "period_covered": "Q3 2025",
                "doc_version": "2025-12",
            },
            "mar-2026": {
                "version_keywords": ["mar", "march", "2026-03", "2026_03"],
                "report_date": "2026-03",
                "period_covered": "Q4 2025",
                "doc_version": "2026-03",
            },
        },
    },
    # ------------------------------------------------------------------
    # BXP — morning session deck (thematic/event, same company)
    # ------------------------------------------------------------------
    {
        "keywords": ["bxp"],
        "secondary_keywords": ["morning", "session", "morning_session"],
        "company": "BXP",
        "ticker": "BXP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2025-12",
        "period_covered": "Q4 2025",
        "doc_version": "2025-12",
    },
    # ------------------------------------------------------------------
    # BXP — investor presentation
    # ------------------------------------------------------------------
    {
        "keywords": ["bxp"],
        "secondary_keywords": ["q4", "investor", "presentation", "2025"],
        "company": "BXP",
        "ticker": "BXP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2025-12",
        "period_covered": "Q4 2025",
        "doc_version": "2025-12",
    },
    # ------------------------------------------------------------------
    # PSA (Public Storage) — company update (March 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["psa", "public storage", "public_storage"],
        "secondary_keywords": ["update", "company_update", "company update"],
        "company": "Public Storage",
        "ticker": "PSA",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # PSA (Public Storage) — merger presentation (March 16, 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["psa", "public storage", "public_storage"],
        "secondary_keywords": ["merger", "acquisition", "acq"],
        "company": "Public Storage",
        "ticker": "PSA",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # VICI Properties — investor presentation
    # ------------------------------------------------------------------
    {
        "keywords": ["vici"],
        "company": "VICI Properties",
        "ticker": "VICI",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-03",
        "period_covered": "Q4 2025",
        "doc_version": "2026-03",
    },
    # ------------------------------------------------------------------
    # Realty Income — investor presentation (February 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["realty income", "realty incom", "realty_income", "rlt", "o_corp"],
        "company": "Realty Income",
        "ticker": "O",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-02",
        "period_covered": "Q4 2025",
        "doc_version": "2026-02",
    },
    # ------------------------------------------------------------------
    # EastGroup Properties — roadshow deck (February 2026 cover)
    # ------------------------------------------------------------------
    {
        "keywords": ["eastgroup", "east_group", "egp"],
        "company": "EastGroup Properties",
        "ticker": "EGP",
        "doc_type": "investor_presentation",
        "versions": None,
        "report_date": "2026-02",
        "period_covered": "2026",
        "doc_version": "2026-02",
    },
    # ------------------------------------------------------------------
    # Simon Property Group — thematic report (November 2018 colophon)
    # ------------------------------------------------------------------
    {
        "keywords": ["simon", "spg"],
        "company": "Simon Property Group",
        "ticker": "SPG",
        "doc_type": "thematic_report",
        "versions": None,
        "report_date": "2018-11",
        "period_covered": "2017-2018",
        "doc_version": "2018-11",
    },
]
