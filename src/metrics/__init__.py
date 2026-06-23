"""Metrics layer - the single KPI extractor (one implementation, no drift).

``kpi_extractor`` computes the seven locked KPIs (kpis.md) from a JSONL trace +
SUMO trip-info XML. The same function is called by eval (final results tables) and
by training validation, so a KPI is defined exactly once.
"""

from __future__ import annotations

from src.metrics.kpi_extractor import EpisodeKPIs, extract_kpis

__all__ = ["EpisodeKPIs", "extract_kpis"]
