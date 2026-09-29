"""Checks for recovery defects that could duplicate work or lose native guidance."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "recover_campaign", Path(__file__).resolve().parents[1] / "scripts/recover_campaign.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_controller_lock_rejects_overlap_and_releases(tmp_path):
    with recovery.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already holds"):
            with recovery.campaign_lock(tmp_path):
                pytest.fail("Duplicate controller entered")
    with recovery.campaign_lock(tmp_path):
        pass


def test_restore_cumulative_native_guidance_and_pending_programs(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "meta_2.txt").write_text(
        "# INDIVIDUAL PROGRAM SUMMARIES\n\n"
        "A seed and a descendant.\n"
        "**Program Identifier:** Generation 0 - Patch Name initial_program\n"
        "**Program Identifier:** Generation 1 - Patch Name learned_flow\n\n"
        "# GLOBAL INSIGHTS SCRATCHPAD\n\nKeep forecasts before outcomes.\n\n"
        "# META RECOMMENDATIONS\n\nMeasure learning on matched observations.\n")

    class Database:
        last_iteration = 3

        def get_programs_by_generation(self, generation):
            return [SimpleNamespace(generation=generation)]

    class Summarizer:
        def add_evaluated_program(self, program):
            self.evaluated_since_last_meta.append(program)

    summarizer = Summarizer()
    result = recovery.restore_recommendations(summarizer, Database(), tmp_path)
    assert result["processed_generations"] == [0, 1]
    assert result["pending_generations"] == [2, 3]
    assert summarizer.meta_recommendations == "Measure learning on matched observations."
    assert summarizer.meta_recommendations_history == [summarizer.meta_recommendations]
    assert summarizer.total_programs_processed == 2
    assert [p.generation for p in summarizer.evaluated_since_last_meta] == [2, 3]
