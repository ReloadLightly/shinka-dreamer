"""Narrow seed-only resume compatibility for pinned native Shinka runner."""
from shinka.core import ShinkaEvolveRunner


class CheckpointRunner(ShinkaEvolveRunner):
    async def _setup_initial_program(self, code):
        # Upstream's resuming_run requires last_iteration > 0. A completed seed
        # is generation 0, so an interrupted first mutation otherwise duplicates
        # its island copies. Reuse the real persisted rows without changing their
        # generation, score, lineage or native sampling/evaluation machinery.
        existing = self.db.get_programs_by_generation(0)
        if existing:
            if any(program.code != code for program in existing):
                raise ValueError("Seed changed since checkpoint; use a new campaign directory")
            await self._restore_resume_progress()
            self.best_program_id = self.db.get_best_program().id
            if self.meta_summarizer:
                self.meta_summarizer.add_evaluated_program(existing[0])
            return
        await super()._setup_initial_program(code)
