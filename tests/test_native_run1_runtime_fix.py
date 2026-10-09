"""Deterministic runtime-clock checks; zero model requests and world episodes."""
from dataclasses import dataclass
from pathlib import Path

import pytest

from dreamer.native_run1_runtime_fix import (evaluation_timeout_view,
    install_headless_child_timeout, install_scheduler_timeout)


@dataclass
class Job:
    job_id: object
    start_time: float = 100.0
    evaluation_started_at: float = 700.0
    evaluation_submitted_at: float = 699.0
    generation: int = 1


class Process:
    pid = 12345
    killed = False

    def kill(self):
        self.killed = True

    def poll(self):
        return -9 if self.killed else None


def test_long_proposal_does_not_consume_evaluation_time(monkeypatch):
    from shinka.launch import scheduler as module
    monkeypatch.setattr(module, 'ProcessWithLogging', Process)
    monkeypatch.setattr(module.time, 'time', lambda: 710.0)
    scheduler = module.JobScheduler('local', module.LocalJobConfig(
        eval_program_path='/unused/evaluate.py', time='00:08:00'))
    job = Job(Process())
    install_scheduler_timeout(scheduler)
    try:
        assert scheduler.check_job_status(job) is True
        assert not job.job_id.killed
        assert job.start_time == 100.0  # Actual pipeline metadata is unchanged.
        monkeypatch.setattr(module.time, 'time', lambda: 1181.0)
        assert scheduler.check_job_status(job) is False
        assert job.job_id.killed
        assert job.start_time == 100.0
    finally:
        scheduler.executor.shutdown(wait=True)


def test_missing_start_falls_back_to_submission_and_invalid_fails():
    job = Job(Process(), evaluation_started_at=None)
    view = evaluation_timeout_view(job)
    assert view.start_time == 699.0
    assert view.job_id is job.job_id
    job.evaluation_submitted_at = None
    with pytest.raises(ValueError, match='valid start'):
        evaluation_timeout_view(job)


def test_headless_supported_child_timeout_is_in_actual_command(monkeypatch):
    from shinka.llm.providers import headless
    monkeypatch.setattr(headless, '_build_headless_command', headless._build_headless_command)
    monkeypatch.setattr(headless, '_run1_child_timeout_installed', False, raising=False)
    monkeypatch.setenv('SHINKA_HEADLESS_COMMAND', '/unused/subscription-wrapper')
    original = headless._build_headless_command
    kwargs = {'model': headless.parse_headless_model('headless/codex@gpt-6.1-sol?effort=high'),
              'prompt_path': Path('/unused/prompt.md'), 'work_dir': '/unused/campaign'}
    before = original(**kwargs)
    install_headless_child_timeout()
    after = headless._build_headless_command(**kwargs)
    assert after == before + ['--timeout', '590']
    assert after[after.index('--reasoning-effort') + 1] == 'high'
    assert after[after.index('--allow') + 1] == 'read-only'
    with pytest.raises(RuntimeError, match='already installed'):
        install_headless_child_timeout()

