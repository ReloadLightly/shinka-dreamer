"""Additive RUN1 runtime fixes; the original frozen adapter remains unchanged.

Only local scheduler timeout inspection receives an evaluation-clock view. The
actual native job retains its proposal/pipeline timestamps and all other fields.
Headless itself receives a child timeout, inside the unchanged outer 600s cap,
so it can terminate its Codex process group before the provider kills Headless.
"""
from dataclasses import replace
import math

from dreamer.native_run1 import Run1Runner

HEADLESS_CHILD_TIMEOUT_SECONDS = 590
INFRASTRUCTURE_FEEDBACK = (
    'Generation1 was killed by a scheduler timing bug before any episodes; its '
    'score0 is an infrastructure failure, not evidence about its algorithm. '
    'The timer now starts at evaluation launch.'
)


def evaluation_timeout_view(job):
    started = job.evaluation_started_at
    if started is None:
        started = job.evaluation_submitted_at
    if not isinstance(started, (int, float)) or not math.isfinite(started) or started <= 0:
        raise ValueError('Local evaluation has no valid start/submission timestamp')
    return replace(job, start_time=started)


def install_scheduler_timeout(scheduler):
    original = scheduler.check_job_status

    def check_job_status(job):
        if scheduler.job_type == 'local':
            return original(evaluation_timeout_view(job))
        return original(job)

    scheduler.check_job_status = check_job_status


def install_headless_child_timeout():
    from shinka.llm.providers import headless
    if getattr(headless, '_run1_child_timeout_installed', False):
        raise RuntimeError('RUN1 child timeout hook is already installed')
    original = headless._build_headless_command

    def build_headless_command(**kwargs):
        command = original(**kwargs)
        if '--timeout' in command:
            raise ValueError('Unexpected existing Headless child timeout')
        return command + ['--timeout', str(HEADLESS_CHILD_TIMEOUT_SECONDS)]

    headless._build_headless_command = build_headless_command
    headless._run1_child_timeout_installed = True


class CorrectedRunner(Run1Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The inherited native sample/fix closures read this boundary at request
        # time, after composing the evolved prompt and operator-specific format.
        self.immutable_prompt_boundary += '\n\n' + INFRASTRUCTURE_FEEDBACK
        install_scheduler_timeout(self.scheduler)

