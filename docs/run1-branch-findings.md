# RUN1 first-action branches: limited negative evidence, validator failures

The bounded development branch experiment is closed. Among the six action-
disagreement states with complete consequences, changing the first action did
not change collision or escape in the eight paired 12-transition futures. One
uniform-law future collected one fewer key after the learned recommendation
than after the prior/known-law recommendation. Five other disagreement states
are unavailable because a state validator rejected their copied memories.
This small, incomplete diagnostic does not establish that adaptation is useless.

The protocol was committed as `edf9c1f` before execution. It reused the first
four fitting-subvalidation memory recordings per original v3 regime, limited
to their first 80 frames. These are exposed development data, not assessment.
The evaluator reconstructed their physical states and checked each public
observation. It retained up to four agreeing and four disagreeing encounters per
regime, with a visible anonymous enemy, at least two legal empty destinations,
at most two states per recording and an eight-frame separation.

At each state the original selected program recommended an action using its
learned forecast. Copies of the same state made recommendations with only the
forecast's movement law replaced by the uniform initial prior or current true
law. This is a one-decision law replacement, not freezing the learning history.
Only that known-law copy was privileged. Every consequence branch restored the
same ordinary learned memory, forced one of the distinct first actions, and then
used the unchanged learned policy on its own subsequent observations. Duplicate
actions share one execution; agreeing controls are zero treatments by construction.

| Regime | Selected states | Disagreement states | Disagreement states with complete consequences |
|:--|--:|--:|--:|
| Uniform | 7 | 3 | 1 |
| Stationary | 8 | 4 | 2 |
| Switch | 8 | 4 | 3 |

The uniform disagreement quota was underfilled; it was not relaxed. There were
40 valid paired futures with different learned/prior first actions: eight in
uniform dynamics and 16 each in stationary and switching dynamics. Thirty-nine
had identical registered return; the remaining uniform future favored the prior
by 0.25 because it collected one additional key. Across 32 valid paired futures
with different known-law/learned first actions, the same event favored known-law;
all other returns agreed. No valid pair differed in collision or escape.
These futures share encounter states and recordings; they are not independent
episode replicates. Twelve transitions may miss later navigation consequences.

The registered return separates collision (−1), escape (+1), keys gained (+0.25
each), opening the door (+0.25), and executed transitions (−0.01 each). All
components are retained. Invalid branches receive conservative return −1.12 and
a collision-or-invalid flag, while remaining distinct from observed collision.
The original all-attempt summary uses that convention. The saved-data
`reviewed-summary.json` additionally isolates complete paired consequences and
explicitly marks missing states. A zero difference between two equally imputed
invalid returns is not an observed zero treatment effect.

| Execution count or cost | Actual |
|:--|--:|
| Reused recording reconstructions | 12 |
| Reconstruction transitions | 449 / 960 cap |
| Selected encounter states | 23 / 24 cap |
| Future worker attempts | 184 / 192 cap |
| Complete / failed future workers | 128 / 56 |
| Distinct action branches: valid / invalid | 176 / 96 |
| Actual branch transitions | 2,032 / 6,912 cap |
| Candidate operations, including selection and replay | 7,251 / 22,272 cap |
| Worker CPU / controller CPU | 116.75 s / 6.00 s |
| Monotonic execution elapsed | 72.03 s |
| Model calls / assessment cases | 0 / 0 |

Valid branches ended in 158 completed windows, two collisions and 16 escapes.
These are short branches from internal episode states, not 176 fresh complete
maze episodes. Execution finished normally at 01:08:06 UTC on 9 October, before
the operator could stop it after identifying the validator issue. No failed
attempt was retried and no new diagnostic was launched.

The validator used a SHA-256 digest of pickle bytes to compare copied candidate
memory. Equal built-in sets can have different serialization order. A synthetic
test demonstrates equal values and unequal pickle bytes; the future-use fix
uses recursively typed canonical values and an explicit built-in value-equality
check. Rejected live memory values were not retained, so this demonstration
cannot retrospectively verify those failed states or validate their unexecuted
branches. All 56 failures remain. The executed worker, its original hash,
protocol and failure checkpoints are preserved; the corrected worker was not
executed in this experiment.

The useful development guidance is narrow: improved dynamics information can
change the selected planner's actions, but the evaluated local changes supplied
no survival advantage and almost no measured short-window progress difference.
Future candidates should connect predictive changes to meaningful decisions;
they should not maximize a contrast by making a disabled counterpart worse.
The new-wave environment factors and absolute task objective are separately
registered and are not justified as validated constants by this diagnostic.

Reproduce the saved-data review, with no candidate or environment execution:

```bash
.venv/bin/python scripts/run1_branch_report.py
```

Evidence is in `artifacts/run1/branches`. The one-shot execution unit is closed;
the branch command is not a continuation authorization.
