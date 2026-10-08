# Native mutation and recommendation worker

This directory is an inference workspace for a native ShinkaEvolve request.
The complete allowed scientific context is in the supplied request: task,
parent/inspiration programs, development feedback and native recommendations.

Do not call tools, run shell commands, read files, inspect parent directories,
or retrieve other experiment outputs. In particular, do not open validation,
comparator fitting or assessment data. Work only from the supplied request.

The repository's bootstrap, implementation, test, reporting, commit and push
workflow belongs to the outer experiment controller. This worker only returns
the code, patch or recommendation in the exact requested native format.
Representations, predictive updates, memory, helpers and planning remain open
to substantive evolution within the task's interface and resource boundaries.
