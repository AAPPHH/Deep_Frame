# Code Style

- Put thematically related code in one file; start a new file only at ~1000 lines. No mini-modules.
- No argparse. Configuration as plain dicts.
- OOP, DRY. No docstrings, no comments, no unnecessary blank lines.
- Limit prints to what is functionally necessary.

# Way of Working

- Work as an orchestrator in a leading role; delegate execution.
- One commit per logical step.
- At the end of each task or every 10 minutes, give a short status summary in the chat.
- Submit compute runs only via the local Ray scheduler, with declared requirements (cores, RAM, GPU GB); no fixed slots, no direct launches.
- After starting a job, check hardware utilization 30 seconds later and then every 60 seconds.

# Project

- **Version 1, classical iterative:** GPU topology optimization → implicit geometry → FEA. Goal: a pipeline that, without human shape input, automatically produces print-ready, validated frames in the style of ManaFly and is robust and fast enough to generate datasets across varied requirements. Teacher for everything that follows.
- **Version 2, partially ML:** ML accelerates or replaces parts of the classical loop, e.g. an FEA surrogate with active learning or neural reparameterization. The surrogate serves as physics guidance in Version 3.
- **Version 3, end-to-end diffusion:** Requirements (motor positions, components, keep-outs, loads, target values) in, 3D shape as density field or SDF out. Trained on data from Version 1, guided by the surrogate from Version 2, every candidate validated with real FEA and printing. Open problem: satisfying hard assembly constraints.

# Rule for This File

- Do not write anything further into this CLAUDE.md.
