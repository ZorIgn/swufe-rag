# Evaluation

`tests/canonical` is the regression suite for typed schemas, tool coverage,
entity generalization, MCP contract, API contract, citation validation and prompt
injection resistance. Development and holdout datasets are kept separate; source
code must never contain complete holdout questions.

The evaluation workflow reports query understanding, entity resolution, plan
exact match, tool precision/recall, structured correctness, citation correctness,
refusal quality, retrieval Recall/MRR/nDCG, and latency/cost percentiles. GPU and
large-model runs are optional scheduled workflows, not standard PR requirements.
