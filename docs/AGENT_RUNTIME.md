# Agent runtime

## State machine

~~~text
UNDERSTAND -> NORMALIZE -> PLAN -> EXECUTE -> SYNTHESIZE -> VALIDATE -> FINISH
                                                     |
                                                     v
                                           TARGETED_RETRIEVAL
                                              (at most once)

NORMALIZE / PLAN -> CLARIFY -> FINISH
~~~

<code>AgentRuntime</code> receives question understanding, normalization, planning, tool execution, synthesis, validation, coverage, repair, sessions and tracing as explicit dependencies.

## Planning and execution

The planner creates a bounded typed DAG. Each operation has a stable operation ID, a registered tool name, typed arguments and explicit dependencies. Each requested output has a contract that names both the required capability and its producer operation IDs.

The executor enforces the registry, total call budget, dependency ordering, per-tool timeouts and one plan deadline. Failure becomes a typed execution result; it is not converted into an empty success.

## Synthesis and validation

The deterministic synthesizer is the default. A request-scoped structured model may assist understanding and synthesis when an endpoint and API key are explicitly supplied. LLM output is still constrained by ClaimDraft / ClaimAtom schemas and must pass evidence, comparator, polarity, unit, scope and coverage validation.

A validation failure may request one targeted policy retrieval. There is no unbounded retry loop. If the repair cannot supply the missing evidence, the final result remains a clarification or refusal.

## Sessions

Sessions store only bounded structured context:

- dataset version;
- canonical program and cohort context;
- a bounded message window;
- TTL and payload-size constraints.

The default in-memory store is suitable for a single process. <code>SWUFE_SESSION_BACKEND=redis</code> selects the tested Redis implementation for shared multi-worker continuity. Redis values are strict JSON, size-bounded, principal-scoped and dataset-version-scoped; unsafe or stale values are deleted. A configured Redis backend fails closed rather than silently falling back to process memory.

Provider keys, raw tool outputs and full prompts are not persisted.

## MCP naming boundary

<code>agent.mcp.MCPAdapter</code> maps one typed operation to the same ToolRegistry used by HTTP. It demonstrates transport-independent tool contracts, but the repository does not implement an MCP server, session transport, discovery endpoint or remote authentication.
