# Agent runtime

The finite state machine is:

```text
UNDERSTAND → NORMALIZE → PLAN → EXECUTE → SYNTHESIZE → VALIDATE → FINISH
                                                        ↘ TARGETED_RETRIEVAL (once) ↗
NORMALIZE/PLAN → CLARIFY → FINISH
```

`AgentRuntime` receives `QuestionUnderstanding`, `QueryNormalizer`,
`ExecutionPlanner`, `ToolExecutor`, `AnswerSynthesizer`, `AnswerValidator`,
`SessionStore`, and `Tracer` explicitly. Sessions retain only structured context
with a TTL; API keys, model clients and full raw secrets are excluded.

The `ToolRegistry` is the sole authority for tool names, schemas, timeouts and
read-only status. Its definitions are also used by the MCP adapter.
