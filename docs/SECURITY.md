# Security

- Provider keys are request-scoped headers and are never written to session,
  trace, metrics, logs, errors or cache.
- Request bodies have a size limit and API errors use `ErrorResponse`, not raw
  Python exceptions or invalid request echoes.
- CORS is an explicit allow-list; rate limiting and per-tool timeouts are bounded.
- Document text is treated solely as data. It cannot modify tool schemas,
  invoke functions, reveal prompts, or generate SQL.
- Source endpoint responses expose publication URLs and provenance only; local
  filesystem paths are never returned.
