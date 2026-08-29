# Security

## Request and model boundary

- Request bodies, question length, concurrency, queue wait and provider timeouts are bounded.
- Provider credentials are request-scoped headers. They are not written to sessions, traces, metrics, logs, errors or caches.
- An API key does not trigger an external call unless a provider endpoint and model are explicitly configured.
- Document text is data. It cannot change tool schemas, invoke functions, generate SQL, reveal prompts or expand the execution plan.
- Errors use a stable public schema and do not echo Python exceptions or local paths.

## Authentication and debug

Local mode may run without authentication for development. <code>SWUFE_DEPLOYMENT_MODE=production</code> requires either a trusted principal resolver or a configured bearer token. A production static bearer token must have at least 32 characters and is intended for controlled demonstrations; public deployments should terminate authentication at a trusted gateway or integrate an organization principal resolver.

Forwarded client addresses are trusted only when the direct peer belongs to <code>SWUFE_TRUSTED_PROXY_CIDRS</code>. Arbitrary tenant or forwarding headers are not treated as identities.

Debug output is disabled by default. It is returned only when:

1. <code>SWUFE_ENABLE_DEBUG_RESPONSES=true</code>;
2. the caller is authenticated with the <code>admin</code> role;
3. the request explicitly sets <code>debug=true</code>.

The debug schema exposes bounded operation metadata, not SQL, provider secrets or raw internal prompts.

## Session boundary

Anonymous session continuity is disabled by default. Authenticated session keys are principal-scoped and hashed before storage. In-memory and Redis implementations enforce TTL, message count and payload size. Redis values use strict JSON and are bound to the dataset version; malformed, oversized or stale values are discarded.

Redis is an optional shared session backend. It is not a retrieval cache or distributed rate limiter. Production Redis transport, ACL, TLS, backup and network policy remain operator responsibilities.

## Release boundary

Production mode cannot opt into an unattested active release. Runtime loading verifies:

- strict JSON with duplicate-key and non-finite-number rejection;
- active pointer and release-directory identity;
- every declared file hash;
- content-addressed attestation publication;
- Ed25519 signature, key ID and trusted issuer;
- release, holdout, model snapshot, Git and evaluation-report binding.

Candidate builds record dirty-tree state and diff digest. Promotion evaluation requires a clean evaluator at the exact candidate commit. A production candidate should therefore be built in a clean CI checkout.

## API and infrastructure boundary

- CORS uses an explicit allow-list.
- The source endpoint returns publication URLs and provenance, never local filesystem paths.
- The built-in rate limiter and concurrency semaphore are process-local.
- Multi-instance deployments require gateway-level authentication, global rate limiting and abuse controls.
- The Docker image runs as a non-root user and uses the checked-in Python dependency lock, but base-image tags and external model snapshots prevent a claim of bit-for-bit reproducibility unless they are separately digest-pinned.

This repository does not provide SSO, multi-tenant authorization policy, WAF rules, secret rotation, SBOM publication or a compliance audit log.
