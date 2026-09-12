# Prototype security and feedback review — 2026-09-12

This is a scoped code/dependency review, not a penetration test or security certification.

## Security findings

- **CORS:** wildcard origins remain intentionally unchanged. The real frontend is same-origin at `https://krishmalik-delta-v.hf.space`, so it does not need CORS. Restricting to that origin is safe for this frontend, but could break external API consumers and embedded clients. Recommend an explicit configurable allowlist after deciding which clients are supported. CORS is not authentication or DoS protection.
- **Inputs/DoS:** added bounded request bodies (rollout JSON 64 MiB, field multipart 24 MiB, other writes 16 KiB), native-grid/channel/finite-value checks before inference, NPY header checks before allocation, strict integer rollout steps 1–30, sample/system ownership checks and stale-revision rejection. A shared non-queueing inference lock returns 429 with Retry-After for competing field/rollout computations; inference runs off the HTTP event loop. Cancellation in the browser does not stop an already-running CPU rollout. Public inference remains a compute-abuse surface.
- **Rate limiting:** no per-IP/time-window limit exists. Single-computation admission is concurrency control, not a rate limit. Recommend a gateway rate limit and request budget before substantial traffic; no visitor identifiers were added to analytics.
- **Secrets:** checked tracked content and all locally available Git revisions for HF/GitHub/OpenAI-style tokens and private-key markers; no matches. No `.env` history found. Added `.env` exclusions to Git and Docker context. HF credentials are read from the environment, not hardcoded. Pattern scanning cannot prove that every possible secret is absent.
- **Dependencies:** `pip-audit -r requirements.txt` resolved 71 packages and reported no known vulnerabilities. This audits the current resolver result, not a frozen production inventory. Requirements use lower bounds, so older installed versions could differ; Docker also installs Torch separately. A reproducible Linux/Python 3.11 lockfile and deployed-image/SBOM scan remain recommended. No claim that the complete container is vulnerability-free.
- **XSS:** share identifiers are resolved against the served catalog and URL-encoded; share and feedback values use input `.value`/textContent, not HTML. Feedback bodies are URL-encoded for GitHub. Saved-run text uses escapeHtml and preview images require PNG data prefixes. Reviewed relevant innerHTML paths: primarily trusted server catalogs/static templates or escaped text/numeric formatting. No direct URL/feedback-text injection found. Backend catalog/artifact integrity remains a trust boundary.
- **Artifacts:** checkpoint loading uses Torch pickle (`weights_only=False`) and trusted parameter-table object arrays. Only owner-controlled artifacts should be published; do not accept arbitrary uploaded checkpoints. Consider immutable revisions/hash verification as supply-chain hardening. No checkpoint was altered and `slices/` was not accessed.

## Feedback storage findings

`static/launch.js` handles the form entirely in the browser. FormData becomes a text message, available for copying or downloading. The optional GitHub link opens `https://github.com/krishoncloud/delta-v/issues/new` with the title and body encoded in the query string. The user must publish the issue on GitHub; “Prepare” alone sends no feedback to Delta-V.

There is **no feedback database, private inbox, webhook, or backend submission endpoint** in this repository. Published issues are public in the public repository; GitHub authentication is required to post, and ordinary users cannot silently overwrite other users' issue bodies. This is intentionally public storage, not suitable for private workplace/contact details. The form now states that clearly and both new fields are optional. A private feedback service would require a separately selected destination and access policy.

The generic FormData serializer includes Current workplace and LinkedIn automatically; no fixed storage schema blocks them. LinkedIn validation is an advisory hostname check, not a rejection or remote URL fetch.

Feedback text is not sent to `/events` or a Delta-V backend endpoint, so this application does not log submitted feedback. Copy/download places it in the user's clipboard/local file. Opening the GitHub link may put the body in browser history, GitHub request logs, and browser-sync systems; publishing makes it public. GitHub/Hugging Face internal log retention and account permissions were not administratively inspected. Do not describe this as private or anonymous feedback.

## Real delivery check

Prepared feedback in the live browser with a synthetic workplace and a clearly synthetic LinkedIn profile URL. Submitted that exact generated body to the destination repository using authenticated GitHub API access, fetched it back and compared the complete body exactly, then closed the QA issue. Both optional fields arrived. The public test record remains recoverable/visible at https://github.com/krishoncloud/delta-v/issues/1 — it was closed, not deleted. This verifies form serialization and actual GitHub storage, not a nonexistent automatic/private submission endpoint. Repository metadata confirmed `private: false` and `has_issues: true`.
