# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Entries for new releases are generated with `cz bump`.

## [Unreleased]

## 0.1.0 (2026-10-23)

### Added

- `ACGSGuard` with `guard.tool_hook`: an Agno tool hook that checks each tool call's name and arguments against
  an ACGS constitution before the tool runs, and returns the denial reason as the tool result. It works in
  `run()`, `arun()` and both streaming modes.
- `guard.guardrail`: an Agno guardrail that stops a run before the model is called when the agent or team uses
  tools the hook cannot check (provider-run tools, externally executed tools, members without the guard, tool or
  member factories) or when the guard is not the last tool hook.
- Review handling: `on_review="deny"` (default), or `on_review="confirmed"`, where Agno's confirmation flow
  counts as the review.
- `allow_ungoverned=True` and `tool_call_formatter=` options; `ACGSGuard.check`, `ungoverned_tools`,
  `evaluate` and `decide`.
