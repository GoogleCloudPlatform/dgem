## Summary

<!-- Briefly describe what this pull request changes and why. Link related issues (for example, Closes #123). -->

## Surface & Documentation Checklist

- [ ] CLI, HTTP gateway, MCP server, and Decision Studio stay in parity (if adding or changing a decision option).
- [ ] `docs/` and `docs-site/` are synced (`make docs-sync-check` and `python3 scripts/docs_link_check.py`), if documentation changed.
- [ ] No real project IDs, endpoint URLs, keys, or build outputs (`studio/dist/*`) are included.

## Verification

<!-- List the exact commands and tests run to verify this change. -->

- [ ] `go vet ./... && go test -race ./...`
- [ ] `go -C examples vet ./... && go -C examples test -race ./...`
- [ ] Python / Studio / Starlight checks covering touched directories
