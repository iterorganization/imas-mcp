## ⚠️ Safety Policy for Remote Execution

### Read-Only Policy

**CRITICAL**: Remote facilities are READ-ONLY. You must NOT:

- **Modify files**: No `mv`, `rm`, `cp`, `touch`, `chmod`, `chown`
- **Edit content**: No `sed -i`, `vim`, `nano`, `echo >`, `cat >`
- **Create files**: No `mkdir`, `touch` except in `~/.cache/` or `~/tmp/`
- **Change state**: No `git commit`, `git push`, database writes

**Exceptions** (home directory only):

- Install utilities to `~/.local/bin/` or `~/bin/` using cargo, pip --user
- Create temporary working files in `~/tmp/` or `~/.cache/`

If you need to modify facility data, report findings and request human intervention.

### Resource Limits

- **Timeout**: Commands should complete in < 60 seconds
- **Scope**: Always specify path arguments for `rg`, `fd`, `find`
  - ✅ `rg 'pattern' /work/codes`
  - ❌ `rg 'pattern'` (searches cwd, may be huge)
- **Pagination**: Use `| head -100` or `--max-count` for discovery
- **Large results**: If > 500 items, summarize instead of listing

### Error Handling

- **SSH failures**: Report error and suggest checking connectivity
- **Timeouts**: Note the timeout and suggest narrower scope
- **Permission denied**: Skip and document, don't retry with sudo

### Credential Safety

- **Never echo** passwords, tokens, or API keys
- **Never include** credentials in command strings
- **Never store** credentials in exploration notes
