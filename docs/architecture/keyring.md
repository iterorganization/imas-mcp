# Keyring & Credential Management

Credentials for wiki authentication (ITER Confluence, TCV Wiki, JET wikis) are stored in the system keyring via the `imas_codex.discovery.wiki.auth.CredentialManager`.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CredentialManager                         │
│                                                              │
│  Lookup order:                                               │
│  1. In-memory cache (fastest, per-process)                   │
│  2. System keyring (GNOME Keyring via SecretService D-Bus)   │
│  3. Legacy keyring names (auto-migrated on first access)     │
│  4. Environment variables (FACILITY_USERNAME/PASSWORD)        │
│  5. Interactive prompt (if TTY available)                     │
└──────────────────────────────────────────────────────────────┘
```

Credentials are stored as JSON under the keyring service `imas-codex/<site>` with username `credentials`:

```json
{"username": "myuser", "password": "mypass"}
```

Each facility is an independent keyring entry. Session cookies are stored separately under the same service with username `session`.

## CLI Commands

```bash
# Store credentials (interactive prompt for password)
imas-codex credentials set iter
imas-codex credentials set tcv -u myuser

# Check if credentials exist
imas-codex credentials get iter
imas-codex credentials get tcv

# List all configured credential services
imas-codex credentials list
imas-codex credentials list iter          # Single facility

# Delete stored credentials
imas-codex credentials delete iter
imas-codex credentials delete tcv --yes   # Skip confirmation

# Keyring backend status and troubleshooting
imas-codex credentials status
```

## Keyring Backends

| Platform | Backend | D-Bus Required |
|----------|---------|----------------|
| Linux (desktop) | GNOME Keyring via SecretService | Yes |
| Linux (headless) | Environment variables fallback | No |
| macOS | Keychain | No |
| Windows | Credential Locker | No |
| WSL | GNOME Keyring (may need unlock) | Yes |

On Linux, the keyring library uses `secretstorage` which communicates with GNOME Keyring's `org.freedesktop.secrets` D-Bus service.

## SLURM Compute Nodes (Auto D-Bus Forwarding)

SLURM compute nodes (e.g., titan) inherit `DBUS_SESSION_BUS_ADDRESS` from the login node, but the D-Bus daemon on compute nodes does **not** have `org.freedesktop.secrets` registered (no `gnome-keyring-daemon` runs there). This causes keyring to resolve to `fail.Keyring` (priority 0).

### How It Works

When `CredentialManager` detects a non-functional keyring on a SLURM compute node (`SLURM_JOB_ID` is set), it automatically:

1. Discovers the login node hostname via facility config (`compute.login_node.hostname` in the private YAML) or `scontrol`
2. Forwards the login node's D-Bus socket via SSH:
   ```
   ssh -N -L /run/user/<uid>/dbus-forward.sock:/run/user/<uid>/bus <login_node>
   ```
3. Updates `DBUS_SESSION_BUS_ADDRESS` to point to the forwarded socket
4. Resets keyring's internal caches (`keyring.core._keyring_backend` and `keyring.backend.get_all_keyring.reset()`) so it re-discovers the now-viable SecretService backend
5. Registers an `atexit` handler to clean up the SSH process and socket

This is fully transparent — no user action required.

### Verifying on a Compute Node

```bash
# SSH to compute node and check
ssh titan 'cd ~/Code/imas-codex && uv run imas-codex credentials status'
# Expected output:
#   Keyring available: True
#   Backend: Keyring
#   ✓ Keyring is working!

ssh titan 'cd ~/Code/imas-codex && uv run imas-codex credentials list'
# Should show ✓ stored for iter, tcv, jet

ssh titan 'cd ~/Code/imas-codex && uv run imas-codex credentials get iter'
# Should show username and masked password
```

### Manual D-Bus Forwarding

If auto-forwarding fails, you can set it up manually:

```bash
# On the compute node:
ssh -fN -L /run/user/$(id -u)/dbus-forward.sock:/run/user/$(id -u)/bus <login_node>
export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/dbus-forward.sock"

# Verify
python -c "import keyring; print(keyring.get_keyring())"
# Should print: keyring.backends.SecretService.Keyring (priority: 5)
```

### Tunnel CLI Integration

The `tunnel` CLI also supports D-Bus forwarding:

```bash
# Forward D-Bus socket alongside other tunnels
imas-codex tunnel start iter --keyring

# Check D-Bus forward status
imas-codex tunnel status
# Shows: D-Bus keyring forward: /run/user/<uid>/dbus-forward.sock
```

## Adding a New Facility

Adding a new facility does **not** require any keyring reset. Each facility stores credentials under its own independent service name (`imas-codex/<facility>`).

To add credentials for a new facility:

```bash
# Store credentials
imas-codex credentials set <new-facility>

# Verify
imas-codex credentials get <new-facility>
imas-codex credentials list <new-facility>
```

The new facility's credentials are immediately available on the login node and (via auto D-Bus forwarding) on any compute nodes in the same cluster.

If the new facility has a `credential_service` configured in its wiki site config (in `imas_codex/config/facilities/<facility>.yaml`), the credentials CLI will auto-resolve the facility name to the service name.

## Troubleshooting

### `_check_keyring` returns False on login node

**Symptoms:** `credentials status` shows "Keyring not available" on the login node.

**Check D-Bus:**
```bash
echo $DBUS_SESSION_BUS_ADDRESS
# Should be: unix:path=/run/user/<uid>/bus

dbus-send --session --dest=org.freedesktop.DBus \
  --type=method_call --print-reply /org/freedesktop/DBus \
  org.freedesktop.DBus.ListNames | grep -i secret
# Should show: "org.freedesktop.secrets"
```

**Check GNOME Keyring daemon:**
```bash
pgrep -a gnome-keyring-daemon
# Should show at least one process
```

If not running, start it:
```bash
eval $(gnome-keyring-daemon --start --components=secrets)
export GNOME_KEYRING_CONTROL
```

### `_check_keyring` returns False on compute node (auto-forward fails)

**Symptoms:** Log shows "D-Bus forwarding failed" or "Keyring still not functional after D-Bus forward".

**Step 1 — Check SSH access from compute node to login node:**
```bash
ssh -o BatchMode=yes <login_node> echo ok
# Must succeed without password prompt (key-based auth)
```

**Step 2 — Check login node hostname in config:**
```bash
# From compute node
imas-codex config private show iter | grep login_node
# Should show: hostname: <login_node_fqdn>
```

If missing, set it:
```python
update_facility_infrastructure('iter', {
    'compute': {'login_node': {'hostname': '<login_node_fqdn>'}}
})
```

Or let the auto-discovery fall back to scontrol:
```bash
scontrol show config | grep SlurmctldHost
```

**Step 3 — Check for stale forward sockets:**
```bash
ls -la /run/user/$(id -u)/dbus-forward.sock
# If exists but not working, remove it:
rm /run/user/$(id -u)/dbus-forward.sock
# Then retry
```

**Step 4 — Check if the forwarding SSH process is alive:**
```bash
pgrep -af dbus-forward
# Should show the SSH -L forward process
# If not, the SSH connection may have failed
```

### Keyring backend caching after D-Bus forward

The Python keyring library caches its backend list with a `@once` decorator on `get_all_keyring()`. After forwarding the D-Bus socket, two caches must be reset:

```python
import keyring.core
import keyring.backend

keyring.core._keyring_backend = None      # Cached selected backend
keyring.backend.get_all_keyring.reset()   # @once-cached list of viable backends
```

This is handled automatically by `CredentialManager._check_keyring()`. If you're using keyring directly in a script where D-Bus was set up mid-process, you need to reset these caches manually.

### Keyring timeout (3 seconds exceeded)

**Symptoms:** Log shows "Keyring check timed out after 3s".

The keyring backend check runs with a 3-second timeout. On slow networks, the D-Bus forwarding SSH setup (which runs outside this timeout) may succeed but the subsequent keyring query may time out.

**Fix:** The initial backend check and the post-forward re-check each get their own 3-second window. If the D-Bus response is slow, check network latency to the login node:
```bash
ssh -o ConnectTimeout=2 <login_node> echo ok
```

### WSL keyring unlock prompt

On WSL, GNOME Keyring may prompt for an unlock password. If you don't know the password:

```bash
# Reset keyring (deletes all stored credentials — you'll need to re-enter them)
rm -rf ~/.local/share/keyrings/*

# Re-store credentials
imas-codex credentials set iter
imas-codex credentials set tcv
imas-codex credentials set jet
```

### Environment variable fallback

When keyring is unavailable, use environment variables:

```bash
export ITER_USERNAME=myuser
export ITER_PASSWORD=mypass
export TCV_USERNAME=myuser
export TCV_PASSWORD=mypass
export JET_USERNAME=myuser
export JET_PASSWORD=mypass
```

These are checked after keyring in the credential lookup chain.

## Internals

### Service naming

| Facility | Keyring Service | Legacy Name (auto-migrated) |
|----------|----------------|-----------------------------|
| iter | `imas-codex/iter` | `imas-codex/iter-confluence` |
| tcv | `imas-codex/tcv` | `imas-codex/tcv-wiki` |
| jet | `imas-codex/jet` | `imas-codex/jet-wiki` |
| jt-60sa | `imas-codex/jt-60sa` | `imas-codex/jt-60sa-wiki` |

Legacy names are auto-migrated on first access — the old entry is copied to the new name and deleted.

### D-Bus socket paths

| Location | Path |
|----------|------|
| Login node D-Bus | `/run/user/<uid>/bus` |
| Forwarded socket (primary) | `/run/user/<uid>/dbus-forward.sock` |
| Forwarded socket (fallback) | `/tmp/dbus-forward-<uid>.sock` |

The forwarded socket path is deterministic so multiple processes on the same compute node share one SSH forward (idempotent).

### Keyring cache reset sequence

When the D-Bus environment changes mid-process (e.g., after socket forwarding), the keyring library's backend discovery must be fully reset:

```
1. keyring.core._keyring_backend = None
   └── Clears the cached "selected backend" singleton

2. keyring.backend.get_all_keyring.reset()
   └── Clears the @once-decorated viable backend list
       (jaraco.functools.once stores result in wrapper.saved_result)

3. keyring.get_keyring()
   └── Triggers init_backend() → _detect_backend()
       → get_all_keyring() (now re-evaluates)
       → SecretService.priority (now succeeds via forwarded D-Bus)
       → Returns SecretService.Keyring (priority: 5)
```

Without step 2, `get_all_keyring()` returns the stale list containing only `fail.Keyring`, even though `SecretService.Keyring.viable` would now return True.
