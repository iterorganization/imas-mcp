# Multi-Site Wiki Scraping with Secure Credentials

## Overview

This implementation enables scraping wiki/documentation sites from multiple facilities with secure credential management. Credentials are stored in your system keyring (GNOME Keyring on Linux, Keychain on macOS) and never written to disk in plain text.

## Architecture

### Components

1. **CredentialManager** (`imas_codex/wiki/auth.py`)
   - Secure credential storage using system keyring
   - Fallback to environment variables for CI/automation
   - Interactive prompts for first-time setup
   - Session cookie persistence with TTL

2. **ConfluenceClient** (`imas_codex/wiki/confluence.py`)
   - REST API client for Atlassian Confluence
   - Session-based authentication with cookie caching
   - Space/page enumeration, content retrieval, search
   - Rate limiting and automatic session restoration

3. **WikiConfig** (refactored in `imas_codex/wiki/discovery.py`)
   - Loads from facility YAML `wiki_sites` list
   - Supports multiple site types: mediawiki, confluence, generic
   - Supports multiple auth types: none, ssh_proxy, basic, session
   - Falls back to hardcoded defaults for backward compatibility

4. **WikiDiscovery** (enhanced in `imas_codex/wiki/discovery.py`)
   - Delegates to site-specific link extraction
   - Confluence: Uses REST API for page/attachment discovery
   - MediaWiki: Uses SSH + curl for link extraction

### Facility Configuration

Wiki sites are configured in facility YAML files (e.g., `imas_codex/config/facilities/iter.yaml`):

```yaml
wiki_sites:
  - url: https://confluence.iter.org
    portal_page: IMP
    site_type: confluence
    auth_type: session
    credential_service: iter-confluence
```

## Setup Instructions

### 1. Verify Keyring is Available

```bash
# Check if keyring is working
uv run python -c "import keyring; print(keyring.get_keyring())"
```

Expected output: `keyring.backends.SecretService.Keyring (priority: 5)` on Linux

### 2. Store Credentials

```bash
# For ITER Confluence
uv run imas-codex wiki credentials set iter-confluence
# Prompts for username and password

# Verify credentials are stored
uv run imas-codex wiki credentials get iter-confluence
```

### 3. List Configured Sites

```bash
# Show all wiki sites for a facility
uv run imas-codex wiki sites iter
uv run imas-codex wiki sites epfl
```

## Usage

### Discover Wiki Content

```bash
# Full discovery pipeline (crawl + score)
uv run imas-codex wiki discover iter

# Crawl only (no LLM scoring)
uv run imas-codex wiki crawl iter

# Score crawled pages
uv run imas-codex wiki score iter --cost-limit 20.0

# Ingest high-score pages
uv run imas-codex wiki ingest iter
```

### Manage Credentials

```bash
# Set credentials for a site
uv run imas-codex wiki credentials set iter-confluence

# Check if credentials exist
uv run imas-codex wiki credentials get iter-confluence

# Delete credentials
uv run imas-codex wiki credentials delete iter-confluence --yes
```

## Credential Security

### Storage Locations (Priority Order)

1. **System Keyring** (Recommended)
   - GNOME Keyring on Linux (via SecretService D-Bus)
   - Keychain on macOS
   - Credential Locker on Windows
   - Encrypted at rest, never written to disk

2. **Environment Variables** (CI/Automation)
   - `ITER_CONFLUENCE_USERNAME`
   - `ITER_CONFLUENCE_PASSWORD`
   - Use for CI/CD pipelines

3. **Interactive Prompt** (First-time Setup)
   - Prompts user if credentials not found
   - Offers to save to keyring

### Why Keyring?

- **Encrypted at rest**: Uses AES-128 (GNOME) or AES-256 (macOS)
- **Per-user**: Each user has separate credentials
- **Session-based**: Credentials survive across sessions
- **No plain text**: Never written to `.env` or config files
- **Azure AD compatible**: Works with SSO tokens if Confluence uses OAuth

## Confluence Authentication

### Supported Methods

1. **Basic Auth** (Username/Password)
   - Store in keyring
   - Credentials used for each API call

2. **API Tokens** (Recommended)
   - Generate in Confluence settings
   - Store token as "password" in keyring
   - More secure than user password

3. **Session Cookies** (Automatic)
   - Cached in keyring after login
   - TTL: 8 hours (configurable)
   - Avoids repeated authentication

### Setup for ITER Confluence

```bash
# 1. Generate API token in Confluence
# Settings → Personal Settings → API Tokens → Create API Token

# 2. Store credentials
uv run imas-codex wiki credentials set iter-confluence
# Username: your_username
# Password: your_api_token

# 3. Verify
uv run imas-codex wiki credentials get iter-confluence

# 4. Start discovery
uv run imas-codex wiki discover iter
```

## Facility Configuration Examples

### ITER (Confluence)

```yaml
# imas_codex/config/facilities/iter.yaml
facility: iter
name: ITER Organization
machine: ITER
description: International Thermonuclear Experimental Reactor
location: Cadarache, France

data_systems:
  - imas
  - mdsplus

wiki_sites:
  - url: https://confluence.iter.org
    portal_page: IMP
    site_type: confluence
    auth_type: session
    credential_service: iter-confluence
```

### EPFL (MediaWiki)

```yaml
# imas_codex/config/facilities/epfl.yaml
facility: epfl
name: École Polytechnique Fédérale de Lausanne
machine: TCV
description: Swiss Plasma Center - TCV Tokamak
location: Lausanne, Switzerland

data_systems:
  - mdsplus
  - tdi

wiki_sites:
  - url: https://spcwiki.epfl.ch/wiki
    portal_page: Portal:TCV
    site_type: mediawiki
    auth_type: ssh_proxy
    ssh_host: epfl
```

## Troubleshooting

### Keyring Not Available

```
❌ System keyring not available.
Keyring requires a running D-Bus session.
```

**Solution**: On headless systems, use environment variables:

```bash
export ITER_CONFLUENCE_USERNAME=your_username
export ITER_CONFLUENCE_PASSWORD=your_api_token
```

### Session Expired

```
Error: Authentication required or session expired
```

**Solution**: Delete cached session and re-authenticate:

```bash
uv run imas-codex wiki credentials delete iter-confluence
uv run imas-codex wiki credentials set iter-confluence
```

### Confluence API Errors

```
Error: 401 Unauthorized
```

**Possible causes**:
- Invalid credentials
- API token expired
- User account disabled
- IP whitelist restriction

**Solution**: Verify credentials in Confluence settings and regenerate API token if needed.

## Implementation Details

### Credential Lookup Chain

```python
# 1. Try keyring
if keyring_available:
    creds = keyring.get_password(service, "credentials")

# 2. Try environment variables
username = os.environ.get("ITER_CONFLUENCE_USERNAME")
password = os.environ.get("ITER_CONFLUENCE_PASSWORD")

# 3. Interactive prompt
if sys.stdin.isatty():
    username = input("Username: ")
    password = getpass.getpass("Password: ")
```

### Session Persistence

```python
# After successful authentication
session_data = {
    "cookies": session.cookies,
    "expires_at": time.time() + 8 * 3600,  # 8 hours
}
keyring.set_password(service, "session", json.dumps(session_data))

# On next run
session_data = json.loads(keyring.get_password(service, "session"))
if time.time() < session_data["expires_at"]:
    session.cookies.update(session_data["cookies"])
```

### Site Type Detection

```python
def detect_site_type(url: str) -> str:
    # Confluence indicators
    if "confluence" in url.lower():
        return "confluence"
    if "/rest/api" in url:
        return "confluence"
    
    # MediaWiki indicators
    if "/wiki/" in url and "Special:" in url:
        return "mediawiki"
    if "/api.php" in url:
        return "mediawiki"
    
    # Default
    return "generic"
```

## Future Enhancements

1. **OAuth/SAML Support**: For Azure AD integration
2. **Multi-user Credentials**: Shared credential store for teams
3. **Credential Rotation**: Automatic token refresh
4. **Audit Logging**: Track credential access
5. **Encrypted Local Cache**: For offline access
6. **Proxy Support**: For restricted networks

## References

- [Python keyring documentation](https://keyring.readthedocs.io/)
- [Confluence REST API](https://developer.atlassian.com/cloud/confluence/rest/v2/)
- [GNOME Keyring](https://wiki.gnome.org/Projects/GnomeKeyring)
- [macOS Keychain](https://support.apple.com/en-us/HT204085)
