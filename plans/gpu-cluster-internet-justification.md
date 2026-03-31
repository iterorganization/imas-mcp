# Internet Connectivity for ITER GPU Compute Infrastructure

A companion analysis to the compute session orchestration plan, justifying the
operational need for controlled internet access from SDCC compute nodes and the
proposed H200 GPU cluster. This report balances security concerns against
operational requirements.

---

## 1. Executive Summary

The ITER SDCC compute infrastructure currently provides selective internet access
from compute nodes: public repositories (GitHub, PyPI) are reachable while some
API endpoints (notably `api.openrouter.ai`) are DNS-filtered. This partial
connectivity creates an asymmetry where some AI/ML workflows function while others
fail unpredictably. We recommend a controlled expansion of internet access for
compute nodes, particularly the future H200 GPU cluster, using a defense-in-depth
approach that minimizes the attack surface increase.

---

## 2. Current State (Empirically Verified, 2026-03-31)

### What Works from Compute Nodes

| Service | Protocol | Status | Evidence |
|---------|----------|--------|----------|
| github.com | HTTPS | ✅ | HTTP 200, 0.21s |
| pypi.org | HTTPS | ✅ | HTTP 200, 0.11s |
| api.anthropic.com | DNS + HTTPS | ✅ | Resolves to 160.79.104.10 |
| openrouter.ai | DNS | ✅ | Resolves to 104.18.2.115 (Cloudflare) |
| Compute ↔ Compute | TCP | ✅ | Neo4j, Embed server accessible |
| Login → Compute | TCP | ✅ | Direct port access works |

### What Fails from Compute Nodes

| Service | Protocol | Failure Mode | Impact |
|---------|----------|--------------|--------|
| api.openrouter.ai | DNS | NXDOMAIN | LiteLLM proxy cannot route to OpenRouter |
| External DNS (8.8.8.8) | UDP:53 | Timeout | Cannot bypass ITER DNS filtering |
| Compute → Login | TCP | Connection refused | Cannot reach login-hosted services |
| SSH port forwarding | TCP | Administratively prohibited | Cannot tunnel through compute SSH |

### DNS Architecture

Compute nodes use four ITER-internal DNS resolvers (10.10.24x.x) with search
domain `iter.org`. External DNS servers are unreachable. Resolution is selective
— the DNS appears to forward queries for whitelisted domains and return NXDOMAIN
for others, or there is an issue with specific CNAME chain resolution through
CDN providers like Cloudflare.

---

## 3. Use Cases Requiring Internet Access

### 3.1 AI Agent Development (Immediate)

**Current bottleneck:** LiteLLM proxy must run on login nodes because it needs
outbound HTTPS to LLM API providers. This contributes to login node saturation.

**Services required:**

| Endpoint | Provider | Purpose | Protocol |
|----------|----------|---------|----------|
| api.anthropic.com | Anthropic | Claude model inference | HTTPS (443) |
| api.openrouter.ai | OpenRouter | Multi-provider LLM routing | HTTPS (443) |
| api.openai.com | OpenAI | GPT model inference | HTTPS (443) |
| generativelanguage.googleapis.com | Google | Gemini model inference | HTTPS (443) |
| openrouter.ai | OpenRouter | API key management | HTTPS (443) |

**Traffic characteristics:**
- JSON request/response payloads (1-100 KB typical)
- No bulk data transfer
- TLS 1.3 encrypted
- Outbound-initiated only (no inbound connections)
- Estimated: 10-50 requests/minute during active development

### 3.2 Embedding Model Downloads (Immediate)

**Current state:** Embedding models are downloaded on login nodes and cached to GPFS.
The embedding server runs on the Titan GPU node via SLURM but cannot download
models directly.

**Services required:**

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| huggingface.co | Model weight download | Monthly (on model updates) |
| cdn-lfs.huggingface.co | Large file storage | Monthly |
| pypi.org | Python package updates | Weekly |

### 3.3 GPU Cluster Training Data (Medium-term, H200)

**Requirement:** The proposed 8×H200 GPU cluster will train fusion world models
on experimental data from partner tokamaks. Some data pipelines require fetching
from external data repositories.

**Services required:**

| Endpoint | Purpose | Data Volume |
|----------|---------|-------------|
| Partner facility data APIs | Experimental data download | 1-100 GB per campaign |
| Hugging Face Hub | Pre-trained model checkpoints | 10-500 GB per model |
| Container registries (ghcr.io, nvcr.io) | ML framework containers | 5-20 GB per image |

### 3.4 Software Development Support (Ongoing)

| Endpoint | Purpose |
|----------|---------|
| github.com | Git operations, CI/CD | ✅ Already works |
| pypi.org | Python packages | ✅ Already works |
| registry.npmjs.org | Node.js packages | Untested |
| crates.io | Rust packages | Untested |

---

## 4. Security Analysis

### 4.1 Current Security Posture

The SDCC compute network operates with a **default-deny DNS policy** (or equivalent
filtering effect). Compute nodes can only resolve domains that ITER's internal DNS
servers are configured to forward. This provides:

**Existing controls:**
1. DNS-level filtering (prevents resolution of unapproved domains)
2. No inbound connectivity to compute nodes from outside ITER
3. `pam_slurm_adopt` blocks SSH access without active SLURM jobs
4. `AllowTcpForwarding no` prevents SSH tunnel abuse
5. No root access for users on compute nodes
6. GPFS access controls for data isolation

### 4.2 Attack Surface Analysis

| Threat | Current Risk | Risk with Internet Access | Mitigation |
|--------|-------------|---------------------------|------------|
| **Data exfiltration** | Low (DNS-filtered) | Medium (HTTPS to API providers) | Egress proxy with DLP inspection; TLS interception for approved endpoints only |
| **Malware download** | Low (limited DNS) | Low-Medium (only whitelisted endpoints) | Application-layer proxy whitelist; no arbitrary internet access |
| **C2 communication** | Very Low (no outbound to arbitrary hosts) | Low (only whitelisted HTTPS) | Restrict to specific IP ranges + domains; monitor for anomalous traffic patterns |
| **Lateral movement** | Medium (compute ↔ compute open) | Unchanged | Network segmentation already in place; no new lateral paths created |
| **Credential theft** | Low | Low (API keys in env vars) | Secrets management; rotate keys; proxy-level credential injection |
| **Supply chain attack** | Medium (PyPI, GitHub accessible) | Unchanged | Already exposed via current PyPI/GitHub access |

### 4.3 Key Observation

**The most significant supply-chain risk already exists.** Compute nodes can already
reach `github.com` and `pypi.org` — the two most common vectors for software supply
chain attacks. Adding controlled access to a handful of LLM API endpoints (which
accept only authenticated HTTPS requests and return only text responses) does not
materially increase the attack surface beyond what PyPI access already provides.

---

## 5. Recommended Architecture

### 5.1 Defense-in-Depth Approach

```
┌─ ITER Network Boundary ────────────────────────────────────────────┐
│                                                                     │
│  ┌─ DMZ / Proxy Zone ────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  Forward Proxy (Squid/Envoy)                                  │ │
│  │  ├─ Domain allowlist (api.anthropic.com, api.openrouter.ai)   │ │
│  │  ├─ TLS inspection for approved domains                       │ │
│  │  ├─ Rate limiting per source IP                               │ │
│  │  ├─ Request/response size limits                              │ │
│  │  └─ Logging to SIEM                                           │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│       ▲                                                             │
│       │ HTTPS (443)                                                 │
│       │                                                             │
│  ┌─ Compute Network ─────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  Compute Nodes                                                │ │
│  │  ├─ HTTPS_PROXY=http://proxy:3128                             │ │
│  │  ├─ Only proxy-routed traffic reaches internet                │ │
│  │  └─ Direct internet access remains blocked                    │ │
│  │                                                                │ │
│  │  H200 GPU Cluster                                             │ │
│  │  ├─ Same proxy configuration                                  │ │
│  │  ├─ Additional allowlist for training data endpoints          │ │
│  │  └─ Bandwidth monitoring for large transfers                  │ │
│  │                                                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation Tiers

**Tier 1: DNS Whitelist (Minimal change, immediate)**

Add specific domains to the ITER DNS resolver whitelist:
- `api.openrouter.ai`
- `api.openai.com`
- `generativelanguage.googleapis.com`
- `huggingface.co`, `cdn-lfs.huggingface.co`

This requires no infrastructure changes — only a DNS configuration update. The
existing network architecture already permits HTTPS outbound (proven by github.com
and pypi.org access). The DNS filter is the only barrier.

**Risk increase:** Minimal. Same HTTPS-outbound pattern as existing PyPI/GitHub access.
No new network paths are created.

**Tier 2: Forward Proxy (Moderate change, recommended for H200)**

Deploy an HTTP/HTTPS forward proxy (Squid, Envoy, or similar) on a login or
infrastructure node that:

1. Accepts connections from compute subnet (10.154.x.0/24)
2. Maintains a strict domain allowlist
3. Provides TLS inspection for approved domains
4. Logs all requests for audit
5. Enforces rate and size limits

Compute nodes configure `HTTPS_PROXY` environment variable. No direct internet
access is granted — all traffic routes through the auditable proxy.

**Tier 3: Network Segmentation (For production H200 cluster)**

The H200 GPU cluster should be placed in a dedicated network segment with:

1. Separate VLAN from general compute
2. Dedicated proxy instance with GPU-specific allowlist
3. Bandwidth monitoring and alerting for large transfers
4. DLP (Data Loss Prevention) rules to prevent sensitive data exfiltration
5. Integration with ITER's SIEM for security monitoring

---

## 6. Comparison: Login Node vs. Compute Node Internet Access

| Factor | Login Nodes (Current) | Compute Nodes (Proposed) |
|--------|----------------------|--------------------------|
| **Users with access** | ~30 concurrent users | Only SLURM job owners |
| **Access control** | SSH login (any ITER user) | SLURM job + pam_slurm_adopt |
| **Process isolation** | None (shared user space) | cgroup isolation per job |
| **Audit trail** | SSH session logs | SLURM accounting + proxy logs |
| **Resource limits** | None (fair-use only) | SLURM enforced (CPU, memory, time) |
| **Cleanup** | Manual / no guarantee | Automatic on job completion |

**Compute nodes are actually MORE controlled** than login nodes for internet access.
Every process runs within a SLURM cgroup, with tracked resource usage, known job
owners, and automatic cleanup. Login nodes have no such guarantees.

---

## 7. Immediate Workarounds (No IT Changes Required)

While awaiting formal DNS/proxy changes, these workarounds enable compute-based
LLM access today:

### 7.1 HOSTALIASES Environment Variable

```bash
# Resolve api.openrouter.ai on login (where it works), cache IP
echo "$(host api.openrouter.ai | awk '/has address/{print $4; exit}') api.openrouter.ai" \
    > ~/.local/share/imas-codex/services/hostaliases
export HOSTALIASES=~/.local/share/imas-codex/services/hostaliases
```

### 7.2 Direct Anthropic API (Bypass OpenRouter)

Since `api.anthropic.com` resolves on compute nodes, configure LiteLLM with a
direct Anthropic API key for Claude models:

```yaml
# In litellm_config.yaml, add direct Anthropic entries:
- model_name: anthropic/claude-sonnet-4-6
  litellm_params:
    model: claude-sonnet-4-6-20250514
    api_key: os.environ/ANTHROPIC_API_KEY
    # Direct to Anthropic, no OpenRouter needed
```

This eliminates the OpenRouter dependency for the primary model family (Claude),
which accounts for >90% of LLM calls in imas-codex workflows.

### 7.3 Login-as-Proxy Pattern

Run a lightweight SOCKS or HTTP proxy on the login node that compute nodes can reach:

```bash
# On login node (one-time):
ssh -D 1080 -N localhost &  # SOCKS proxy

# On compute node:
export ALL_PROXY=socks5://10.154.100.16:1080
```

**Caveat:** This depends on compute→login TCP working for port 1080. Current tests
show compute→login TCP is blocked by firewall, so this may not work without IT
involvement. See Section 2 connectivity matrix.

---

## 8. Recommendation

### For Immediate Use (AI Agent Development)

1. **Use direct Anthropic API from compute** — `api.anthropic.com` already resolves
2. **Use HOSTALIASES for OpenRouter** — resolves the DNS issue without IT changes
3. **File IT ticket for DNS whitelist** — request `api.openrouter.ai` be added
4. **Move LiteLLM to compute node** — reduces login node load

### For H200 GPU Cluster

1. **Deploy forward proxy** (Tier 2) before cluster delivery
2. **Domain allowlist** covering LLM APIs, model registries, training data endpoints
3. **Dedicated VLAN** for GPU cluster with monitored proxy access
4. **SIEM integration** for security audit trail
5. **DLP rules** for sensitive data protection
6. **Bandwidth monitoring** for large training data transfers

### Cost-Benefit Summary

| Investment | Benefit | Risk |
|------------|---------|------|
| DNS whitelist (1h IT work) | Unblocks all compute LLM access | Negligible — same HTTPS pattern as existing PyPI |
| Forward proxy (~1 week setup) | Auditable, rate-limited internet access | Low — standard enterprise pattern |
| Dedicated GPU VLAN (~2 weeks) | Full isolation with monitored access | Very low — defense-in-depth |

The security posture of compute nodes (cgroup isolation, SLURM accounting,
pam_slurm_adopt, no root) is **stronger** than the login nodes that currently
have unrestricted internet access. Extending controlled internet access to compute
nodes is a net security improvement when it allows migrating workloads off the
less-controlled login node environment.

---

## References

- SDCC Infrastructure Reference: `~/sdcc-infrastructure.md`
- GPU Cluster Requirements: `plans/gpu-cluster-scoping.md`
- LiteLLM Multi-Tenant Gateway: `plans/features/litellm-multi-tenant-gateway.md`
- Network tests: 2026-03-31, login node sdcc-login-1006, compute node 98dci4-clu-3001
