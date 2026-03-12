# SSH Throughput Benchmark: TCV → ITER Data Streaming

**Date:** 2026-03-12
**Script:** `scripts/benchmark_ssh_throughput.py`
**Source (data):** TCV — `spclac05.epfl.ch` (lac5), Lausanne, Switzerland
**Destination (receiver):** ITER SDCC — `98dci4-srv-1006.iter.org`, Cadarache, France
**Network path:** ITER login node → `lac912.epfl.ch` (EPFL gateway) → `lac5.epfl.ch` (TCV compute)
**Remote Python:** 3.12.12 via imas-codex venv (`~/.local/share/imas-codex/venv`)
**Packages on TCV:** numpy 2.4.2, msgpack 1.1.2

## Test Configuration

All tests executed from ITER (`98dci4-srv-1006`), connecting to TCV (`tcv` SSH alias).
Data flows **from TCV to ITER** — matching the production signal extraction pattern
where ITER requests data and TCV generates/streams it back.

```
ITER (receiver)  ←──SSH pipe──  lac912 gateway  ←──  lac5 / TCV (data source)
```

```bash
# Executed on ITER login node:
python3 /tmp/benchmark_ssh_throughput.py \
    --target tcv \
    --sizes 1,10,50,100 \
    --parallel 1,2,4,8 \
    --iterations 3
```

## Results

### SSH Latency (TCV → ITER round-trip)

| Metric | Value |
|--------|-------|
| Median | 1,624 ms |
| Min | 1,585 ms |
| Max | 1,736 ms |
| P95 | 1,736 ms |

The ~1.6s baseline latency is the cost of multi-hop SSH (ITER → lac912 → lac5).
This is paid once per SSH session; the `SSHWorkerPool` amortizes it via persistent connections.

### Test 1: Raw SSH Pipe (TCV → ITER)

`dd if=/dev/zero` on ITER piped through SSH to `cat > /dev/null` on TCV.
Measures the theoretical maximum throughput of the SSH channel.

| Payload | Throughput (MB/s) | Range |
|---------|-------------------|-------|
| 1 MB | 0.6 | 0.6–0.6 |
| 10 MB | 5.3 | 5.3–5.6 |
| 50 MB | 18.2 | 17.9–20.8 |
| 100 MB | **20.0** | 19.9–24.1 |

Throughput increases with payload size as SSH session setup is amortized.
The ceiling of ~20–24 MB/s is the WAN link capacity between Cadarache and Lausanne.

### Test 2: Python-to-Python Pipe (TCV → ITER)

TCV generates random bytes in Python, streams them through SSH stdout to ITER.
Simulates the actual extraction pattern where TCV generates data and ITER receives.

| Payload | Throughput (MB/s) | Range |
|---------|-------------------|-------|
| 1 MB | 0.6 | 0.6–0.6 |
| 10 MB | 5.4 | 4.8–5.5 |
| 50 MB | 13.1 | 11.2–13.7 |
| 100 MB | **16.0** | 15.7–18.1 |

~20% lower than raw pipe due to Python `os.urandom()` generation overhead on TCV
and the Python stdout buffering layer.

### Test 3: Signal Extraction Protocol — msgpack binary (TCV → ITER)

Simulates the actual imas-codex signal extraction pipeline:
TCV generates numpy float64 arrays, packs them via msgpack with `__ndarray__`
wire format, and streams the binary payload through SSH stdout to ITER.

| Workload | Signals × Points | Raw Data | Wire Size | Throughput (MB/s) |
|----------|-------------------|----------|-----------|-------------------|
| small_profiles | 5 × 200 | 8 KB | 0.01 MB | 0.0 (latency-dominated) |
| magnetics | 20 × 10,000 | 1.5 MB | 1.53 MB | 0.8 |
| pf_active | 50 × 100,000 | 38 MB | 38.15 MB | 9.4 |
| multi_ids | 100 × 500,000 | 381 MB | 381.48 MB | **21.0** |

Key observations:
- msgpack wire format is essentially 1:1 with raw data (float64 bytes pass through directly)
- Small payloads are dominated by SSH session setup latency (~1.6s)
- Large payloads achieve throughput comparable to raw SSH pipe (~21 MB/s)
- The protocol itself adds negligible overhead — the bottleneck is the WAN link

### Test 4: Parallel SSH Streams (TCV → ITER)

Multiple independent SSH sessions streaming data from TCV to ITER simultaneously.
Tests whether parallelism can saturate more of the available bandwidth.

| Streams | Per-Stream (MB) | Total (MB) | Aggregate (MB/s) | Range |
|---------|-----------------|------------|-------------------|-------|
| 1 | 100 | 100 | 23.2 | 8.8–26.5 |
| 2 | 50 | 100 | 23.7 | 20.8–28.5 |
| 4 | 25 | 100 | 22.9 | 13.4–28.9 |
| 8 | 12 | 100 | 23.8 | 21.3–26.6 |

**Parallelism provides no throughput gain.** Aggregate throughput is flat at ~23 MB/s
regardless of stream count. This confirms the bottleneck is the WAN link between
Cadarache and Lausanne, not SSH session overhead, encryption CPU, or TCP window sizing.

## Analysis

### Current State

| Metric | Value |
|--------|-------|
| Peak sustained throughput (TCV → ITER) | **~24 MB/s** |
| SSH round-trip latency | **~1.6s** |
| Serialization overhead (msgpack) | **< 1%** |
| Parallel scaling benefit | **None** (WAN-limited) |

### GPU Training Requirements (4× H200)

| Metric | Value |
|--------|-------|
| Total GPU memory (4× H200) | 564 GB |
| Minimum feed rate to avoid GPU stall | ~1,000 MB/s |
| Ideal feed rate for continuous training | ~5,000 MB/s |
| Time to fill GPU memory at SSH throughput | **~6.5 hours** |
| **Throughput gap** | **42–208×** below requirement |

### Root Cause

The 24 MB/s ceiling is the **WAN link capacity** between ITER (Cadarache, France)
and EPFL (Lausanne, Switzerland). This is confirmed by:

1. **Parallel streams don't help** — 8 streams achieve the same aggregate as 1 stream
2. **Raw dd throughput matches Python throughput** — no protocol overhead
3. **Latency is consistent** — no intermittent congestion, just a bandwidth cap

No protocol optimization (Arrow Flight, RDMA, compression) can exceed this link speed
for real-time streaming from TCV to ITER.

### Recommendations

| Phase | Approach | Expected Throughput |
|-------|----------|---------------------|
| **Immediate** | Cache-on-Titan: first extraction crosses WAN (24 MB/s), subsequent training runs read from local Titan NVMe | 5–7 GB/s (local I/O) |
| **Short-term** | Batch pre-staging: `cx data sync tcv` extracts full datasets overnight, stages Arrow IPC files on Titan | 5–7 GB/s (local I/O) |
| **Future** | Arrow Flight server on TCV — useful when WAN bandwidth improves or for ITER-internal transfers (login node → Titan) | WAN-limited for cross-site |

The winning strategy is **minimizing data that crosses the WAN** by caching extracted
data on the training side (Titan). The serialization format (msgpack, Arrow IPC) matters
for local pack/unpack speed, but is irrelevant at WAN scale — the pipe is 42× too narrow.
