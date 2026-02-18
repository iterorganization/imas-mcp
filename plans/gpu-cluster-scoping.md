# Accelerating Scientific Discovery at ITER Through Local GPU Infrastructure

**Science Division — GPU Compute Requirements & Use Cases**
February 2026

## Abstract

ITER represents the largest single investment in fusion energy research. The return on this investment will be measured in scientific discovery, underpinned by vast volumes of experimental data generated across decades of operation. The deep integration of AI across ITER's scientific and engineering activities offers unprecedented returns for an incremental increase in cost. This document presents the Science Division's requirements for a dedicated GPU compute facility based on NVIDIA H200 hardware. Three use cases are presented in order of operational need: (1) document and data embedding for semantic knowledge retrieval, (2) local deployment of open-weight agentic LLMs to remove third-party rate limits that currently constrain our development velocity, and (3) the development and training of a Fusion World Model capable of generative simulation of tokamak plasmas. A local GPU facility would also provide an additional layer of data protection by keeping sensitive information on-site, complementing our continued use of third-party AI services.

## 1. Introduction

The landscape of scientific software development has undergone a fundamental transformation. Historically, physicists and engineers were compelled to act as computer scientists — maintaining fragile software stacks, losing focus on their scientific objectives, or relying on dated analysis tools such as EFIT, built around least-squares fitting of limited diagnostic sets. In the era of autonomous LLM agents, this model is being displaced at remarkable speed. In under eighteen months the field has progressed from a single human steering a single agent, through a user managing independent agent teams on hour-long tasks, to the present day where a user can direct multiple collaborative agent teams — agents that question each other's assumptions and coordinate through an agentic team leader. By harnessing this new capability, we can rapidly bootstrap ourselves into expert users of applied AI, using agentic code generation to build the very infrastructure needed for advanced physics-based AI systems. The purchase of a dedicated GPU facility is the enabling investment that allows ITER to realise this potential at a pace commensurate with the scientific opportunity.

## 2. Current Infrastructure and Capability Gap

The ITER SDCC currently provides two GPU resources:

| Resource | GPU | Count | VRAM | Compute Capability | Status |
|---|---|---|---|---|---|
| Login node | Tesla T4 | 2 | 15 GB each | 7.5 | Consumed by ~80 desktop rendering processes (gnome-shell, SALOME, MATLAB, Firefox). GPU 0 at 73% VRAM utilisation. Currently runs one Qwen3 0.6B embedding model on GPU 1. |
| Titan partition | Tesla P100 | 8 | 16 GB each | 6.0 | Idle. No tensor cores. Cannot run modern LLMs. 128 GB total VRAM insufficient for models >13B parameters at BF16. |

The P100 (2016 architecture) lacks tensor cores entirely — the fundamental hardware unit for efficient LLM inference. The T4 login GPUs can run only the smallest embedding models and are shared with desktop rendering for ~30 concurrent users. Neither resource can run the current generation of open-weight agentic models:

| Model | Architecture | Total / Active Params | Min. VRAM (FP8) | Required GPUs (H200) |
|---|---|---|---|---|
| GLM-5 [1] | MoE | 744B / 40B | ~400 GB | 4 × H200 |
| Kimi K2 [2] | MoE | 1,000B / 32B | ~500 GB | 4 × H200 |
| Qwen3 0.6B (current) | Dense | 0.6B | 1.2 GB | 1 × T4 (current) |

**Proposed acquisition — NVIDIA H200:**

|  | Per GPU | 4-GPU Server |
|---|---|---|
| VRAM (HBM3e) | 141 GB | 564 GB |
| Memory bandwidth | 4.8 TB/s | 19.2 TB/s aggregate |
| FP8 Tensor Core | 3,958 TFLOPS | 15,832 TFLOPS |
| BF16 Tensor Core | 1,979 TFLOPS | 7,916 TFLOPS |
| NVLink | 900 GB/s | Full mesh |

A single 4-GPU H200 server (564 GB VRAM) comfortably serves GLM-5 and Kimi K2 in FP8 with room for concurrent embedding workloads, and provides substantial compute headroom for model training. This represents a >30× increase in effective AI compute over the entire existing ITER GPU estate.

## 3. Use Cases

### 3.1 Document Embedding and Semantic Retrieval

**Need:** Immediate. **Third-party alternative:** Cloud embedding APIs (OpenAI, Cohere) at ~$0.10/M tokens; limited by upload of sensitive internal documents.

ITER's IDM platform hosts tens of thousands of documents (PDFs, DOCX, Excel, images, presentations) that are currently searchable only by metadata. ITER's Lucy chatbot demonstrates the potential of embedding-based retrieval but is limited to document metadata. A local GPU facility would enable full-content embedding of every resource on the IDM platform using open-weight models, so that detailed technical information buried deep within a PowerPoint presentation can be surfaced through natural language search.

A demonstration project is currently operational across three partner tokamaks (JET, TCV, JT-60SA) using the Qwen3 0.6B embedding model to extract meaning from multilingual documents — the JT-60SA facility has significant content written in a mix of Japanese and English, making it difficult for both Japanese and European staff to make progress. Scaling this service to larger, more capable embedding models (4B, 8B parameters) requires GPU VRAM that exceeds the current T4 capacity.

### 3.2 Local Agentic LLM Deployment

**Need:** Immediate. **Third-party alternative:** OpenRouter, Anthropic API, GitHub Copilot — all subject to rate limits.

This is our most impactful near-term use case. We currently rely on third-party providers (GitHub Copilot, Claude Code via Anthropic/OpenRouter) for agentic software development. **The binding constraint is not cost but rate limits.** Third-party providers impose hard caps on concurrent requests that limit parallel agent deployment to 2–3 agents at any time:

- **GitHub Copilot:** Fair-use policy blocks requests when running 3+ agents for any sustained period. Current usage as of 18 February: 1,100% of monthly premium budget consumed.
- **Claude Code / OpenRouter:** Per-minute request limits cap throughput regardless of willingness to pay. Claude's agent teams capability (released February 2026) enables collaborative multi-agent workflows but is immediately throttled by these limits.

**Cost comparison for sustained agent team usage (5 agents, 8 hours/day):**

| Deployment | Monthly Cost | Rate Limited | Concurrent Agents |
|---|---|---|---|
| OpenRouter (Opus 4.6, $15/$75 per M tokens) | ~€3,000–8,000 | Yes, 2–3 effective | Hard cap |
| Local GLM-5 on 4× H200 (~€250K, 5-year life) | ~€4,200/month amortised | No | Limited only by GPU capacity |

The open-weight GLM-5 achieves an intelligence score of 49.5 on standard benchmarks — matching Anthropic's previous best (Opus 4.5) at 53 for the current Opus 4.6 [1]. Local deployment eliminates rate limits entirely: throughput scales linearly with GPU capacity and can be expanded as demand grows. We would be limited only by the size of the system we design and purchase, not by third-party policies.

### 3.3 Fusion World Model Development

**Need:** Medium-term (12–24 months), with infrastructure required now. **Third-party alternative:** Cloud GPU rental (Google Cloud, Lambda Labs) at ~€3–5/GPU-hour for H100; viable for prototyping but prohibitively expensive for sustained training campaigns.

Microsoft's recent Nature publication demonstrates that generative world models, trained on observational data, can learn deep physical laws rather than simple interpolating patterns [3]. Their WHAM model (1.6B parameters) was trained on ~1.4 billion state transitions from gameplay data to generate consistent and diverse future sequences. We propose to develop an analogous Fusion World Model trained on experimental data from ITER's partner tokamaks.

**Available training data from partner facilities (estimated):**

| Facility | Approx. Shots | Avg. Duration | Key Diagnostics | Est. State Transitions |
|---|---|---|---|---|
| JET | ~100,000 | ~5 s | ~200 | ~10B |
| TCV | ~82,000 | ~2 s | ~50 (2,108 signals in graph) | ~800M |
| JT-60SA | ~2,500 | ~10 s | ~80 | ~200M |
| ASDEX-U | ~42,000 | ~5 s | ~150 | ~3B |
| DIII-D | ~200,000 | ~5 s | ~100 | ~10B |
| **Total** | | | | **~24B** |

A 1–2B parameter Fusion World Model trained on ~24 billion state transitions is computationally feasible on a 4× H200 server. Estimated training time: 4–8 weeks for initial convergence, with ongoing refinement as data pipelines mature. Such a model could:

- Generate in-silico simulations of ITER pulses before first plasma, based solely on numerical simulation and partner tokamak data.
- Dream up new operating regimes that extend beyond training data, as the model learns underlying physical laws.
- Identify chains of events leading to off-normal scenarios, enabling operations teams to harden plans before these scenarios become reality.

**Starting this work today is essential.** The competences and infrastructure developed now will be ready for the start of commissioning and SRO. With a suitable pipeline in place, models can be retrained on morning experimental data so that afternoon sessions benefit from updated predictions.

## 4. Recommendation

A 4-GPU NVIDIA H200 server (~€250K) provides a balanced entry point that addresses all three use cases simultaneously: embedding services, agentic LLM inference, and Fusion World Model training. Usage could be monitored and the system scaled (to 8 GPUs or additional nodes) as demand warrants. The investment secures ITER's ability to develop and deploy AI capabilities at a pace set by our scientific ambition rather than by third-party rate limits.

## References

[1] GLM-5 Team. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv:2602.15763, February 2026. https://arxiv.org/abs/2602.15763

[2] Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv:2507.20534, July 2025. https://arxiv.org/abs/2507.20534

[3] Kanervisto, A. et al. "World and Human Action Models towards gameplay ideation." Nature, 2025. https://doi.org/10.1038/s41586-025-08600-3

[4] Bodnar, C. et al. "Aurora: A Foundation Model of the Earth System." arXiv:2405.13063, 2024. https://arxiv.org/abs/2405.13063

[5] NVIDIA. "NVIDIA H200 GPU Specifications." https://www.nvidia.com/en-us/data-center/h200/
