# Accelerating Scientific Discovery at ITER Through Local GPU Infrastructure

Science Division, GPU Compute Requirements and Use Cases

Author: Simon McIntosh
Co-author: Simon Pinches
Date: 18/02/2026

## Abstract

ITER represents the largest single investment in fusion energy research. The return on that investment is scientific discovery, backed by vast volumes of experimental data. For a relatively modest additional expenditure, the deep integration of AI across ITER's scientific and engineering activities can substantially increase these returns. This document presents the Science Division's requirements and use cases for an on-site GPU compute facility. The acquisition is driven by two corporate objectives on agentic simulation and experimental data mapping whose delivery depends on GPU infrastructure that ITER does not currently possess. Three use cases are presented in order of operational need: first, a multilingual embedding service for semantic knowledge retrieval and IMAS data mapping; second, a local deployment of open-weight agentic LLMs to remove third-party rate limits that currently constrain our development velocity; and third, the development of a Fusion World Model capable of generative simulation of plasmas. The timing of this investment is important. The infrastructure, competences, and AI pipelines developed now will be mature and tested by the time ITER produces its first plasma. A local GPU facility will also provide an additional layer of data protection by keeping sensitive information on-site, complementing our continued use of corporate AI services such as GitHub Copilot for non-sensitive work.

## 1. Introduction

This document is prepared by the Science Division to set out our requirements for GPU compute infrastructure and the scientific use cases that motivate its acquisition. Our focus is on maximising the scientific return of the ITER investment through the integration of AI across our activities.

The Science Division has two corporate objectives for 2026 that directly motivate this acquisition: first, to demonstrate the creation of an AI agent that can autonomously run an IMAS physics code given natural language requests; and second, to develop AI capabilities to understand, label, and map experimental data for scientific analysis, including data from ITER Members and the Magnet Cold Test Facility. Both objectives depend on GPU compute that the existing SDCC estate cannot provide. Both also sit within the Science and Integration Department alongside the IT Division that would procure and host this infrastructure, providing a natural alignment of scientific ambition and technical support.

The landscape of scientific software development has changed significantly over the past two years. Historically, physicists and engineers were compelled to act as computer scientists, maintaining fragile software stacks, losing focus on their scientific objectives, or relying on dated analysis tools such as EFIT, built around least-squares fitting of limited diagnostic sets. Autonomous LLM agents are displacing this model at a considerable pace. In under eighteen months the field has progressed from a single human steering a single agent, through a user managing independent agents on clearly defined tasks, to the present day where a user can direct multiple collaborative agent teams to work towards acomplising stratigic goals laid out by the user. Individual agent team members can question each other's assumptions and coordinate through an agentic team leader. By harnessing this capability, we can bootstrap ourselves into expert users of applied AI, using agentic code generation to build the infrastructure needed for more advanced physics-based AI systems.

The window for preparation is now. Commissioning and the start of research operations (SRO) define a hard deadline by which these capabilities must be mature, tested, and operational. A purchase today enables the robust application of AI from the very first day of plasma operations, ensuring that ITER's scientific data is exploited to a standard commensurate with the scale of the investment made by the IO and its partner Domestic Agencies.

## 2. Current Infrastructure and Capability Gap

The ITER SDCC currently provides two GPU resources:

| Resource | GPU | Count | VRAM | Compute Capability | Status |
|---|---|---|---|---|---|
| Login node | Tesla T4 | 2 | 15 GB each | 7.5 | Consumed by ~80 desktop rendering processes (gnome-shell, SALOME, MATLAB, Firefox). GPU 0 at 73% VRAM utilisation. Currently runs one Qwen3 0.6B embedding model on GPU 1. |
| Titan partition | Tesla P100 | 8 | 16 GB each | 6.0 | Idle. No tensor cores. Cannot run modern LLMs. 128 GB total VRAM insufficient for models >13B parameters at BF16. |

The P100 (2016 architecture) lacks tensor cores entirely, the fundamental hardware unit for efficient LLM inference. The T4 login GPUs can run only the smallest embedding models and are shared with desktop rendering for approximately 30 concurrent users. Neither resource can run the current generation of open-weight agentic models:

| Model | Architecture | Total / Active Params | Min. VRAM (FP8) | Required GPUs (H200) |
|---|---|---|---|---|
| GLM-5 [1] | MoE | 744B / 40B | ~400 GB | 4 x H200 |
| Kimi K2 [2] | MoE | 1,000B / 32B | ~500 GB | 4 x H200 |
| Qwen3 0.6B (current) | Dense | 0.6B | 1.2 GB | 1 x T4 (current) |

Proposed acquisition, NVIDIA H200:

|  | Per GPU | 4-GPU Server |
|---|---|---|
| VRAM (HBM3e) | 141 GB | 564 GB |
| Memory bandwidth | 4.8 TB/s | 19.2 TB/s aggregate |
| FP8 Tensor Core | 3,958 TFLOPS | 15,832 TFLOPS |
| BF16 Tensor Core | 1,979 TFLOPS | 7,916 TFLOPS |
| NVLink | 900 GB/s | Full mesh |

A single 4-GPU H200 server (564 GB VRAM) can serve GLM-5 and Kimi K2 in FP8 with room for concurrent embedding workloads, and provides substantial compute headroom for model training. This represents a greater than 30x increase in effective AI compute over the entire existing ITER GPU estate.

## 3. Use Cases

### 3.1 Multilingual Embedding and IMAS Data Mapping

Need: Immediate, directly enables Use Case 3.3. Third-party alternative: cloud embedding APIs (OpenAI, Cohere) at ~$0.10/M tokens; limited by upload of sensitive internal documents and inability to run custom models.

The IMAS applications being developed for ITER operations must be validated against real experimental data before commissioning begins. The ITER Research Plan is being actively refined using these tools now, and partner tokamak data provides the only available source of real experimental measurements for this validation. Each partner facility, however, stores its data in its own native format, using facility-specific naming conventions, coordinate systems, and access methods. Converting this heterogeneous data into a unified IMAS representation is therefore a prerequisite for validation and, by extension, for operational readiness. The investment in embedding infrastructure is needed now precisely because this validation work cannot wait for ITER's own experimental data.

A demonstration project is currently operational across three partner tokamaks (JET, TCV, JT-60SA) using the Qwen3 0.6B embedding model deployed on the ITER login node's single T4 GPU. The project uses LLM agents to discover and extract data mappings from native facility-specific formats to the standard IMAS data model used at ITER. This is a multilingual challenge: JT-60SA has significant technical content written in a mix of Japanese and English; TCV's documentation is partly in French; and future mapping activities will extend to EAST (China), KSTAR (Korea), and ASDEX Upgrade (Germany), each with documentation in their respective languages. We deploy multilingual embedding models so that the semantic meaning of source documents is preserved without translation loss. A Japanese equipment specification and its English-language IMAS counterpart are mapped into the same vector space, enabling automated discovery of correspondences that would otherwise require bilingual domain experts. The provision of partner tokamak data in a unified IMAS format is a necessary prerequisite for training the Fusion World Model detailed in Section 3.3. Scaling this service to larger, more capable embedding models (4B, 8B parameters) requires GPU VRAM that exceeds the current T4 capacity.

While not a direct concern of the Science Division, we note that the same embedding infrastructure would immediately benefit ITER's IDM platform. IDM hosts tens of thousands of documents (PDFs, DOCX, Excel, images, presentations) that are currently searchable only by metadata. ITER's Lucy chatbot demonstrates the potential of embedding-based retrieval but is limited to document metadata. A local GPU facility would enable full-content embedding of every resource on IDM using open-weight models, so that detailed technical information buried within a PowerPoint presentation can be found through natural language search. This remains a key and immediate motivating use case for the acquisition of the cluster across multiple ITER departments.

### 3.2 Local Agentic LLM Deployment

Need: Immediate. Third-party alternative: OpenRouter, Anthropic API, GitHub Copilot, all subject to rate limits.

This is our most impactful near-term use case. We currently rely on third-party providers (GitHub Copilot, Claude Code via Anthropic/OpenRouter) for agentic software development. The binding constraint is not cost but rate limits. Third-party providers impose hard caps on concurrent requests that limit parallel agent deployment to two or three agents at any time:

- GitHub Copilot: Fair-use policy blocks requests when running three or more agents for any sustained period. Current usage as of 18 February: 1,100% of monthly premium budget consumed.
- Claude Code / OpenRouter: Per-minute request limits cap throughput regardless of willingness to pay. Claude's agent teams capability (released February 2026) enables collaborative multi-agent workflows but is immediately throttled by these limits.

Cost comparison for sustained agent team usage (5 agents, 8 hours/day):

| Deployment | Monthly Cost | Rate Limited | Concurrent Agents |
|---|---|---|---|
| OpenRouter (Opus 4.6, $15/$75 per M tokens) | ~EUR 3,000-8,000 | Yes, 2-3 effective | Hard cap |
| Local GLM-5 on 4x H200 (~EUR 250K, 5-year life) | ~EUR 4,200/month amortised | No | Limited only by GPU capacity |

The open-weight GLM-5 achieves an intelligence score of 49.5 on standard benchmarks, compared to 53 for the current best commercial model (Anthropic Opus 4.6) [1]. Local deployment eliminates rate limits entirely: throughput scales linearly with GPU capacity and can be expanded as demand grows. We would be limited only by the size of the system we design and purchase, not by third-party policies.

### 3.3 Fusion World Model Development

Need: Medium-term (12-24 months), with infrastructure required now. Third-party alternative: cloud GPU rental (Google Cloud, Lambda Labs) at ~EUR 3-5/GPU-hour for H100; viable for prototyping but prohibitively expensive for sustained training campaigns.

Microsoft recently published a generative world model in Nature that, trained purely on observational data, was able to learn the underlying physical laws governing a system rather than simply interpolating patterns [3]. Their WHAM model (World and Human Action Model, 1.6B parameters) was trained to play video games in real time, generating each frame on the fly in response to player actions. The model does not replay pre-recorded sequences; it generates entirely new, physically consistent gameplay as it unfolds. WHAM was trained on approximately 1.4 billion state transitions drawn from 60,986 recorded gameplay matches. In that context, a state transition is a single frame-to-frame step: given the current visual frame (the game state) and a player action (keyboard or controller input), the model predicts the next frame. The training corpus therefore consists of 1.4 billion (state, action, next-state) tuples from which the model learns the rules governing the system's evolution.

We propose to develop an analogous Fusion World Model trained on experimental data from ITER's partner tokamaks. Whilst the comparison between a computer game and a major international science project is not immediately obvious, the underlying technical problem is quite similar. A tokamak pulse produces a dense time series of diagnostic measurements (magnetic field, plasma current, electron temperature and density profiles, radiated power, and many others) sampled at rates from 1 kHz to 1 MHz depending on the diagnostic. A single state transition in the fusion context is one time step across all diagnostics: given the current plasma state (the vector of all diagnostic measurements at time t) and the control actions applied during that interval (coil currents, gas injection rates, heating power), the model predicts the plasma state at time t+1. Both cases reduce to the same mathematical structure: a sequence of (state, action, next-state) tuples from which a generative model learns to predict future evolution. A five-second pulse sampled at 1 kHz across 200 diagnostics produces approximately 1 million state transitions, each encoding the full observable state of the plasma and the control inputs that shaped its evolution.

Available training data from partner facilities (estimated):

| Facility | Approx. Shots | Avg. Duration | Key Diagnostics | Est. State Transitions |
|---|---|---|---|---|
| JET | ~105,000 | ~5 s | ~200 | ~10B |
| TCV | ~75,000 | ~2 s | ~50 | ~750M |
| JT-60U | ~15,000 | ~15 s | ~80 | ~1.8B |
| JT-60SA (dagger) | ~300 | ~10 s | ~80 | ~24M |
| ASDEX-U | ~42,000 | ~5 s | ~150 | ~3B |
| DIII-D | ~200,000 | ~5 s | ~100 | ~10B |
| Total | | | | ~26B |

(dagger) JT-60SA: first plasma October 2023; data volume growing with each operational campaign. JT-60U data (predecessor machine, 1991-2008) is accessible through the same QST data infrastructure.

A 1-2B parameter Fusion World Model trained on ~26 billion state transitions is computationally feasible on a 4x H200 server. Estimated training time: 4-8 weeks for initial convergence, with ongoing refinement as data pipelines mature.

The practical value of such a model lies in its ability to pre-play planned pulses before they are executed on the real machine. Given only the planned control waveforms as input, the model would generate a predicted pulse evolution time step by time step. This would enable session leaders and operations teams to:

- Validate the physics assumptions underpinning a planned pulse by running it through the world model in silico and comparing the predicted plasma response against expectations.
- Check machine limit avoidance by observing whether the predicted pulse trajectory approaches or exceeds operational boundaries on quantities such as plasma current, stored energy, or heat loads on plasma-facing components.
- Assess susceptibility to disruptions by examining whether the predicted state trajectory passes through regions of operational space that the model has learned are associated with disruption precursors in the training data.
- Explore alternative scenarios by modifying control waveforms and re-running the prediction to compare outcomes, without consuming machine time.

All of this would take place in silico before the pulse is run on the real machine. The technical feasibility of real-time generative world models has already been demonstrated by Microsoft, and there is no fundamental barrier to applying the same approach to the tokamak case, where both the time-step cadence and state dimensionality are comparable.

Starting this work today is important. The competences and infrastructure developed now will be ready for the start of commissioning and SRO. On the first day that ITER produces a plasma, the continual training pipelines, developed and proven on partner facility data, will be ready to begin learning the new physics that the ITER machine gives us access to. As we progress through our research plan and experimental programme, the models will learn alongside the physicists, complementing their activities and providing support. With each successive campaign the models grow more capable, their understanding of ITER's operating space deepening in step with that of the scientific team.

We envisage a control room in which physicists have direct access to LLM agents capable of serving requests across the full spectrum of complexity, from simple plotting tasks such as "show me the plasma current profiles for the last three disrupted pulses", to hypothesis testing backed by our experimental catalogue and coupled to physics simulation codes that agent teams could execute independently, reporting their findings back to the scientists. The agentic infrastructure described in Sections 3.1 and 3.2 provides exactly this capability, and the Fusion World Model adds a generative layer that enables the agents to reason about what might happen as well as what has happened. With a suitable pipeline in place, models can be retrained on morning experimental data so that afternoon sessions benefit from updated predictions.

## 4. Recommendation

A 4-GPU NVIDIA H200 server (~EUR 250K) provides a balanced entry point that addresses all three use cases simultaneously: multilingual embedding services, agentic LLM inference, and Fusion World Model training. Usage could be monitored and the system scaled (to 8 GPUs or additional nodes) as demand warrants. The investment secures ITER's ability to develop and deploy AI capabilities at a pace set by our scientific ambition rather than by third-party rate limits. Acquiring this infrastructure now provides the lead time necessary to ensure that mature, tested pipelines are operational from the first day of commissioning, increasing the return on investment for the IO and its partner Domestic Agencies.

The IO occupies a position that no individual Domestic Agency can replicate: it is simultaneously the architect, builder, and operator of the ITER machine. This role is complemented by federated access to experimental data from several partner fusion facilities, an arrangement not available to individual DAs. A dedicated GPU server allows the IO to exploit this position by developing, training, and integrating AI tools directly into its scientific and operational workflows. Individual DAs will develop their own AI capabilities around ITER data, but the IO is best placed to build the foundational infrastructure on which those efforts can build. Investing now ensures that this infrastructure is mature by the time it is needed most.

## References

[1] GLM-5 Team. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv:2602.15763, February 2026. https://arxiv.org/abs/2602.15763

[2] Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv:2507.20534, July 2025. https://arxiv.org/abs/2507.20534

[3] Kanervisto, A. et al. "World and Human Action Models towards gameplay ideation." Nature, 2025. https://doi.org/10.1038/s41586-025-08600-3

[4] Bodnar, C. et al. "Aurora: A Foundation Model of the Earth System." arXiv:2405.13063, 2024. https://arxiv.org/abs/2405.13063

[5] NVIDIA. "NVIDIA H200 GPU Specifications." https://www.nvidia.com/en-us/data-center/h200/
