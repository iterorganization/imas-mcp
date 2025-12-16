# Codex: The Federated Fusion Knowledge Graph Builder

**System:** `imas-codex`
**Role:** The Factory (Backend Engineering & Ontology Management)
**Output:** A versioned, containerized Neo4j Database Artifact

---

## 1. Executive Vision

**Codex** is the engineering engine that constructs the "Map" of the fusion world. It does not serve user requests directly; instead, it produces the **Federated Fusion Knowledge Graph (FFKG)**—a comprehensive, versioned artifact that links facility-specific data (Source) to the IMAS standard (Target).

**Core Philosophy: "Graph-as-Code"**
We do not manually draw a graph. We engineer a software pipeline. The graph is the deterministic output of running:
1.  **Discovery Engine** (Surveyor & Investigator Modules)
2.  **Ontology** (Agile LinkML definitions)
3.  **Logic Synthesis** (LLM-driven mapping)

---

## 2. Architecture: The Four-Zone Topology

The graph is structured into four distinct zones of "Truth":

| Zone | Name | Role | Source of Truth |
| :--- | :--- | :--- | :--- |
| **Zone A** | **Target Definition** (IMAS) | The immutable standard we map *to*. | **IMAS Python Library** (`imas.dd`) |
| **Zone B** | **Source Inventory** (Facility) | The catalog of what exists (Files, Trees). | **Discovery Engine** (Surveyor) |
| **Zone C** | **Forensic Evidence** (Context) | Documentation, legacy code, diagrams. | **Discovery Engine** (Investigator) |
| **Zone D** | **Transformation Logic** (The Bridge) | The computed link between Source and Target. | **LLM / LinkML** |

---

## 3. The Agile Schema Workflow (LinkML)

We reject the "Waterfall" approach to ontology. The schema must evolve as we discover new patterns in the wild.

### 3.1 The "Core" vs. "Wild" Split
The LinkML definitions (stored in `ontology/`) are divided:
*   **`ontology/core.yaml`**: Rigid definitions for Zone A (IMAS) and base classes.
*   **`ontology/wild.yaml`**: Permissive definitions for Zone B (Facility). Initially contains generic `RawArtifact` nodes.

### 3.2 The "LLM-as-Ontologist" Loop
1.  **Discover:** The Surveyor Module scans the facility and returns raw JSON.
2.  **Ingest:** The pipeline attempts to validate JSON against current LinkML models.
3.  **Identify Misses:** Data that fails validation (e.g., a new file type "FastCameraLog") is flagged.
4.  **Synthesize:** An LLM Agent analyzes the "Misses" and **generates a Schema Proposal (Pull Request)**:
    *   It defines a new class `FastCameraLog` in `ontology/wild.yaml`.
    *   It adds the necessary attributes found in the raw JSON.
5.  **Validate:** A human engineer (or senior agent) reviews the PR. CI/CD runs `linkml-validate` to ensure the new definition doesn't break the graph.

---

## 4. Implementation Strategy

### 4.1 The Discovery Engine (Zone B/C)
The discovery process employs a dual-layer strategy to balance performance with semantic depth. Instead of uploading static scripts, the system utilizes **Fabric** to establish a secure, interactive, read-only shell session on the remote facility.

#### A. The Surveyor Module (High-Throughput Scanning)
*   **Objective:** Construct the structural skeleton of the remote filesystem (Zone B).
*   **Mechanism:** Executes optimized, non-interactive shell commands (e.g., `find`, `ls -R`, `du`) via the Fabric connection.
*   **Configuration:** Controlled by `config/core.yaml` which sets global filters (e.g., exclude `.git`, `tmp`, `__pycache__`) and content limits (e.g., max file size for analysis).
*   **Output:** A comprehensive JSON tree representing the file hierarchy and metadata, without content analysis.

#### B. The Investigator Module (Semantic Exploration)
*   **Objective:** Analyze specific areas of interest to extract semantic context and provenance (Zone C/D).
*   **Mechanism:** An LLM-driven interactive loop that simulates a human operator.
    1.  **Targeting:** Identifies "Clusters of Interest" from the Surveyor's output (e.g., directories containing custom diagnostic codes).
    2.  **Exploration:** Dynamically chains available tools to inspect file contents and dependencies. The LLM is not restricted to a single command but orchestrates a toolset.
        *   *Tools:* `rg` (ripgrep) for semantic search, `head` for headers, `cat` (with size limits).
        *   *Example:* Using `rg` to find all files importing a specific library, then reading their headers.
    3.  **Synthesis:** Updates the Knowledge Graph with the extracted semantic relationships.
*   **Safety:** The Fabric connection is encapsulated in a **Read-Only Sandbox** that strictly enforces a whitelist of non-destructive commands at the client level.

### 4.2 The Builder Pipeline
A containerized Python application that:
1.  Reads `config/{machine}.yaml`.
2.  **Phase 1:** Executes the **Surveyor Module** to populate the Zone B skeleton.
3.  **Phase 2:** The **Investigator Module** iterates through "Unknown" or high-priority nodes in the graph.
    *   It establishes an interactive session to read context and propose new LinkML definitions.
4.  **Phase 3:** Synthesizes Zone D (Logic) and populates Neo4j.

### 4.3 Quality Assurance (CI/CD)
Since the LLM can modify the schema, we need rigorous checks.

*   **Pre-Commit Hook:**
    *   Runs `linkml-lint` on `ontology/*.yaml`.
    *   Ensures no breaking changes to `core.yaml`.
*   **CI Pipeline:**
    *   Generates Pydantic models from LinkML.
    *   Runs a "Dry Run" ingestion with sample data.
    *   If successful, merges the schema update.

---

## 5. Artifact Management & Versioning

We treat the Database as an Artifact.

1.  **Build:** The pipeline runs and populates a local Neo4j instance.
2.  **Dump:** Run `neo4j-admin database dump`.
3.  **Package:** Wrap the dump in an OCI-compliant container layer.
4.  **Publish:** Push to **GitHub Container Registry (GHCR)**.
    *   Tag: `ghcr.io/iter/imas-codex-graph:sha-12345`
5.  **Traceability:** The Git commit hash of the `imas-codex` repo is the single source of truth. It references the specific GHCR tag used for that version.

---

## 6. Roadmap

1.  **Foundation:** Setup `imas-codex` repo with `ontology/` folder and basic LinkML setup.
2.  **Surveyor v1:** Implement the basic "File Walker" module using Fabric.
3.  **Builder v1:** Script to ingest Surveyor JSON into Neo4j using LinkML models.
4.  **Agile Loop:** Implement the LLM agent that can read "Misses" and update `wild.yaml`.
5.  **CI/CD:** Configure GitHub Actions for LinkML validation and GHCR pushing.
