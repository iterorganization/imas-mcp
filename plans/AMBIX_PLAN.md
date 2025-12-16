# Ambix: The Universal Fusion Data Client

**System:** `imas-ambix`
**Role:** The Product (Runtime Client & Data Pump)
**Output:** Local IMAS Interface Data Structures (IDS)

---

## 1. Executive Vision

**Ambix** is the user-facing tool that "hydrates" IMAS objects. It is designed to be **Zero-Install** on the remote facility side. It assumes the remote machine is a "Dumb Data Pump" and performs all intelligent logic on the user's local machine.

**Core Philosophy: "Deterministic Execution"**
Ambix does not "think" or "search". It executes a pre-compiled **Recipe**. If you have the Recipe, you get the data.

---

## 2. Architecture: The Cloud-Agnostic Pump

### 2.1 The Recipe (Static Configuration)
A Recipe is a JSON document that tells Ambix exactly how to fetch and map data.
*   **Source:** Stored in a **Git Repository** (e.g., `imas-recipes`).
*   **Versioning:** Recipes are versioned alongside code.
*   **Structure:**
    ```json
    {
      "id": "JET/magnetics/ip/v1",
      "target": "ids.magnetics.flux_loop[0].flux",
      "pump": {
        "driver": "ppf",
        "args": ["mag", "ip"]
      },
      "transform": {
        "scale": -1.0,
        "units_in": "T",
        "units_out": "T"
      }
    }
    ```

### 2.2 The Transport (Apache Arrow / Parquet)
We solve the "Python Loop" bottleneck by streaming typed, columnar data.

1.  **Inject:** Ambix (Local) uses SSH to send a tiny, dependency-free Python script (`pump.py`) to the remote `/tmp`.
2.  **Execute:** `pump.py` calls the native data library (MDSplus/HDF5), reads the arrays, and writes them to a **Parquet** buffer.
3.  **Stream:** The Parquet bytes are streamed back over the SSH `stdout` (or SFTP) to Ambix.
4.  **Hydrate:** Ambix reads the Parquet table and maps the columns to the IMAS IDS structure in memory.

---

## 3. User Workflow

1.  **Install:** `pip install imas-ambix`
2.  **Select:** User requests data: `ambix.get("JET", 12345, "magnetics")`.
3.  **Resolve:** Ambix looks up the "magnetics" Recipe for JET in the local/cached Recipe Repo.
4.  **Fetch:** Ambix connects to JET via SSH, runs the pump, and retrieves Parquet data.
5.  **Return:** User gets a populated `imas.ids` object.

---

## 4. Implementation Strategy

### 4.1 The Recipe Repo
*   A separate Git repository containing only JSON/YAML files.
*   Organized by Machine and IDS: `recipes/jet/magnetics.json`.
*   Ambix client automatically pulls/updates this repo on first run.

### 4.2 The "Dumb" Pump
*   A standalone Python script with **zero dependencies** other than the facility's installed libraries (e.g., `MDSplus`).
*   If `pyarrow` is available remotely, use it. If not, fallback to a simple binary struct format or JSON (slower, but compatible).

### 4.3 The Local Client
*   **Language:** Python.
*   **Dependencies:** `imas-python`, `pyarrow`, `fabric`, `pydantic`.
*   **Logic:**
    *   Recipe Parser.
    *   SSH Connection Manager (Fabric).
    *   Parquet Deserializer.
    *   IMAS Mapper (applies scaling, coordinate alignment).

---

## 5. Roadmap

1.  **Prototype:** Manually write a Recipe for `TCV/magnetics`.
2.  **Transport:** Implement the `pump.py` script and the SSH streaming logic.
3.  **Client:** Build the `ambix.get()` function that ties it all together.
4.  **Recipe Repo:** Set up the Git structure for distributing recipes.
