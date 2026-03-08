"""Plot JET machine description evolution across structural epochs.

Shows the temporal evolution of:
- Active probe count and probe changes (enabled/disabled)
- PF coil configuration (DMSS versions)
- Wall configuration (limiter vs divertor eras)
- Device XML versions (when new calibrations occurred)

Queries Neo4j StructuralEpoch nodes created by the DeviceXMLScanner.

Usage:
    uv run python scripts/plot_jet_epoch_evolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imas_codex.graph.client import GraphClient


def fetch_epochs() -> list[dict]:
    """Query all JET structural epochs with metadata."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (se:StructuralEpoch {facility_id: 'jet', data_source_name: 'device_xml'})
                OPTIONAL MATCH (se)-[:INTRODUCED_IN]-(dn:DataNode)
                WHERE dn.path CONTAINS ':magprobes:'
                WITH se, count(dn) AS probe_count
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.last_shot AS last_shot,
                       se.description AS description,
                       se.pf_configuration AS pf_config,
                       se.wall_configuration AS wall_config,
                       se.device_xml_path AS xml_path,
                       se.snap_file_path AS snap_path,
                       se.probes_enabled AS probes_enabled,
                       se.probes_disabled AS probes_disabled,
                       probe_count
                ORDER BY se.first_shot
            """)
        )


def fetch_probe_counts_by_epoch() -> list[dict]:
    """Get active probe counts per epoch from DataNode status fields."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (se:StructuralEpoch {facility_id: 'jet', data_source_name: 'device_xml'})
                WHERE se.device_xml_path IS NOT NULL
                OPTIONAL MATCH (se)-[:INTRODUCED_IN]-(dn:DataNode)
                WHERE dn.path CONTAINS ':magprobes:'
                WITH se, count(dn) AS total_probes
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.description AS description,
                       total_probes
                ORDER BY se.first_shot
            """)
        )


# PF config display ordering
PF_CONFIG_ORDER = {
    None: 0,
    "pre-EFIT": 1,
    "DMSS=091": 2,
    "DMSS=105": 3,
}

PF_CONFIG_COLORS = {
    None: "#cccccc",
    "pre-EFIT": "#88aacc",
    "DMSS=091": "#cc8844",
    "DMSS=105": "#44aa88",
}

WALL_COLORS = {
    "limiter": "#ddaa44",
    "divertor": "#4488cc",
}


def main():
    epochs = fetch_epochs()
    if not epochs:
        print("No epochs found in graph")
        return

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Extract data
    shots = [e["first_shot"] for e in epochs]
    labels = []
    for e in epochs:
        version = e["id"].split(":")[-1]
        labels.append(version)

    x = np.arange(len(epochs))

    # --- Panel 1: Wall configuration timeline ---
    ax1 = axes[0]
    for i, e in enumerate(epochs):
        wc = e["wall_config"] or "unknown"
        color = WALL_COLORS.get(wc, "#999999")
        width = 0.8
        ax1.barh(
            0,
            width,
            left=i - width / 2,
            height=0.6,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if i == 0 or epochs[i - 1].get("wall_config") != wc:
            ax1.text(
                i,
                0.5,
                wc.title(),
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                rotation=45,
            )

    ax1.set_yticks([])
    ax1.set_title("Wall Configuration", fontsize=10, fontweight="bold", loc="left")
    ax1.set_ylim(-0.5, 1.2)

    # --- Panel 2: PF configuration ---
    ax2 = axes[1]
    pf_values = [e.get("pf_config") for e in epochs]
    for i, pf in enumerate(pf_values):
        color = PF_CONFIG_COLORS.get(pf, "#999999")
        level = PF_CONFIG_ORDER.get(pf, 0)
        ax2.bar(i, level + 1, color=color, edgecolor="white", linewidth=0.5, width=0.8)
        if i == 0 or pf_values[i - 1] != pf:
            display = pf or "none"
            ax2.text(
                i,
                level + 1.1,
                display,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
            )

    ax2.set_ylabel("PF Config", fontsize=8)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(["none", "pre-EFIT", "DMSS=091", "DMSS=105"], fontsize=7)
    ax2.set_title("PF Coil Configuration", fontsize=10, fontweight="bold", loc="left")

    # --- Panel 3: Probe changes ---
    ax3 = axes[2]
    for i, e in enumerate(epochs):
        enabled = e.get("probes_enabled") or []
        disabled = e.get("probes_disabled") or []
        n_enabled = len(enabled) if isinstance(enabled, list) else 0
        n_disabled = len(disabled) if isinstance(disabled, list) else 0

        if n_enabled > 0:
            ax3.bar(
                i,
                n_enabled,
                color="#44aa44",
                edgecolor="white",
                linewidth=0.5,
                width=0.4,
                align="edge",
            )
        if n_disabled > 0:
            ax3.bar(
                i + 0.4,
                -n_disabled,
                color="#cc4444",
                edgecolor="white",
                linewidth=0.5,
                width=0.4,
                align="edge",
            )

    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_ylabel("Probe Changes", fontsize=8)
    ax3.set_title(
        "Probe Enable/Disable Events per Epoch",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    ax3.legend(["Enabled", "Disabled"], fontsize=7, loc="upper right")

    # --- Panel 4: Device XML version changes ---
    ax4 = axes[3]
    xml_paths = [e.get("xml_path") or "" for e in epochs]
    unique_xmls = sorted({p for p in xml_paths if p})
    xml_colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_xmls), 1)))

    for i, xml in enumerate(xml_paths):
        if xml:
            idx = unique_xmls.index(xml)
            ax4.bar(
                i, 1, color=xml_colors[idx], edgecolor="white", linewidth=0.5, width=0.8
            )
        else:
            ax4.bar(i, 1, color="#eeeeee", edgecolor="white", linewidth=0.5, width=0.8)

    # Add legend for XML versions
    for idx, xml in enumerate(unique_xmls):
        name = xml.split("/")[-1].replace(".xml", "")
        ax4.plot([], [], "s", color=xml_colors[idx], label=name, markersize=8)
    ax4.legend(fontsize=7, loc="upper left", ncol=2)
    ax4.set_yticks([])
    ax4.set_title("Device XML Version", fontsize=10, fontweight="bold", loc="left")

    # X-axis labels
    ax4.set_xticks(x)
    ax4.set_xticklabels(
        [f"{label}\n({shot})" for label, shot in zip(labels, shots, strict=True)],
        fontsize=6,
        rotation=45,
        ha="right",
    )
    ax4.set_xlabel("Structural Epoch (first shot)", fontsize=9)

    fig.suptitle(
        "JET Machine Description Evolution — 20 Structural Epochs",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output = Path("docs/images/jet_epoch_evolution.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved: {output} ({len(epochs)} epochs)")
    plt.close()


if __name__ == "__main__":
    main()
