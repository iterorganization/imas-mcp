"""Plot JET machine description evolution across structural epochs.

Produces a comprehensive 6-panel figure showing:
1. Wall configuration timeline (limiter types across shot ranges)
2. PF coil configuration evolution (DMSS versions)
3. Device XML versions (calibration file changes)
4. Probe change events (enabled/disabled per epoch)
5. Component inventory (magprobes, PF coils, passives, circuits)
6. Limiter contour complexity (number of boundary points)

All data sourced entirely from the Neo4j knowledge graph.

Usage:
    uv run --with matplotlib python scripts/plot_jet_epoch_evolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imas_codex.graph.client import GraphClient

# --- Color palettes ---

PF_COLORS = {
    None: "#d0d0d0",
    "pre-EFIT": "#6e9ecf",
    "DMSS=091": "#e8963e",
    "DMSS=105": "#59b389",
}

WALL_COLORS = {
    "limiter": "#e6a832",
    "divertor": "#3d7fc4",
}

LIMITER_NAMES = {
    "Limiter": ("Carbon belt limiter", "#e6a832"),
    "Mk1": ("Mk I divertor", "#3976af"),
    "Mk2A": ("Mk IIA Gas Box", "#5a9bd5"),
    "Mk2GB": ("Mk IIGB", "#7ab8e0"),
    "Mk2GB-NS": ("Mk IIGB-NS (no sep.)", "#89c4e4"),
    "Mk2GB-SR": ("Mk IIGB-SR (sep. rem.)", "#69b0d8"),
    "Mk2HD": ("Mk II High Delta", "#4e8cc7"),
    "Mk2ILW": ("Mk II ILW (Be/W)", "#2c6faa"),
}

COMPONENT_COLORS = {
    "magprobes": "#5778a4",
    "pfcoils": "#e49444",
    "passives": "#6a9f58",
    "circuits": "#d1605e",
}


def fetch_epochs() -> list[dict]:
    """Query all JET structural epochs with full metadata."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (se:StructuralEpoch {facility_id: 'jet', data_source_name: 'device_xml'})
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.last_shot AS last_shot,
                       se.description AS description,
                       se.pf_configuration AS pf_config,
                       se.wall_configuration AS wall_config,
                       se.device_xml_path AS xml_path,
                       se.snap_file_path AS snap_path,
                       se.probes_enabled AS probes_enabled,
                       se.probes_disabled AS probes_disabled
                ORDER BY se.first_shot
            """)
        )


def fetch_component_counts() -> dict[str, dict]:
    """Get component breakdown per epoch from DataNode relationships."""
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (dn:DataNode {facility_id: 'jet'})-[:INTRODUCED_IN]->(e:StructuralEpoch)
            WITH e,
                 count(CASE WHEN dn.path CONTAINS 'magprobe' THEN 1 END) AS magprobes,
                 count(CASE WHEN dn.path CONTAINS 'pfcoil' THEN 1 END) AS pfcoils,
                 count(CASE WHEN dn.path CONTAINS 'pfpassive' THEN 1 END) AS passives,
                 count(CASE WHEN dn.path CONTAINS 'pfcircuit' THEN 1 END) AS circuits
            RETURN e.id AS epoch, magprobes, pfcoils, passives, circuits
        """)
    return {r["epoch"]: r for r in results}


def fetch_limiter_contours() -> list[dict]:
    """Get limiter contour data from DataNode nodes."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (dn:DataNode {facility_id: 'jet'})
                WHERE dn.path STARTS WITH 'jet:device_xml:limiter:'
                RETURN dn.path AS path,
                       dn.first_shot AS first_shot,
                       dn.last_shot AS last_shot,
                       dn.n_points AS n_points
                ORDER BY dn.first_shot
            """)
        )


def short_name(epoch_id: str) -> str:
    return epoch_id.split(":")[-1]


def main():
    epochs = fetch_epochs()
    if not epochs:
        print("No epochs found in graph")
        return

    comp_counts = fetch_component_counts()
    limiters = fetch_limiter_contours()

    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 20),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1.2, 1, 1.6, 1.2]},
    )
    fig.patch.set_facecolor("white")

    # ========== Panel 1: Wall / Limiter Configuration Timeline ==========
    ax = axes[0]

    # Map each epoch to a limiter based on shot overlap
    limiter_lookup = {}
    for lim in limiters:
        name = lim["path"].split(":")[-1]
        fs, ls = lim["first_shot"], lim.get("last_shot")
        limiter_lookup[name] = (fs, ls)

    def find_limiter_for_epoch(first_shot):
        for name, (fs, ls) in limiter_lookup.items():
            if ls is None:
                if first_shot >= fs:
                    return name
            elif fs <= first_shot <= ls:
                return name
        return None

    prev_lim = None
    for i, e in enumerate(epochs):
        lim = find_limiter_for_epoch(e["first_shot"])
        wc = e["wall_config"] or "unknown"
        if lim and lim in LIMITER_NAMES:
            _, color = LIMITER_NAMES[lim]
        else:
            color = WALL_COLORS.get(wc, "#999")

        ax.bar(i, 1, color=color, edgecolor="white", linewidth=0.5, width=0.85)

        # Label on transition
        if lim != prev_lim and lim:
            display = LIMITER_NAMES.get(lim, (lim, "#999"))[0]
            ax.annotate(
                display,
                xy=(i, 1.02),
                fontsize=6,
                rotation=55,
                ha="left",
                va="bottom",
                fontweight="bold",
                color=color,
            )
        prev_lim = lim

    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Wall / Limiter Configuration",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Panel 2: PF Configuration ==========
    ax = axes[1]
    pf_order = {None: 0, "pre-EFIT": 1, "DMSS=091": 2, "DMSS=105": 3}
    pf_labels_y = ["None", "pre-EFIT", "DMSS=091", "DMSS=105"]

    prev_pf = None
    for i, e in enumerate(epochs):
        pf = e.get("pf_config")
        level = pf_order.get(pf, 0)
        color = PF_COLORS.get(pf, "#ccc")
        ax.bar(
            i,
            level + 0.5,
            bottom=0,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            width=0.85,
        )

        if pf != prev_pf:
            display = pf or "None"
            ax.annotate(
                display,
                xy=(i, level + 0.6),
                fontsize=6,
                rotation=45,
                ha="left",
                va="bottom",
                color="#333",
                fontweight="bold",
            )
        prev_pf = pf

    ax.set_ylabel("PF Config Level", fontsize=8)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(pf_labels_y, fontsize=7)
    ax.set_ylim(0, 4.5)
    ax.set_title(
        "PF Coil Configuration", fontsize=11, fontweight="bold", loc="left", pad=8
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Panel 3: Device XML Versions ==========
    ax = axes[2]
    xml_paths = [e.get("xml_path") or "" for e in epochs]
    unique_xmls = sorted({p for p in xml_paths if p})
    cmap = plt.cm.Set2(np.linspace(0, 0.8, max(len(unique_xmls), 1)))

    prev_xml = None
    for i, xml in enumerate(xml_paths):
        if xml:
            idx = unique_xmls.index(xml)
            ax.bar(i, 1, color=cmap[idx], edgecolor="white", linewidth=0.5, width=0.85)
            if xml != prev_xml:
                name = xml.split("/")[-1].replace(".xml", "")
                ax.annotate(
                    name,
                    xy=(i, 1.02),
                    fontsize=6,
                    rotation=45,
                    ha="left",
                    va="bottom",
                    fontweight="bold",
                    color=cmap[idx] * 0.7,
                )
        else:
            ax.bar(i, 1, color="#f0f0f0", edgecolor="#ddd", linewidth=0.5, width=0.85)
        prev_xml = xml

    # Legend
    legend_handles = []
    for idx, xml in enumerate(unique_xmls):
        name = xml.split("/")[-1].replace(".xml", "")
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=cmap[idx], label=name))
    legend_handles.append(
        plt.Rectangle((0, 0), 1, 1, fc="#f0f0f0", ec="#ddd", label="No device XML")
    )
    ax.legend(
        handles=legend_handles, fontsize=6, loc="upper left", ncol=3, framealpha=0.9
    )
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title(
        "Device XML Version (Calibration File)",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Panel 4: Probe Change Events ==========
    ax = axes[3]
    has_events = False
    for i, e in enumerate(epochs):
        enabled = e.get("probes_enabled") or []
        disabled = e.get("probes_disabled") or []
        n_en = len(enabled) if isinstance(enabled, list) else 0
        n_dis = len(disabled) if isinstance(disabled, list) else 0
        if n_en > 0:
            ax.bar(
                i - 0.2,
                n_en,
                color="#4caf50",
                edgecolor="white",
                linewidth=0.5,
                width=0.35,
            )
            has_events = True
        if n_dis > 0:
            ax.bar(
                i + 0.2,
                -n_dis,
                color="#e53935",
                edgecolor="white",
                linewidth=0.5,
                width=0.35,
            )
            has_events = True

    ax.axhline(y=0, color="#666", linewidth=0.5, linestyle="-")
    if has_events:
        ax.legend(
            [
                plt.Rectangle((0, 0), 1, 1, fc="#4caf50"),
                plt.Rectangle((0, 0), 1, 1, fc="#e53935"),
            ],
            ["Enabled", "Disabled"],
            fontsize=7,
            loc="upper right",
        )
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(
        "Probe Enable / Disable Events",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Panel 5: Component Inventory (stacked bar) ==========
    ax = axes[4]
    comp_keys = ["magprobes", "pfcoils", "passives", "circuits"]
    comp_labels = ["Magnetic Probes", "PF Coils", "Passives", "Circuits"]

    bottoms = np.zeros(n)
    for key, label in zip(comp_keys, comp_labels, strict=True):
        vals = []
        for e in epochs:
            c = comp_counts.get(e["id"], {})
            vals.append(c.get(key, 0))
        vals_arr = np.array(vals, dtype=float)
        ax.bar(
            x,
            vals_arr,
            bottom=bottoms,
            color=COMPONENT_COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            width=0.85,
            label=label,
        )

        # Label the count inside the bar for the first epoch that has data
        for i, v in enumerate(vals_arr):
            if v > 0 and bottoms[i] == 0 and key == "magprobes":
                ax.text(
                    i,
                    bottoms[i] + v / 2,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                    fontweight="bold",
                )

        bottoms += vals_arr

    # Add total label on top
    for i in range(n):
        total = int(bottoms[i])
        if total > 0:
            ax.text(
                i,
                total + 3,
                str(total),
                ha="center",
                va="bottom",
                fontsize=6,
                color="#333",
            )

    ax.set_ylabel("Component Count", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title(
        "Component Inventory per Epoch",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Panel 6: Limiter Contour Complexity ==========
    ax = axes[5]
    lim_shots = [lim["first_shot"] for lim in limiters]
    lim_points = [lim["n_points"] for lim in limiters]
    lim_names = [lim["path"].split(":")[-1] for lim in limiters]

    colors = [LIMITER_NAMES.get(name, (name, "#888"))[1] for name in lim_names]
    bars = ax.bar(
        range(len(limiters)),
        lim_points,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        width=0.7,
    )
    for i, (_bar, pts, name) in enumerate(
        zip(bars, lim_points, lim_names, strict=True)
    ):
        ax.text(
            i, pts + 3, str(pts), ha="center", va="bottom", fontsize=7, color="#333"
        )
        ax.text(
            i, -8, name, ha="center", va="top", fontsize=6, rotation=45, color="#555"
        )

    ax.set_ylabel("Contour Points", fontsize=8)
    ax.set_xticks(range(len(limiters)))
    ax.set_xticklabels(
        [f"({lim_shots[i]})" for i in range(len(limiters))],
        fontsize=6,
        rotation=45,
        ha="right",
    )
    ax.set_title(
        "Limiter Contour Complexity (Boundary Points)",
        fontsize=11,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ========== Shared X-axis for panels 1-5 ==========
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(
        [f"{label}\n({shot})" for label, shot in zip(labels, shots, strict=True)],
        fontsize=5.5,
        rotation=55,
        ha="right",
    )

    # ========== Title & Layout ==========
    fig.suptitle(
        "JET Machine Description Evolution — Structural Epochs from Knowledge Graph",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    subtitle = (
        f"{n} epochs · {len(limiters)} limiter configurations · "
        f"{len(unique_xmls)} device XML versions · "
        f"shot range {shots[0]}–{shots[-1]}+"
    )
    fig.text(
        0.5, 0.982, subtitle, ha="center", fontsize=9, color="#666", style="italic"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.975], h_pad=1.5)
    output = Path("docs/images/jet_epoch_evolution.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output}")
    print(f"  {n} epochs, {len(limiters)} limiters, {len(unique_xmls)} XML versions")

    # --- Also save a compact 3-panel summary ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig2.patch.set_facecolor("white")

    # Compact Panel 1: Wall + PF combined timeline
    ax = axes2[0]
    for i, e in enumerate(epochs):
        lim = find_limiter_for_epoch(e["first_shot"])
        wc = e["wall_config"] or "unknown"
        if lim and lim in LIMITER_NAMES:
            _, color = LIMITER_NAMES[lim]
        else:
            color = WALL_COLORS.get(wc, "#999")
        ax.bar(
            i,
            0.45,
            bottom=0.55,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            width=0.9,
        )

        pf = e.get("pf_config")
        pf_color = PF_COLORS.get(pf, "#ccc")
        ax.bar(
            i,
            0.45,
            bottom=0,
            color=pf_color,
            edgecolor="white",
            linewidth=0.3,
            width=0.9,
        )

    ax.axhline(y=0.55, color="white", linewidth=2)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["PF Config", "Wall Type"], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Wall & PF Configuration Timeline", fontsize=11, fontweight="bold", loc="left"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact Panel 2: Component counts
    ax = axes2[1]
    bottoms2 = np.zeros(n)
    for key, label in zip(comp_keys, comp_labels, strict=True):
        vals = [comp_counts.get(e["id"], {}).get(key, 0) for e in epochs]
        vals_arr = np.array(vals, dtype=float)
        ax.bar(
            x,
            vals_arr,
            bottom=bottoms2,
            color=COMPONENT_COLORS[key],
            edgecolor="white",
            linewidth=0.5,
            width=0.85,
            label=label,
        )
        bottoms2 += vals_arr
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Component Inventory", fontsize=11, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact Panel 3: Probe events + XML transitions
    ax = axes2[2]
    for i, e in enumerate(epochs):
        enabled = e.get("probes_enabled") or []
        disabled = e.get("probes_disabled") or []
        n_en = len(enabled) if isinstance(enabled, list) else 0
        n_dis = len(disabled) if isinstance(disabled, list) else 0
        if n_en > 0:
            ax.bar(
                i - 0.15,
                n_en,
                color="#4caf50",
                width=0.3,
                edgecolor="white",
                linewidth=0.5,
            )
        if n_dis > 0:
            ax.bar(
                i + 0.15,
                -n_dis,
                color="#e53935",
                width=0.3,
                edgecolor="white",
                linewidth=0.5,
            )

        # Mark XML version transitions
        xml = xml_paths[i]
        if i > 0 and xml != xml_paths[i - 1] and xml:
            ax.axvline(
                x=i - 0.5, color="#ff9800", linewidth=1.5, linestyle="--", alpha=0.7
            )

    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.set_ylabel("Probe Changes", fontsize=8)
    ax.set_title(
        "Probe Events & XML Version Transitions (dashed)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    axes2[2].set_xticks(x)
    axes2[2].set_xticklabels(
        [f"{label}\n({shot})" for label, shot in zip(labels, shots, strict=True)],
        fontsize=5.5,
        rotation=55,
        ha="right",
    )

    fig2.suptitle(
        "JET Epoch Evolution — Compact Summary",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.0)
    output2 = Path("docs/images/jet_epoch_summary.png")
    fig2.savefig(output2, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output2}")
    plt.close("all")


if __name__ == "__main__":
    main()
