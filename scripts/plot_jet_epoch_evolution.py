"""Plot JET machine description evolution across structural epochs.

Generates separate multi-window plots showing the evolution of all machine
description items, including PF coils, magnetic probes, passive structures,
flux loops, circuits, limiter contours, and calibration epochs.

Output files (all saved to docs/images/):
  - jet_wall_limiter_evolution.png: Wall/limiter configuration timeline
  - jet_pf_coil_evolution.png: PF coil inventory and configuration
  - jet_magnetics_evolution.png: Magnetic probes, flux loops, and probe events
  - jet_passive_circuit_evolution.png: Passive structures and PF circuits
  - jet_component_overview.png: Full component inventory stacked bar
  - jet_limiter_contours.png: Limiter contour complexity
  - jet_epoch_summary.png: Compact 4-panel overview

All data sourced entirely from the Neo4j knowledge graph.

Usage:
    uv run --with matplotlib python scripts/plot_jet_epoch_evolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imas_codex.graph.client import GraphClient

OUTPUT_DIR = Path("docs/images")

# --- Color palettes ---

PF_CONFIG_COLORS = {
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
    "flux_loops": "#9c6fb0",
}


# =================================================================
# Data fetching
# =================================================================


def fetch_epochs() -> list[dict]:
    """Query all JET structural epochs with full metadata."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (se:SignalEpoch {facility_id: 'jet', data_source_name: 'device_xml'})
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
    """Get detailed component breakdown per epoch from DataNode relationships."""
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (dn:DataNode {facility_id: 'jet'})-[:INTRODUCED_IN]->(e:SignalEpoch)
            WITH e,
                 count(CASE WHEN dn.path CONTAINS 'magprobe' THEN 1 END) AS magprobes,
                 count(CASE WHEN dn.path CONTAINS 'pfcoil' THEN 1 END) AS pfcoils,
                 count(CASE WHEN dn.path CONTAINS 'pfpassive' THEN 1 END) AS passives,
                 count(CASE WHEN dn.path CONTAINS 'pfcircuit' THEN 1 END) AS circuits,
                 count(CASE WHEN dn.path CONTAINS 'flux' THEN 1 END) AS flux_loops
            RETURN e.id AS epoch, magprobes, pfcoils, passives, circuits, flux_loops
        """)
    return {r["epoch"]: r for r in results}


def fetch_jec2020_pf_coils() -> list[dict]:
    """Get JEC2020 PF coil data with coil names (P1-P4, D1-D4)."""
    with GraphClient() as gc:
        return list(
            gc.query("""
                MATCH (dn:DataNode {facility_id: 'jet', data_source_name: 'jec2020_geometry'})
                WHERE dn.path CONTAINS 'pf_coil'
                RETURN dn.path AS path,
                       dn.description AS description,
                       dn.rCentre AS r, dn.zCentre AS z,
                       dn.dR AS dr, dn.dZ AS dz
                ORDER BY dn.path
            """)
        )


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
                       dn.n_points AS n_points,
                       dn.r_contour AS r_contour,
                       dn.z_contour AS z_contour
                ORDER BY dn.first_shot
            """)
        )


def fetch_jec2020_summary() -> dict[str, int]:
    """Get JEC2020 global component counts."""
    with GraphClient() as gc:
        result = gc.query("""
            MATCH (dn:DataNode {facility_id: 'jet', data_source_name: 'jec2020_geometry'})
            RETURN
                count(CASE WHEN dn.path CONTAINS 'probe' THEN 1 END) AS probes,
                count(CASE WHEN dn.path CONTAINS 'pf_coil' THEN 1 END) AS pf_coils,
                count(CASE WHEN dn.path CONTAINS 'pf_circuit' THEN 1 END) AS pf_circuits,
                count(CASE WHEN dn.path CONTAINS 'flux_loop' THEN 1 END) AS flux_loops,
                count(CASE WHEN dn.path CONTAINS 'iron' THEN 1 END) AS iron,
                count(CASE WHEN dn.path CONTAINS 'limiter' THEN 1 END) AS limiter
        """)
        return dict(next(iter(result)))


def short_name(epoch_id: str) -> str:
    return epoch_id.split(":")[-1]


# =================================================================
# Shared helpers
# =================================================================


def _style_ax(ax: plt.Axes) -> None:
    """Remove top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _epoch_xticks(
    ax: plt.Axes, labels: list[str], shots: list[int], rotation: float = 55
) -> None:
    """Set epoch x-tick labels."""
    x = np.arange(len(labels))
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n({s})" for lbl, s in zip(labels, shots, strict=True)],
        fontsize=5.5,
        rotation=rotation,
        ha="right",
    )


def _find_limiter(limiter_lookup: dict, first_shot: int) -> str | None:
    for name, (fs, ls) in limiter_lookup.items():
        if ls is None:
            if first_shot >= fs:
                return name
        elif fs <= first_shot <= ls:
            return name
    return None


# =================================================================
# Individual plot functions
# =================================================================


def plot_wall_limiter(epochs, limiters) -> None:
    """Figure 1: Wall/limiter configuration + limiter contour shapes."""
    n = len(epochs)
    # Build limiter lookup
    limiter_lookup = {}
    for lim in limiters:
        name = lim["path"].split(":")[-1]
        limiter_lookup[name] = (lim["first_shot"], lim.get("last_shot"))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[3, 2])
    fig.patch.set_facecolor("white")

    # Left: epoch timeline colored by limiter
    ax = axes[0]
    prev_lim = None
    for i, e in enumerate(epochs):
        lim = _find_limiter(limiter_lookup, e["first_shot"])
        wc = e["wall_config"] or "unknown"
        if lim and lim in LIMITER_NAMES:
            _, color = LIMITER_NAMES[lim]
        else:
            color = WALL_COLORS.get(wc, "#999")

        ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=0.5, height=0.7)
        if lim != prev_lim and lim:
            display = LIMITER_NAMES.get(lim, (lim, "#999"))[0]
            ax.annotate(
                display,
                xy=(i + 0.5, 0.45),
                fontsize=7,
                rotation=60,
                ha="left",
                va="bottom",
                fontweight="bold",
                color=color,
            )
        prev_lim = lim

    ax.set_yticks([])
    ax.set_xlim(-0.5, n - 0.5)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(
        [f"{lbl}\n({s})" for lbl, s in zip(labels, shots, strict=True)],
        fontsize=5,
        rotation=55,
        ha="right",
    )
    ax.set_title(
        "Wall Configuration Timeline", fontsize=11, fontweight="bold", loc="left"
    )
    _style_ax(ax)

    # Right: overlaid limiter contours
    ax = axes[1]
    for lim in limiters:
        name = lim["path"].split(":")[-1]
        r = lim.get("r_contour", [])
        z = lim.get("z_contour", [])
        if r and z:
            _, color = LIMITER_NAMES.get(name, (name, "#888"))
            ax.plot(
                r,
                z,
                color=color,
                linewidth=1.2,
                label=f"{name} ({lim['n_points']} pts)",
            )

    ax.set_xlabel("R (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Limiter Contour Shapes", fontsize=11, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.3)
    _style_ax(ax)

    fig.suptitle(
        "JET First Wall & Limiter Evolution",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_wall_limiter_evolution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_pf_coils(epochs, comp_counts, jec2020_pf) -> None:
    """Figure 2: PF coil evolution — inventory per epoch + JEC2020 coil map."""
    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # Top-left: PF coil count per epoch (from device XML)
    ax = axes[0, 0]
    pf_counts = [comp_counts.get(e["id"], {}).get("pfcoils", 0) for e in epochs]
    colors = []
    for e in epochs:
        wc = e.get("wall_config", "")
        pf = e.get("pf_config")
        if wc == "limiter":
            colors.append("#e6a832")  # limiter era — no divertor coils
        elif pf is None:
            colors.append("#d0d0d0")
        else:
            colors.append(PF_CONFIG_COLORS.get(pf, "#ccc"))

    ax.bar(x, pf_counts, color=colors, edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(pf_counts):
        ax.text(i, v + 0.3, str(v), ha="center", va="bottom", fontsize=6, color="#333")

    # Annotate pre-divertor: no coil data available but PF coils existed
    for i, e in enumerate(epochs):
        if pf_counts[i] == 0 and e["wall_config"] == "limiter":
            ax.annotate(
                "No divertor\ncoils",
                xy=(i, 0.5),
                fontsize=5,
                ha="center",
                va="bottom",
                color="#999",
                style="italic",
            )
        elif pf_counts[i] == 0:
            ax.annotate(
                "Pre-EFIT++\n(no XML)",
                xy=(i, 0.5),
                fontsize=5,
                ha="center",
                va="bottom",
                color="#999",
                style="italic",
            )

    ax.set_ylabel("PF Coil Count", fontsize=9)
    ax.set_ylim(0, max(pf_counts) * 1.2 if max(pf_counts) > 0 else 5)
    ax.set_title(
        "PF Coils per Epoch (Device XML)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _epoch_xticks(ax, labels, shots)
    _style_ax(ax)

    # Top-right: PF configuration evolution
    ax = axes[0, 1]
    pf_order = {None: 0, "pre-EFIT": 1, "DMSS=091": 2, "DMSS=105": 3}
    pf_labels_y = [
        "None\n(limiter era)",
        "pre-EFIT\n(Mk1-Mk2HD)",
        "DMSS=091",
        "DMSS=105",
    ]

    prev_pf = None
    for i, e in enumerate(epochs):
        pf = e.get("pf_config")
        level = pf_order.get(pf, 0)
        color = PF_CONFIG_COLORS.get(pf, "#ccc")
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
        "PF Coil Configuration",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _epoch_xticks(ax, labels, shots)
    _style_ax(ax)

    # Bottom-left: JEC2020 PF coil R,Z positions (main vs divertor)
    ax = axes[1, 0]
    main_coils = []
    div_coils = []
    for coil in jec2020_pf:
        desc = coil.get("description", "")
        r = coil.get("r")
        z = coil.get("z")
        if r is None or z is None:
            continue
        # D1-D4 are divertor coils (names contain "D1", "D2", "D3", "D4")
        if any(f"D{i}" in desc for i in range(1, 5)):
            div_coils.append(coil)
        else:
            main_coils.append(coil)

    if main_coils:
        mr = [c["r"] for c in main_coils]
        mz = [c["z"] for c in main_coils]
        ax.scatter(
            mr,
            mz,
            c="#3976af",
            s=80,
            zorder=5,
            label=f"Main PF (P1-P4): {len(main_coils)}",
        )
        for c in main_coils:
            # Extract short name from description like "PF coil 1 (P1/ME)"
            name = c["description"].split("(")[-1].rstrip(")")
            ax.annotate(
                name,
                (c["r"], c["z"]),
                fontsize=5,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
                color="#3976af",
            )

    if div_coils:
        dr = [c["r"] for c in div_coils]
        dz = [c["z"] for c in div_coils]
        ax.scatter(
            dr,
            dz,
            c="#e53935",
            s=80,
            marker="D",
            zorder=5,
            label=f"Divertor (D1-D4): {len(div_coils)}",
        )
        for c in div_coils:
            name = c["description"].split("(")[-1].rstrip(")")
            ax.annotate(
                name,
                (c["r"], c["z"]),
                fontsize=5,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
                color="#e53935",
            )

    ax.set_xlabel("R (m)", fontsize=9)
    ax.set_ylabel("Z (m)", fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(
        "JEC2020 PF Coil Positions (Main vs Divertor)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax.grid(True, alpha=0.3)
    _style_ax(ax)

    # Bottom-right: PF circuits per epoch
    ax = axes[1, 1]
    ci_counts = [comp_counts.get(e["id"], {}).get("circuits", 0) for e in epochs]
    ax.bar(x, ci_counts, color="#d1605e", edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(ci_counts):
        if v > 0:
            ax.text(
                i, v + 0.2, str(v), ha="center", va="bottom", fontsize=6, color="#333"
            )

    ax.set_ylabel("Circuit Count", fontsize=9)
    ax.set_title(
        "PF Circuits per Epoch",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _epoch_xticks(ax, labels, shots)
    _style_ax(ax)

    fig.suptitle(
        "JET PF Coil System Evolution",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_pf_coil_evolution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_magnetics(epochs, comp_counts) -> None:
    """Figure 3: Magnetic probes and flux loops evolution."""
    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: Magnetic probe count per epoch
    ax = axes[0]
    mp_counts = [comp_counts.get(e["id"], {}).get("magprobes", 0) for e in epochs]
    ax.bar(x, mp_counts, color="#5778a4", edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(mp_counts):
        if v > 0:
            ax.text(
                i, v + 2, str(v), ha="center", va="bottom", fontsize=6, color="#333"
            )
        else:
            ax.annotate(
                "no XML",
                xy=(i, 1),
                fontsize=5,
                ha="center",
                color="#999",
                style="italic",
            )

    ax.set_ylabel("Probe Count", fontsize=9)
    ax.set_title(
        "Magnetic Probes (BPME pick-up coils) per Epoch",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _style_ax(ax)

    # Panel 2: Flux loop count per epoch
    ax = axes[1]
    fl_counts = [comp_counts.get(e["id"], {}).get("flux_loops", 0) for e in epochs]
    ax.bar(x, fl_counts, color="#9c6fb0", edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(fl_counts):
        if v > 0:
            ax.text(
                i, v + 0.2, str(v), ha="center", va="bottom", fontsize=6, color="#333"
            )

    ax.set_ylabel("Flux Loop Count", fontsize=9)
    ax.set_title(
        "Flux Loops per Epoch",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _style_ax(ax)

    # Panel 3: Probe enable/disable events + cumulative active count
    ax = axes[2]
    ax2 = ax.twinx()

    # Track cumulative probe status changes
    cumulative_changes = 0
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

        cumulative_changes += n_en + n_dis

    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.legend(
        [
            plt.Rectangle((0, 0), 1, 1, fc="#4caf50"),
            plt.Rectangle((0, 0), 1, 1, fc="#e53935"),
        ],
        ["Enabled", "Disabled"],
        fontsize=7,
        loc="upper right",
    )
    ax.set_ylabel("Probe Changes", fontsize=9)
    ax.set_title(
        "Probe Enable / Disable Events",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _style_ax(ax)
    ax2.spines["top"].set_visible(False)

    _epoch_xticks(axes[2], labels, shots)

    fig.suptitle(
        "JET Magnetics System Evolution",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_magnetics_evolution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_passives_circuits(epochs, comp_counts) -> None:
    """Figure 4: Passive structures and PF circuit evolution."""
    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: Passive structure count
    ax = axes[0]
    ps_counts = [comp_counts.get(e["id"], {}).get("passives", 0) for e in epochs]
    ax.bar(x, ps_counts, color="#6a9f58", edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(ps_counts):
        if v > 0:
            ax.text(
                i, v + 0.5, str(v), ha="center", va="bottom", fontsize=6, color="#333"
            )
        else:
            ax.annotate(
                "no XML",
                xy=(i, 1),
                fontsize=5,
                ha="center",
                color="#999",
                style="italic",
            )

    ax.set_ylabel("Passive Element Count", fontsize=9)
    ax.set_title(
        "Passive Structures (Vessel Wall, Stabilizing Plates)",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _style_ax(ax)

    # Panel 2: PF circuit count
    ax = axes[1]
    ci_counts = [comp_counts.get(e["id"], {}).get("circuits", 0) for e in epochs]
    ax.bar(x, ci_counts, color="#d1605e", edgecolor="white", linewidth=0.5, width=0.85)
    for i, v in enumerate(ci_counts):
        if v > 0:
            ax.text(
                i, v + 0.2, str(v), ha="center", va="bottom", fontsize=6, color="#333"
            )

    ax.set_ylabel("Circuit Count", fontsize=9)
    ax.set_title(
        "PF Circuit-to-Coil Mappings",
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    _style_ax(ax)

    _epoch_xticks(axes[1], labels, shots)

    fig.suptitle(
        "JET Passive Structures & PF Circuits",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_passive_circuit_evolution.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_component_overview(epochs, comp_counts) -> None:
    """Figure 5: Full component inventory stacked bar chart."""
    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("white")

    comp_keys = ["magprobes", "pfcoils", "passives", "circuits", "flux_loops"]
    comp_labels = [
        "Magnetic Probes (BPME)",
        "PF Coils",
        "Passive Structures",
        "PF Circuits",
        "Flux Loops",
    ]

    bottoms = np.zeros(n)
    for key, label in zip(comp_keys, comp_labels, strict=True):
        vals = [comp_counts.get(e["id"], {}).get(key, 0) for e in epochs]
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

        # Label count inside bar for first data epoch
        for i, v in enumerate(vals_arr):
            if v > 5:
                ax.text(
                    i,
                    bottoms[i] + v / 2,
                    str(int(v)),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                )
        bottoms += vals_arr

    # Total label on top
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

    ax.set_ylabel("Component Count", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.set_title(
        "Full Machine Description Component Inventory per Epoch",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    _epoch_xticks(ax, labels, shots)
    _style_ax(ax)

    fig.suptitle(
        "JET Machine Description — Component Inventory",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_component_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_limiter_contours(limiters) -> None:
    """Figure 6: Limiter contour complexity + individual contour shapes."""
    n_lim = len(limiters)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor("white")

    # Top row: individual limiter contours
    for i in range(min(n_lim, 7)):
        row, col = divmod(i, 4)
        ax = axes[row, col]
        lim = limiters[i]
        name = lim["path"].split(":")[-1]
        r = lim.get("r_contour", [])
        z = lim.get("z_contour", [])
        _, color = LIMITER_NAMES.get(name, (name, "#888"))

        if r and z:
            ax.plot(r, z, color=color, linewidth=1.5)
            ax.fill(r, z, color=color, alpha=0.1)
            ax.set_aspect("equal")
            ax.set_xlim(1.5, 4.5)
            ax.set_ylim(-2.5, 2.5)

        display_name = LIMITER_NAMES.get(name, (name, "#888"))[0]
        ax.set_title(
            f"{display_name}\n({lim['n_points']} pts, shot {lim['first_shot']})",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_xlabel("R (m)", fontsize=7)
        ax.set_ylabel("Z (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
        _style_ax(ax)

    # Hide unused subplots
    for i in range(n_lim, 8):
        row, col = divmod(i, 4)
        axes[row, col].set_visible(False)

    # Bottom-right: bar chart of contour complexity
    if n_lim <= 4:
        ax = axes[1, 3]
    else:
        ax = axes[1, 3]
    ax.set_visible(True)
    lim_names = [lim["path"].split(":")[-1] for lim in limiters]
    lim_points = [lim["n_points"] for lim in limiters]
    colors = [LIMITER_NAMES.get(name, (name, "#888"))[1] for name in lim_names]
    ax.bar(range(n_lim), lim_points, color=colors, edgecolor="white", width=0.7)
    for j, (pts, _name) in enumerate(zip(lim_points, lim_names, strict=True)):
        ax.text(
            j, pts + 2, str(pts), ha="center", va="bottom", fontsize=7, color="#333"
        )
    ax.set_xticks(range(n_lim))
    ax.set_xticklabels(lim_names, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Contour Points", fontsize=8)
    ax.set_title("Contour Complexity", fontsize=9, fontweight="bold", loc="left")
    _style_ax(ax)

    fig.suptitle(
        "JET Limiter Contour Evolution — 7 Physical Configurations",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "jet_limiter_contours.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_summary(epochs, comp_counts, limiters) -> None:
    """Figure 7: Compact 4-panel summary."""
    n = len(epochs)
    x = np.arange(n)
    labels = [short_name(e["id"]) for e in epochs]
    shots = [e["first_shot"] for e in epochs]

    # Build limiter lookup
    limiter_lookup = {}
    for lim in limiters:
        name = lim["path"].split(":")[-1]
        limiter_lookup[name] = (lim["first_shot"], lim.get("last_shot"))

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: Wall + PF config combined
    ax = axes[0]
    for i, e in enumerate(epochs):
        lim = _find_limiter(limiter_lookup, e["first_shot"])
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
        pf_color = PF_CONFIG_COLORS.get(pf, "#ccc")
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
    ax.set_title("Wall & PF Configuration", fontsize=11, fontweight="bold", loc="left")
    _style_ax(ax)

    # Panel 2: Stacked component inventory
    ax = axes[1]
    comp_keys = ["magprobes", "pfcoils", "passives", "circuits", "flux_loops"]
    comp_labels_list = ["Mag. Probes", "PF Coils", "Passives", "Circuits", "Flux Loops"]
    bottoms = np.zeros(n)
    for key, label in zip(comp_keys, comp_labels_list, strict=True):
        vals = [comp_counts.get(e["id"], {}).get(key, 0) for e in epochs]
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
        bottoms += vals_arr
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("Component Inventory", fontsize=11, fontweight="bold", loc="left")
    _style_ax(ax)

    # Panel 3: Device XML versions
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
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title("Device XML Version", fontsize=11, fontweight="bold", loc="left")
    _style_ax(ax)

    # Panel 4: Probe events
    ax = axes[3]
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
    ax.axhline(y=0, color="#666", linewidth=0.5)
    ax.set_ylabel("Probe Changes", fontsize=8)
    ax.set_title("Probe Events", fontsize=11, fontweight="bold", loc="left")
    _style_ax(ax)

    _epoch_xticks(axes[3], labels, shots)

    fig.suptitle(
        "JET Machine Description Evolution — Summary",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    plt.tight_layout(h_pad=1.2)
    out = OUTPUT_DIR / "jet_epoch_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    print("Fetching data from knowledge graph...")
    epochs = fetch_epochs()
    if not epochs:
        print("No epochs found in graph")
        return

    comp_counts = fetch_component_counts()
    limiters = fetch_limiter_contours()
    jec2020_pf = fetch_jec2020_pf_coils()
    jec2020_summary = fetch_jec2020_summary()

    n = len(epochs)
    n_signals = sum(
        sum(
            comp_counts.get(e["id"], {}).get(k, 0)
            for k in ["magprobes", "pfcoils", "passives", "circuits", "flux_loops"]
        )
        for e in epochs
    )

    print(f"  {n} epochs, {len(limiters)} limiter configs")
    print(f"  {n_signals} DataNodes across EFIT++ epochs")
    print(f"  JEC2020: {jec2020_summary}")
    print(f"  {len(jec2020_pf)} JEC2020 PF coils")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_wall_limiter(epochs, limiters)
    plot_pf_coils(epochs, comp_counts, jec2020_pf)
    plot_magnetics(epochs, comp_counts)
    plot_passives_circuits(epochs, comp_counts)
    plot_component_overview(epochs, comp_counts)
    plot_limiter_contours(limiters)
    plot_summary(epochs, comp_counts, limiters)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
