"""Composite JET geometry plot from all ingested graph data.

Queries Neo4j for ALL geometry data ingested by the device_xml scanner
and renders a publication-quality poloidal cross-section with layers:

  1. Limiter contours (7 versions, color-coded) + JEC2020 high-res
  2. PF coil rectangles (JEC2020 latest config)
  3. Magnetic probe positions (JEC2020, with orientation)
  4. MCFG sensor positions (CATIA CAD reference)
  5. Iron core boundary (96 segments)
  6. Flux loop positions
  7. Passive structure elements (rectangles)
  8. Epoch timeline (StructuralEpoch nodes with wall_configuration)

Usage:
    uv run python scripts/plot_jet_composite_geometry.py
    uv run python scripts/plot_jet_composite_geometry.py --layers limiter,pf,probes,timeline
    uv run python scripts/plot_jet_composite_geometry.py -o docs/images/jet_composite_geometry.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from imas_codex.graph.client import GraphClient

ALL_LAYERS = {"limiter", "pf", "probes", "mcfg", "iron", "flux", "passive", "timeline"}

# Color per limiter version — consistent across plot and timeline
EPOCH_COLORS = [
    "#7f7f7f",  # Limiter (pre-divertor, grey)
    "#17becf",  # Mk1 (teal)
    "#1f77b4",  # Mk2A
    "#2ca02c",  # Mk2GB
    "#9467bd",  # Mk2GB-SR
    "#d62728",  # Mk2HD
    "#ff7f0e",  # Mk2ILW
    "#8c564b",  # spare
]

LIMITER_STYLE = {"linestyle": "--", "linewidth": 2.5}
DIVERTOR_STYLE = {"linestyle": "-", "linewidth": 1.5}


def query_limiters() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet'
                  AND dn.path STARTS WITH 'jet:device_xml:limiter:'
                  AND dn.r_contour IS NOT NULL
                OPTIONAL MATCH (se:StructuralEpoch)-[:USES_LIMITER]->(dn)
                WITH dn, collect(DISTINCT se.wall_configuration) AS wall_configs,
                     min(se.first_shot) AS epoch_first_shot
                RETURN dn.path AS path, dn.r_contour AS r, dn.z_contour AS z,
                       dn.first_shot AS first_shot, dn.last_shot AS last_shot,
                       dn.n_points AS n_points,
                       CASE WHEN 'divertor' IN wall_configs THEN 'divertor'
                            WHEN 'limiter' IN wall_configs THEN 'limiter'
                            ELSE 'divertor' END AS wall_config
                ORDER BY dn.first_shot ASC
                """
            )
        )


def query_epochs() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (se:StructuralEpoch {data_source_name: 'device_xml'})
                OPTIONAL MATCH (se)-[:USES_LIMITER]->(lim:DataNode)
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.last_shot AS last_shot,
                       se.description AS description,
                       se.wall_configuration AS wall_config,
                       lim.path AS limiter_path
                ORDER BY se.first_shot ASC
                """
            )
        )


def query_jec2020_limiter() -> dict | None:
    with GraphClient() as gc:
        results = list(
            gc.query(
                """
                MATCH (dn:DataNode {id: 'jet:jec2020:limiter'})
                RETURN dn.r_contour AS r, dn.z_contour AS z, dn.n_points AS n_points
                """
            )
        )
    return results[0] if results else None


def query_pf_coils() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet' AND dn.system = 'PF'
                RETURN dn.path AS path, dn.rCentre AS r, dn.zCentre AS z,
                       dn.dR AS dr, dn.dZ AS dz, dn.data_source_name AS source
                """
            )
        )


def query_probes() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet' AND dn.system = 'MP'
                  AND dn.data_source_name = 'jec2020_geometry'
                RETURN dn.r AS r, dn.z AS z, dn.angle AS angle, dn.path AS path
                """
            )
        )


def query_mcfg_sensors() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet'
                  AND dn.data_source_name = 'sensor_calibration'
                RETURN dn.r AS r, dn.z AS z, dn.angle AS angle,
                       dn.jpf_name AS jpf_name, dn.path AS path
                """
            )
        )


def query_iron_boundary() -> dict | None:
    with GraphClient() as gc:
        results = list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet' AND dn.system = 'FE'
                RETURN dn.r_contour AS r, dn.z_contour AS z,
                       dn.permeabilities AS permeabilities
                """
            )
        )
    return results[0] if results else None


def query_flux_loops() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet' AND dn.system = 'FL'
                RETURN dn.r AS r, dn.z AS z, dn.path AS path
                """
            )
        )


def query_passive() -> list[dict]:
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet' AND dn.system = 'PS'
                RETURN dn.path AS path,
                       dn.r AS r, dn.z AS z,
                       dn.dr AS dr, dn.dz AS dz
                """
            )
        )


def draw_limiter_layer(ax, limiters, jec2020):
    for i, lim in enumerate(limiters):
        r, z = lim.get("r"), lim.get("z")
        if not r or not z:
            continue
        color = EPOCH_COLORS[i % len(EPOCH_COLORS)]
        name = lim["path"].split(":")[-1]
        first = lim.get("first_shot", "?")
        last = lim.get("last_shot", "?")
        wall = lim.get("wall_config", "divertor")
        style = LIMITER_STYLE if wall == "limiter" else DIVERTOR_STYLE
        suffix = " [limiter]" if wall == "limiter" else ""
        ax.plot(
            r,
            z,
            color=color,
            label=f"{name} ({first}–{last}){suffix}",
            zorder=2,
            **style,
        )

    if jec2020 and jec2020.get("r"):
        ax.plot(
            jec2020["r"],
            jec2020["z"],
            "k--",
            lw=0.8,
            alpha=0.7,
            label=f"JEC2020 ({jec2020.get('n_points', '?')} pts)",
            zorder=3,
        )


def _draw_rect_elements(
    ax, r, z, dr, dz, facecolor, edgecolor, linewidth, alpha, zorder
):
    """Draw rectangle(s) for scalar or array-valued geometry properties."""
    if isinstance(r, list):
        for ri, zi, dri, dzi in zip(r, z, dr, dz, strict=True):
            ax.add_patch(
                Rectangle(
                    (ri - dri / 2, zi - dzi / 2),
                    dri,
                    dzi,
                    fill=True,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
    else:
        ax.add_patch(
            Rectangle(
                (r - dr / 2, z - dz / 2),
                dr,
                dz,
                fill=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
        )


def draw_pf_layer(ax, coils):
    for coil in coils:
        r, z = coil.get("r"), coil.get("z")
        dr, dz = coil.get("dr"), coil.get("dz")
        if r is None or z is None or dr is None or dz is None:
            continue
        _draw_rect_elements(
            ax,
            r,
            z,
            dr,
            dz,
            facecolor="#aec7e8",
            edgecolor="#1f77b4",
            linewidth=0.5,
            alpha=0.6,
            zorder=5,
        )


def draw_probe_layer(ax, probes):
    if not probes:
        return
    r = [p["r"] for p in probes if p.get("r") is not None]
    z = [p["z"] for p in probes if p.get("z") is not None]
    if r:
        ax.scatter(
            r,
            z,
            s=12,
            c="red",
            marker="x",
            alpha=0.6,
            label=f"JEC2020 probes ({len(r)})",
            zorder=6,
        )


def draw_mcfg_layer(ax, sensors):
    if not sensors:
        return
    r = [s["r"] for s in sensors if s.get("r") is not None]
    z = [s["z"] for s in sensors if s.get("z") is not None]
    if r:
        ax.scatter(
            r,
            z,
            s=6,
            c="green",
            marker=".",
            alpha=0.4,
            label=f"MCFG sensors ({len(r)})",
            zorder=4,
        )


def draw_iron_layer(ax, iron):
    if not iron or not iron.get("r"):
        return
    r, z = iron["r"], iron["z"]
    # Close the contour
    rc = list(r) + [r[0]]
    zc = list(z) + [z[0]]
    ax.plot(
        rc,
        zc,
        color="#8c564b",
        lw=1.0,
        alpha=0.7,
        label=f"Iron core ({len(r)} pts)",
        zorder=3,
    )


def draw_flux_layer(ax, loops):
    if not loops:
        return
    r = [fl["r"] for fl in loops if fl.get("r") is not None]
    z = [fl["z"] for fl in loops if fl.get("z") is not None]
    if r:
        ax.scatter(
            r,
            z,
            s=20,
            c="blue",
            marker="o",
            facecolors="none",
            alpha=0.5,
            label=f"Flux loops ({len(r)})",
            zorder=6,
        )


def draw_passive_layer(ax, passive):
    for elem in passive:
        r, z = elem.get("r"), elem.get("z")
        dr, dz = elem.get("dr"), elem.get("dz")
        if r is None or z is None or dr is None or dz is None:
            continue
        _draw_rect_elements(
            ax,
            r,
            z,
            dr,
            dz,
            facecolor="#d9d9d9",
            edgecolor="#777777",
            linewidth=0.3,
            alpha=0.4,
            zorder=1,
        )


def draw_timeline(ax, epochs):
    """Draw epoch timeline showing wall configuration evolution."""
    if not epochs:
        return

    ax.set_xlim(0, 100000)
    ax.set_ylim(-0.5, len(epochs) - 0.5)
    ax.invert_yaxis()

    for i, ep in enumerate(epochs):
        first = ep.get("first_shot", 0) or 0
        last = ep.get("last_shot") or 100000
        wall = ep.get("wall_config", "divertor")

        color = "#e67e22" if wall == "limiter" else "#3498db"
        alpha = 0.9 if wall == "limiter" else 0.7

        width = last - first
        rect = FancyBboxPatch(
            (first, i - 0.35),
            width,
            0.7,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # Label inside bar
        version = ep["id"].split(":")[-1]
        label = f"{version}"
        mid = first + width / 2
        ax.text(
            mid,
            i,
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            color="white",
            fontweight="bold",
        )

    ax.set_xlabel("Shot number", fontsize=10)
    ax.set_title("Machine Description Epochs", fontsize=11)
    ax.set_yticks([])

    # Mark eras
    ax.axvline(x=26087, color="#e67e22", ls=":", lw=1, alpha=0.5)
    ax.annotate(
        "Divertor\ninstalled",
        xy=(26087, -0.3),
        fontsize=7,
        ha="center",
        va="bottom",
        color="#e67e22",
    )

    # Add shot number ticks
    ax.set_xticks(np.arange(0, 100001, 10000))
    ax.tick_params(axis="x", labelsize=7)


def main():
    parser = argparse.ArgumentParser(
        description="Composite JET geometry plot from graph data"
    )
    parser.add_argument(
        "--layers",
        default=",".join(sorted(ALL_LAYERS)),
        help=f"Comma-separated layers to draw (default: all). Options: {','.join(sorted(ALL_LAYERS))}",
    )
    parser.add_argument(
        "-o", "--output", default="docs/images/jet_composite_geometry.png"
    )
    args = parser.parse_args()

    layers = set(args.layers.split(","))

    has_timeline = "timeline" in layers

    if has_timeline:
        fig, (ax, ax_tl) = plt.subplots(
            2,
            1,
            figsize=(14, 20),
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax_tl = None

    drawn = []

    if "limiter" in layers:
        print("Querying limiters...")
        limiters = query_limiters()
        jec2020 = query_jec2020_limiter()
        draw_limiter_layer(ax, limiters, jec2020)
        n = len(limiters) + (1 if jec2020 else 0)
        drawn.append(f"limiters: {n}")

    if "pf" in layers:
        print("Querying PF coils...")
        coils = query_pf_coils()
        draw_pf_layer(ax, coils)
        drawn.append(f"PF coils: {len(coils)}")

    if "probes" in layers:
        print("Querying magnetic probes...")
        probes = query_probes()
        draw_probe_layer(ax, probes)
        drawn.append(f"probes: {len(probes)}")

    if "mcfg" in layers:
        print("Querying MCFG sensors...")
        sensors = query_mcfg_sensors()
        draw_mcfg_layer(ax, sensors)
        drawn.append(f"MCFG sensors: {len(sensors)}")

    if "iron" in layers:
        print("Querying iron boundary...")
        iron = query_iron_boundary()
        draw_iron_layer(ax, iron)
        drawn.append(f"iron: {'yes' if iron else 'no'}")

    if "flux" in layers:
        print("Querying flux loops...")
        loops = query_flux_loops()
        draw_flux_layer(ax, loops)
        drawn.append(f"flux loops: {len(loops)}")

    if "passive" in layers:
        print("Querying passive structures...")
        passive = query_passive()
        draw_passive_layer(ax, passive)
        drawn.append(f"passive: {len(passive)}")

    if has_timeline and ax_tl is not None:
        print("Querying epochs...")
        epochs = query_epochs()
        draw_timeline(ax_tl, epochs)
        drawn.append(f"epochs: {len(epochs)}")

    ax.set_xlabel("R [m]", fontsize=12)
    ax.set_ylabel("Z [m]", fontsize=12)
    ax.set_title(
        "JET Machine Description — All Epochs (Limiter + Divertor)", fontsize=14
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)

    print(f"\nSaved: {args.output}")
    for d in drawn:
        print(f"  {d}")


if __name__ == "__main__":
    main()
