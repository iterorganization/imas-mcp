"""Plot JET poloidal cross-section geometry from graph-persisted data.

Queries Neo4j for magnetic probes, PF coils, passive structures, and limiter
contours ingested by the DeviceXMLScanner, and renders a publication-quality
poloidal cross-section plot.

Usage:
    uv run python scripts/plot_jet_poloidal.py
"""

from __future__ import annotations

import numpy as np

from imas_codex.graph.client import GraphClient


def _parse_multi_value(raw: str | float | int | None) -> list[float]:
    """Parse a scalar or space-separated string into a list of floats."""
    if raw is None:
        return []
    if isinstance(raw, int | float):
        return [float(raw)]
    # String of space/newline separated values
    vals = []
    for token in str(raw).split():
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals


def query_geometry(epoch: str = "p89440") -> dict:
    """Query all geometry data from the graph for a given epoch."""
    facility = "jet"
    prefix = f"{facility}:device_xml:{epoch}"

    with GraphClient() as gc:
        # Magnetic probes — scalar r, z, angle
        magprobes = gc.query(
            """
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH $prefix + ':magprobes'
            RETURN dn.path AS path,
                   toFloat(dn.r) AS r, toFloat(dn.z) AS z,
                   toFloat(dn.angle) AS angle
            ORDER BY dn.path
            """,
            prefix=prefix,
        )

        # PF coils — may have multi-element r/z/dr/dz strings
        pfcoils = gc.query(
            """
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH $prefix + ':pfcoils'
            RETURN dn.path AS path,
                   dn.r AS r, dn.z AS z,
                   dn.dr AS dr, dn.dz AS dz
            ORDER BY dn.path
            """,
            prefix=prefix,
        )

        # Passive structures — scalar r, z, dr, dz
        passives = gc.query(
            """
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH $prefix + ':pfpassive'
            RETURN dn.path AS path,
                   toFloat(dn.r) AS r, toFloat(dn.z) AS z,
                   toFloat(dn.dr) AS dr, toFloat(dn.dz) AS dz,
                   toFloat(dn.resistance) AS resistance
            ORDER BY dn.path
            """,
            prefix=prefix,
        )

        # Limiter contours — R,Z arrays on DataNode
        limiters = gc.query(
            """
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH 'jet:device_xml:limiter:'
            RETURN dn.path AS path,
                   dn.r_contour AS r, dn.z_contour AS z,
                   dn.n_points AS n_points
            ORDER BY dn.path
            """,
        )

        # Epoch metadata
        epoch_info = gc.query(
            """
            MATCH (se:StructuralEpoch {id: $id})
            RETURN se.first_shot AS first_shot,
                   se.last_shot AS last_shot,
                   se.description AS description
            """,
            id=f"{facility}:device_xml:{epoch}",
        )

    return {
        "magprobes": magprobes,
        "pfcoils": pfcoils,
        "passives": passives,
        "limiters": limiters,
        "epoch": epoch_info[0] if epoch_info else {},
    }


def plot_poloidal(data: dict, epoch: str = "p89440") -> None:
    """Create a poloidal cross-section plot from graph data."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))

    # --- Limiter contours ---
    colors = {"Mk2ILW": "#2c3e50", "Mk2HD": "#7f8c8d", "Mk2GB": "#bdc3c7"}
    for lim in data["limiters"]:
        name = lim["path"].split(":")[-1]
        r = np.array(lim["r"], dtype=float)
        z = np.array(lim["z"], dtype=float)
        style = {"Mk2ILW": "-", "Mk2HD": "--", "Mk2GB": ":"}
        ax.plot(
            r,
            z,
            style.get(name, "-"),
            color=colors.get(name, "#555"),
            linewidth=2 if name == "Mk2ILW" else 1,
            label=f"Limiter: {name} ({lim['n_points']} pts)",
            zorder=1,
        )

    # --- PF coils (rectangles) ---
    pf_patches = []
    pf_labels = []
    for coil in data["pfcoils"]:
        r_vals = _parse_multi_value(coil["r"])
        z_vals = _parse_multi_value(coil["z"])
        dr_vals = _parse_multi_value(coil["dr"])
        dz_vals = _parse_multi_value(coil["dz"])

        if not r_vals or not z_vals:
            continue

        # For multi-filament coils, compute bounding box
        r_center = np.mean(r_vals)
        z_center = np.mean(z_vals)

        if dr_vals and dz_vals:
            # Total extent from filament positions + element sizes
            r_min = min(r_vals) - max(dr_vals) / 2
            r_max = max(r_vals) + max(dr_vals) / 2
            z_min = min(z_vals) - max(dz_vals) / 2
            z_max = max(z_vals) + max(dz_vals) / 2
            width = r_max - r_min
            height = z_max - z_min
        else:
            width = 0.1
            height = 0.1

        rect = mpatches.Rectangle(
            (r_center - width / 2, z_center - height / 2),
            width,
            height,
        )
        pf_patches.append(rect)

        # Label coil
        coil_id = coil["path"].split(":")[-1]
        pf_labels.append((r_center, z_center, coil_id))

    if pf_patches:
        pc = PatchCollection(
            pf_patches,
            facecolor="#e74c3c",
            edgecolor="#c0392b",
            alpha=0.6,
            linewidth=1.5,
            zorder=3,
        )
        ax.add_collection(pc)
        for r, z, label in pf_labels:
            ax.annotate(
                label,
                (r, z),
                fontsize=5,
                ha="center",
                va="center",
                color="#c0392b",
                fontweight="bold",
            )

    # --- Passive structures (rectangles) ---
    passive_patches = []
    for ps in data["passives"]:
        r, z = ps["r"], ps["z"]
        dr, dz = ps.get("dr", 0.05), ps.get("dz", 0.05)
        if r is None or z is None:
            continue
        dr = dr or 0.05
        dz = dz or 0.05
        rect = mpatches.Rectangle(
            (r - dr / 2, z - dz / 2),
            dr,
            dz,
        )
        passive_patches.append(rect)

    if passive_patches:
        pc = PatchCollection(
            passive_patches,
            facecolor="#95a5a6",
            edgecolor="#7f8c8d",
            alpha=0.5,
            linewidth=0.5,
            zorder=2,
        )
        ax.add_collection(pc)

    # --- Magnetic probes (scatter) ---
    mp_r = [mp["r"] for mp in data["magprobes"] if mp["r"] is not None]
    mp_z = [mp["z"] for mp in data["magprobes"] if mp["z"] is not None]
    mp_angle = [mp.get("angle", 0) for mp in data["magprobes"] if mp["r"] is not None]

    if mp_r:
        ax.scatter(
            mp_r,
            mp_z,
            c="#3498db",
            s=12,
            marker="^",
            zorder=4,
            label=f"Magnetic probes ({len(mp_r)})",
            edgecolors="#2980b9",
            linewidth=0.5,
        )

        # Draw orientation arrows for a subset
        arrow_scale = 0.08
        for i in range(0, len(mp_r), 5):
            angle_rad = np.radians(mp_angle[i])
            dx = arrow_scale * np.cos(angle_rad)
            dy = arrow_scale * np.sin(angle_rad)
            ax.annotate(
                "",
                xy=(mp_r[i] + dx, mp_z[i] + dy),
                xytext=(mp_r[i], mp_z[i]),
                arrowprops={"arrowstyle": "->", "color": "#2980b9", "lw": 0.5},
            )

    # --- Proxy artists for legend ---
    if pf_patches:
        ax.plot(
            [],
            [],
            "s",
            color="#e74c3c",
            markersize=8,
            label=f"PF coils ({len(pf_patches)})",
        )
    if passive_patches:
        ax.plot(
            [],
            [],
            "s",
            color="#95a5a6",
            markersize=8,
            label=f"Passive ({len(passive_patches)})",
        )

    # --- Formatting ---
    epoch_info = data.get("epoch", {})
    first = epoch_info.get("first_shot", "?")
    last = epoch_info.get("last_shot", "end")
    desc = epoch_info.get("description", "")
    title = f"JET Poloidal Cross-Section — Epoch {epoch}\n"
    title += f"Shots {first}–{last or 'end'}"
    if desc:
        title += f"\n{desc}"

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("R [m]", fontsize=12)
    ax.set_ylabel("Z [m]", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Auto-scale to data extent with padding
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad = 0.3
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Add data source annotation
    ax.annotate(
        "Data source: Neo4j graph (device_xml scanner)\n"
        f"Probes: {len(mp_r)} | Coils: {len(pf_patches)} | "
        f"Passive: {len(passive_patches)} | "
        f"Limiters: {len(data['limiters'])}",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="#666",
        fontstyle="italic",
    )

    plt.tight_layout()
    out = f"jet_poloidal_{epoch}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    epoch = "p89440"
    print(f"Querying graph for JET epoch {epoch}...")
    data = query_geometry(epoch)

    print(
        f"  Magnetic probes: {len(data['magprobes'])}\n"
        f"  PF coils: {len(data['pfcoils'])}\n"
        f"  Passive structures: {len(data['passives'])}\n"
        f"  Limiters: {len(data['limiters'])}\n"
        f"  Epoch: {data['epoch']}"
    )

    plot_poloidal(data, epoch)


if __name__ == "__main__":
    main()
