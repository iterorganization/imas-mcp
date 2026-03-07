"""Plot JET poloidal cross-section geometry from graph-persisted data.

Queries Neo4j for magnetic probes, PF coils, passive structures, and limiter
contours ingested by the DeviceXMLScanner, and renders publication-quality
poloidal cross-section plots for each distinct geometrical era.

The script identifies which limiter contour is active for each epoch by
overlapping the epoch shot range with the limiter shot validity windows,
and only produces plots for epochs with significantly different geometry.

Usage:
    uv run python scripts/plot_jet_poloidal.py            # all significant eras
    uv run python scripts/plot_jet_poloidal.py p89440     # single epoch
"""

from __future__ import annotations

import sys

import numpy as np

from imas_codex.graph.client import GraphClient


def _parse_multi_value(raw: str | float | int | None) -> list[float]:
    """Parse a scalar or space-separated string into a list of floats."""
    if raw is None:
        return []
    if isinstance(raw, int | float):
        return [float(raw)]
    vals = []
    for token in str(raw).split():
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals


def _active_limiter(limiters: list[dict], epoch_first_shot: int) -> dict | None:
    """Return the limiter contour whose shot range covers epoch_first_shot."""
    for lim in limiters:
        fs = lim.get("first_shot")
        ls = lim.get("last_shot")
        if fs is None:
            continue
        if epoch_first_shot >= fs and (ls is None or epoch_first_shot <= ls):
            return lim
    return None


def query_geometry(epoch: str = "p89440") -> dict:
    """Query all geometry data from the graph for a given epoch."""
    facility = "jet"
    prefix = f"{facility}:device_xml:{epoch}"

    with GraphClient() as gc:
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

        limiters = gc.query(
            """
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH 'jet:device_xml:limiter:'
            RETURN dn.path AS path,
                   dn.r_contour AS r, dn.z_contour AS z,
                   dn.n_points AS n_points,
                   dn.first_shot AS first_shot,
                   dn.last_shot AS last_shot
            ORDER BY dn.first_shot
            """,
        )

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


def get_significant_epochs() -> list[dict]:
    """Return representative epochs for each distinct geometry era.

    Groups epochs by their active limiter contour and selects one
    representative per group (the first epoch in each limiter era).
    """
    with GraphClient() as gc:
        epochs = gc.query("""
            MATCH (se:StructuralEpoch)
            WHERE se.facility_id = 'jet'
            RETURN se.id AS id,
                   se.first_shot AS first_shot,
                   se.last_shot AS last_shot,
                   se.description AS description
            ORDER BY toInteger(se.first_shot)
        """)

        limiters = gc.query("""
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH 'jet:device_xml:limiter:'
            RETURN dn.path AS path,
                   dn.first_shot AS first_shot,
                   dn.last_shot AS last_shot
            ORDER BY dn.first_shot
        """)

    # Group epochs by their active limiter
    seen_limiters: dict[str, dict] = {}
    for ep in epochs:
        fs = ep["first_shot"]
        if fs is None:
            continue
        active = _active_limiter(limiters, fs)
        lim_name = active["path"].split(":")[-1] if active else "unknown"
        if lim_name not in seen_limiters:
            epoch_name = ep["id"].split(":")[-1]
            seen_limiters[lim_name] = {
                "epoch": epoch_name,
                "limiter": lim_name,
                "first_shot": ep["first_shot"],
                "last_shot": ep["last_shot"],
                "description": ep["description"],
            }

    return list(seen_limiters.values())


def plot_poloidal(data: dict, epoch: str = "p89440") -> None:
    """Create a poloidal cross-section plot from graph data."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))

    # --- Limiter contour (active for this epoch) ---
    epoch_info = data.get("epoch", {})
    epoch_first_shot = epoch_info.get("first_shot", 0)
    active_lim = _active_limiter(data["limiters"], epoch_first_shot)

    if active_lim:
        name = active_lim["path"].split(":")[-1]
        r = np.array(active_lim["r"], dtype=float)
        z = np.array(active_lim["z"], dtype=float)
        ax.plot(
            r,
            z,
            "-",
            color="#2c3e50",
            linewidth=2,
            label=f"Limiter: {name} ({active_lim['n_points']} pts)",
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
    min_dim = 0.06  # minimum visual size in metres
    passive_patches = []
    for ps in data["passives"]:
        r, z = ps["r"], ps["z"]
        dr, dz = ps.get("dr", min_dim), ps.get("dz", min_dim)
        if r is None or z is None:
            continue
        dr = max(dr or min_dim, min_dim)
        dz = max(dz or min_dim, min_dim)
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
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Auto-scale to data extent with padding
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad = 0.3
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    # Add data source annotation
    lim_name = active_lim["path"].split(":")[-1] if active_lim else "none"
    ax.annotate(
        "Data source: Neo4j graph (device_xml scanner)\n"
        f"Probes: {len(mp_r)} | Coils: {len(pf_patches)} | "
        f"Passive: {len(passive_patches)} | "
        f"Limiter: {lim_name}",
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


def plot_limiter_comparison() -> None:
    """Overlay all first-wall / divertor contours, color-coded by era."""
    import matplotlib.pyplot as plt

    with GraphClient() as gc:
        limiters = gc.query("""
            MATCH (dn:DataNode)
            WHERE dn.path STARTS WITH 'jet:device_xml:limiter:'
            RETURN dn.path AS path,
                   dn.r_contour AS r, dn.z_contour AS z,
                   dn.n_points AS n_points,
                   dn.first_shot AS first_shot,
                   dn.last_shot AS last_shot
            ORDER BY dn.first_shot
        """)

    colors = {
        "Mk2GB": "#e67e22",  # orange
        "Mk2HD": "#2980b9",  # blue
        "Mk2ILW": "#27ae60",  # green
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))

    for lim in limiters:
        name = lim["path"].split(":")[-1]
        r = np.array(lim["r"], dtype=float)
        z = np.array(lim["z"], dtype=float)
        fs = lim["first_shot"]
        ls = lim["last_shot"]
        color = colors.get(name, "#333")
        shot_range = f"shots {fs}–{ls}" if ls else f"shots {fs}–end"
        ax.plot(
            r,
            z,
            "-",
            color=color,
            linewidth=2.0,
            label=f"{name} ({lim['n_points']} pts, {shot_range})",
            zorder=2,
        )

    ax.set_title(
        "JET First Wall & Divertor — All Configurations\n"
        "Mk2GB → Mk2HD → Mk2ILW (ITER-Like Wall)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("R [m]", fontsize=12)
    ax.set_ylabel("Z [m]", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad = 0.3
    ax.set_xlim(max(0, xmin - pad), xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.annotate(
        "Data source: Neo4j graph (device_xml scanner)",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=7,
        color="#666",
        fontstyle="italic",
    )

    plt.tight_layout()
    out = "jet_limiter_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main():
    # --compare: overlay all limiter contours on one plot
    # <epoch>: single epoch poloidal plot
    # (no args): comparison plot
    if len(sys.argv) > 1 and sys.argv[1] != "--compare":
        epochs_to_plot = [sys.argv[1]]
        for epoch in epochs_to_plot:
            print(f"Querying graph for JET epoch {epoch}...")
            data = query_geometry(epoch)
            print(
                f"  Magnetic probes: {len(data['magprobes'])}\n"
                f"  PF coils: {len(data['pfcoils'])}\n"
                f"  Passive structures: {len(data['passives'])}\n"
                f"  Limiters available: {len(data['limiters'])}\n"
                f"  Epoch: {data['epoch']}"
            )
            plot_poloidal(data, epoch)
    else:
        print("Generating first-wall / divertor comparison plot...")
        plot_limiter_comparison()


if __name__ == "__main__":
    main()
