"""Plot JET limiter contour evolution across structural epochs.

Queries Neo4j for all limiter DataNodes ingested by the DeviceXMLScanner
and plots their R,Z contours color-coded by era. Overlays PF coils and
magnetic probe positions when available.

Usage:
    uv run python scripts/plot_jet_limiter_epochs.py
    uv run python scripts/plot_jet_limiter_epochs.py --no-probes
"""

from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt

from imas_codex.graph.client import GraphClient

# Color palette for limiter epochs
EPOCH_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
]


def fetch_limiter_epochs() -> list[dict]:
    """Query all limiter DataNodes with R,Z contour data."""
    with GraphClient() as gc:
        results = list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet'
                  AND dn.path STARTS WITH 'jet:device_xml:limiter:'
                  AND dn.r_contour IS NOT NULL
                  AND dn.z_contour IS NOT NULL
                RETURN dn.path AS path,
                       dn.r_contour AS r_contour,
                       dn.z_contour AS z_contour,
                       dn.n_points AS n_points,
                       dn.first_shot AS first_shot,
                       dn.last_shot AS last_shot,
                       dn.provenance AS provenance
                ORDER BY dn.first_shot ASC
                """
            )
        )
    return results


def fetch_jec2020_limiter() -> dict | None:
    """Query JEC2020 high-res limiter if available."""
    with GraphClient() as gc:
        results = list(
            gc.query(
                """
                MATCH (dn:DataNode {id: 'jet:jec2020:limiter'})
                RETURN dn.r_contour AS r_contour,
                       dn.z_contour AS z_contour,
                       dn.n_points AS n_points
                """
            )
        )
    return results[0] if results else None


def fetch_pf_coils() -> list[dict]:
    """Query PF coil DataNodes for overlay."""
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet'
                  AND dn.system = 'PF'
                  AND dn.data_source_name IN ['jec2020_geometry', 'efit_device_xml']
                RETURN dn.path AS path,
                       dn.rCentre AS r,
                       dn.zCentre AS z,
                       dn.dR AS dr,
                       dn.dZ AS dz
                """
            )
        )


def fetch_probes() -> list[dict]:
    """Query magnetic probe positions for overlay."""
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (dn:DataNode)
                WHERE dn.facility_id = 'jet'
                  AND dn.system = 'MP'
                  AND dn.data_source_name = 'jec2020_geometry'
                RETURN dn.r AS r, dn.z AS z, dn.path AS path
                """
            )
        )


def plot_epochs(
    limiters: list[dict],
    jec2020: dict | None = None,
    pf_coils: list[dict] | None = None,
    probes: list[dict] | None = None,
) -> plt.Figure:
    """Create composite limiter epoch plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))

    # Plot limiter contours
    for i, lim in enumerate(limiters):
        r = lim["r_contour"]
        z = lim["z_contour"]
        if not r or not z:
            continue

        color = EPOCH_COLORS[i % len(EPOCH_COLORS)]
        name = lim["path"].split(":")[-1]
        first = lim.get("first_shot", "?")
        last = lim.get("last_shot", "?")
        label = f"{name} (shots {first}–{last})"

        ax.plot(r, z, color=color, linewidth=1.5, label=label, zorder=2)

    # Overlay JEC2020 high-res limiter
    if jec2020 and jec2020.get("r_contour"):
        ax.plot(
            jec2020["r_contour"],
            jec2020["z_contour"],
            color="black",
            linewidth=0.8,
            linestyle="--",
            label=f"JEC2020 ({jec2020.get('n_points', '?')} pts)",
            alpha=0.7,
            zorder=3,
        )

    # Overlay PF coils as rectangles
    if pf_coils:
        for coil in pf_coils:
            r, z = coil.get("r"), coil.get("z")
            dr, dz = coil.get("dr"), coil.get("dz")
            if r is None or z is None or dr is None or dz is None:
                continue
            if isinstance(r, list):
                r, z, dr, dz = r[0], z[0], dr[0], dz[0]
            rect = plt.Rectangle(
                (r - dr / 2, z - dz / 2),
                dr,
                dz,
                fill=False,
                edgecolor="gray",
                linewidth=0.5,
                alpha=0.6,
                zorder=1,
            )
            ax.add_patch(rect)

    # Overlay probe positions
    if probes:
        pr = [p["r"] for p in probes if p.get("r")]
        pz = [p["z"] for p in probes if p.get("z")]
        if pr:
            ax.scatter(
                pr,
                pz,
                s=8,
                c="red",
                marker="x",
                alpha=0.5,
                label=f"Mag probes ({len(pr)})",
                zorder=4,
            )

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("JET Limiter Contour Evolution (Shots 1–104522)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot JET limiter epoch evolution from graph data"
    )
    parser.add_argument(
        "--no-probes", action="store_true", help="Skip magnetic probe overlay"
    )
    parser.add_argument("--no-coils", action="store_true", help="Skip PF coil overlay")
    parser.add_argument(
        "-o", "--output", default="jet_limiter_epochs.png", help="Output file"
    )
    args = parser.parse_args()

    print("Querying limiter epochs...")
    limiters = fetch_limiter_epochs()
    if not limiters:
        print("No limiter DataNodes found in graph. Run ingestion first.")
        sys.exit(1)
    print(f"  Found {len(limiters)} limiter epochs")

    print("Querying JEC2020 limiter...")
    jec2020 = fetch_jec2020_limiter()
    if jec2020:
        print(f"  JEC2020 limiter: {jec2020.get('n_points', '?')} points")

    pf_coils = None
    if not args.no_coils:
        print("Querying PF coils...")
        pf_coils = fetch_pf_coils()
        print(f"  Found {len(pf_coils)} PF coils")

    probes = None
    if not args.no_probes:
        print("Querying magnetic probes...")
        probes = fetch_probes()
        print(f"  Found {len(probes)} probes")

    fig = plot_epochs(limiters, jec2020, pf_coils, probes)
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
