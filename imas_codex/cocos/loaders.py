"""Data loaders for COCOS determination.

Provides facility-agnostic loading of physics quantities from various sources.
The COCOS calculation is deterministic - these loaders just extract the
required physics values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EquilibriumData:
    """Physics quantities needed for COCOS determination.

    These are the minimum required inputs for determine_cocos() and
    validate_cocos_from_data(). The source field tracks data provenance.

    All quantities should be in SI units:
    - psi: Weber [Wb] or Wb/rad depending on e_bp
    - ip: Ampere [A]
    - b0: Tesla [T]
    - q: dimensionless
    - dp_dpsi: Pa/Wb
    """

    psi_axis: float
    psi_edge: float
    ip: float
    b0: float
    q: float | None = None
    dp_dpsi: float | None = None
    source: str = ""


def load_from_imas_ids(equilibrium_ids) -> EquilibriumData:
    """Load equilibrium data from an IMAS IDS object.

    Extracts physics quantities from the first time_slice of an
    equilibrium IDS for COCOS determination.

    Args:
        equilibrium_ids: An IMAS equilibrium IDS object (e.g., from imas.imasdef.equilibrium)

    Returns:
        EquilibriumData with extracted physics quantities.

    Raises:
        ValueError: If required quantities are missing from the IDS.

    Example:
        >>> import imas
        >>> entry = imas.DBEntry("...")
        >>> eq = entry.get("equilibrium")
        >>> data = load_from_imas_ids(eq)
        >>> cocos, conf = determine_cocos(**data.__dict__)
    """
    try:
        ts = equilibrium_ids.time_slice[0]
        gq = ts.global_quantities

        # Required quantities
        psi_axis = float(gq.psi_axis)
        psi_edge = float(gq.psi_boundary)
        ip = float(gq.ip)

        # B0 at magnetic axis
        try:
            b0 = float(gq.magnetic_axis.b_field_tor)
        except (AttributeError, IndexError):
            b0 = float(equilibrium_ids.vacuum_toroidal_field.b0[0])

        # Optional: q at mid-radius
        q = None
        try:
            profiles_1d = ts.profiles_1d
            if hasattr(profiles_1d, "q") and len(profiles_1d.q) > 0:
                mid_idx = len(profiles_1d.q) // 2
                q = float(profiles_1d.q[mid_idx])
        except (AttributeError, IndexError):
            pass

        # Optional: dp/dpsi at mid-radius
        dp_dpsi = None
        try:
            profiles_1d = ts.profiles_1d
            if (
                hasattr(profiles_1d, "dpressure_dpsi")
                and len(profiles_1d.dpressure_dpsi) > 0
            ):
                mid_idx = len(profiles_1d.dpressure_dpsi) // 2
                dp_dpsi = float(profiles_1d.dpressure_dpsi[mid_idx])
        except (AttributeError, IndexError):
            pass

        return EquilibriumData(
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=dp_dpsi,
            source="imas",
        )

    except (AttributeError, IndexError) as e:
        msg = f"Failed to extract equilibrium data from IDS: {e}"
        raise ValueError(msg) from e


def load_from_eqdsk(filepath: Path | str) -> EquilibriumData:
    """Load equilibrium data from a G-EQDSK file.

    Parses the standard G-EQDSK format used by many equilibrium codes
    (EFIT, CHEASE, etc.) and extracts physics quantities for COCOS
    determination.

    Args:
        filepath: Path to G-EQDSK file.

    Returns:
        EquilibriumData with extracted physics quantities.

    Raises:
        ValueError: If file cannot be parsed or required quantities are missing.
        FileNotFoundError: If file does not exist.

    Example:
        >>> data = load_from_eqdsk("g012345.00100")
        >>> cocos, conf = determine_cocos(**data.__dict__)

    Note:
        G-EQDSK format stores ψ in Wb/rad (divided by 2π).
        The e_bp parameter in COCOS determines if this normalization applies.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    try:
        with filepath.open() as f:
            lines = f.readlines()

        # Parse header line (first line contains case info and grid dimensions)
        header = lines[0].split()
        # Last 3 entries are idum, nw, nh (grid dimensions, not used here)
        _ = int(header[-2])  # noqa: F841 - nw kept for format documentation

        # Parse second line: rdim, zdim, rcentr, rleft, zmid
        _parse_fortran_floats(lines[1])  # rcentr not needed for COCOS

        # Parse third line: rmaxis, zmaxis, simag, sibry, bcentr
        line3 = _parse_fortran_floats(lines[2])
        psi_axis = line3[2]  # ψ at magnetic axis (Wb/rad)
        psi_edge = line3[3]  # ψ at boundary (Wb/rad)
        b0 = line3[4]  # B0 at rcentr (T)

        # Parse fourth line: current, simag, xdum, rmaxis, xdum
        line4 = _parse_fortran_floats(lines[3])
        ip = line4[0]  # Plasma current (A)

        # Parse q profile if available (after boundary data)
        q = None
        try:
            # Skip to q profile section (after psi, p, ffprime, pprime, rbdry, zbdry)
            # This is complex due to variable-length boundary data
            # For now, skip q extraction from raw EQDSK
            pass
        except Exception:
            pass

        return EquilibriumData(
            psi_axis=psi_axis,
            psi_edge=psi_edge,
            ip=ip,
            b0=b0,
            q=q,
            dp_dpsi=None,  # Would need pprime and coordinate transform
            source=f"eqdsk:{filepath.name}",
        )

    except Exception as e:
        msg = f"Failed to parse EQDSK file {filepath}: {e}"
        raise ValueError(msg) from e


def _parse_fortran_floats(line: str) -> list[float]:
    """Parse Fortran-formatted floating point numbers.

    Handles both space-separated and concatenated scientific notation
    (e.g., "1.234E+00-5.678E-01" without space separator).
    """
    import re

    # Try simple split first
    parts = line.split()
    if len(parts) >= 5:
        try:
            return [float(p) for p in parts[:5]]
        except ValueError:
            pass

    # Handle concatenated scientific notation
    pattern = r"[+-]?\d+\.?\d*[eEdD]?[+-]?\d*"
    matches = re.findall(pattern, line)
    return [float(m.replace("D", "E").replace("d", "e")) for m in matches if m]
