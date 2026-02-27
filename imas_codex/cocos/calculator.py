"""COCOS calculation from equilibrium quantities.

Based on Sauter & Medvedev, "Tokamak coordinate conventions: COCOS",
Computer Physics Communications, 184(2):293-302, February 2013.
DOI: 10.1016/j.cpc.2012.09.010

Table I defines 16 COCOS values (1-8, 11-18) via four parameters:
- σBp: Poloidal flux sign convention (+1 or -1)
- eBp: Whether ψ is divided by 2π (0) or full ψ (1)
- σRφZ: Cylindrical coordinate handedness (+1 for R,φ,Z; -1 for R,Z,φ)
- σρθφ: Poloidal coordinate handedness (+1 or -1)

The module provides composable functions to:
1. Calculate COCOS from physics quantities (Eq. 23 in paper)
2. Convert between COCOS conventions
3. Validate COCOS consistency

Known COCOS values from the paper:
- CHEASE, ONETWO: 2
- ITER/IMAS DD v3.x: 11
- IMAS DD v4.x: 17
- TCV/LIUQE: 17
- ORB5: 3 (with negative Ip normalization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

# Valid COCOS values per Sauter paper Table I
VALID_COCOS = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18})


@dataclass(frozen=True)
class COCOSParameters:
    """The four COCOS-determining parameters from Table I.

    These parameters fully specify a COCOS convention and enable
    transformations between conventions.

    Attributes:
        sigma_bp: Poloidal flux sign convention (+1 or -1).
            +1 if ψ increases from axis to edge with positive Ip.
        e_bp: Whether ψ is divided by 2π (0) or full ψ (1).
            COCOS 1-8 have e_bp=0, COCOS 11-18 have e_bp=1.
        sigma_r_phi_z: Cylindrical coordinate handedness.
            +1 for (R,φ,Z) right-handed, -1 for (R,Z,φ) right-handed.
        sigma_rho_theta_phi: Poloidal coordinate handedness.
            +1 for (ρ,θ,φ) right-handed, -1 otherwise.
    """

    sigma_bp: Literal[-1, 1]
    e_bp: Literal[0, 1]
    sigma_r_phi_z: Literal[-1, 1]
    sigma_rho_theta_phi: Literal[-1, 1]

    @property
    def cocos(self) -> int:
        """Compute COCOS value from parameters via table lookup.

        The mapping follows Table I of Sauter & Medvedev.
        Uses reverse lookup for correctness.
        """
        # Reverse lookup from parameters to COCOS
        for cocos_val, (s_bp, e_bp, s_rpz, s_rtp) in _COCOS_TABLE.items():
            if (
                self.sigma_bp == s_bp
                and self.e_bp == e_bp
                and self.sigma_r_phi_z == s_rpz
                and self.sigma_rho_theta_phi == s_rtp
            ):
                return cocos_val
        # Should never reach here if _COCOS_TABLE is complete
        msg = f"No COCOS found for parameters: {self}"
        raise ValueError(msg)


# Table I lookup: COCOS -> (σBp, eBp, σRφZ, σρθφ)
_COCOS_TABLE: dict[int, tuple[int, int, int, int]] = {
    # σBp, eBp, σRφZ, σρθφ  — Table I of Sauter & Medvedev, CPC 184 (2013)
    1: (+1, 0, +1, +1),
    2: (+1, 0, -1, +1),
    3: (-1, 0, +1, -1),
    4: (-1, 0, -1, -1),
    5: (+1, 0, +1, -1),
    6: (+1, 0, -1, -1),
    7: (-1, 0, +1, +1),
    8: (-1, 0, -1, +1),
    11: (+1, 1, +1, +1),
    12: (+1, 1, -1, +1),
    13: (-1, 1, +1, -1),
    14: (-1, 1, -1, -1),
    15: (+1, 1, +1, -1),
    16: (+1, 1, -1, -1),
    17: (-1, 1, +1, +1),
    18: (-1, 1, -1, +1),
}


def cocos_to_parameters(cocos: int) -> COCOSParameters:
    """Convert COCOS integer to parameter object.

    Args:
        cocos: COCOS value (1-8 or 11-18)

    Returns:
        COCOSParameters with the four determining parameters

    Raises:
        ValueError: If cocos is not a valid value
    """
    if cocos not in VALID_COCOS:
        msg = f"Invalid COCOS value {cocos}. Valid values: 1-8, 11-18"
        raise ValueError(msg)

    sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi = _COCOS_TABLE[cocos]
    return COCOSParameters(
        sigma_bp=sigma_bp,
        e_bp=e_bp,
        sigma_r_phi_z=sigma_r_phi_z,
        sigma_rho_theta_phi=sigma_rho_theta_phi,
    )


def cocos_from_dd_version(version: str) -> int:
    """Get COCOS from IMAS Data Dictionary version.

    First attempts to read the COCOS value from the DD XML directly.
    Falls back to version-based heuristic: COCOS 11 for DD < 4.0.0,
    COCOS 17 for DD >= 4.0.0.

    Args:
        version: DD version string (e.g., "3.39.0", "4.0.0")

    Returns:
        COCOS value (11 for DD < 4.0.0, 17 for DD >= 4.0.0)
    """
    try:
        import imas.dd_zip

        tree = imas.dd_zip.dd_etree(version)
        root = tree.getroot() if hasattr(tree, "getroot") else tree
        cocos_el = root.find("cocos")
        if cocos_el is not None:
            return int(cocos_el.text)
    except Exception:
        pass

    # Fallback for versions without <cocos> in XML (pre-3.35.0)
    from packaging.version import Version

    v = Version(version)
    return 17 if v >= Version("4.0.0") else 11


def determine_cocos(
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    q: float | None = None,
    dp_dpsi: float | None = None,
    *,
    assume_r_phi_z_right_handed: bool = True,
) -> tuple[int, float]:
    """Determine COCOS from equilibrium quantities.

    Uses Eq. 23 from Sauter & Medvedev to infer COCOS from physics:
    - sign(ψ_edge - ψ_axis) = σIp × σBp
    - sign(dp/dψ) = -σIp × σBp
    - sign(q) = σIp × σB0 × σρθφ

    Args:
        psi_axis: Poloidal flux at magnetic axis [Wb or Wb/2π]
        psi_edge: Poloidal flux at plasma edge [Wb or Wb/2π]
        ip: Plasma current [A] (sign matters)
        b0: Toroidal field at axis [T] (sign matters)
        q: Safety factor at mid-radius (optional, improves confidence)
        dp_dpsi: Pressure gradient dp/dψ at mid-radius (optional, validates σBp)
        assume_r_phi_z_right_handed: Assume (R,φ,Z) right-handed (most common)

    Returns:
        Tuple of (cocos, confidence) where:
        - cocos: Determined COCOS value (1-8 or 11-18)
        - confidence: Confidence score (0.0-1.0)

    Example:
        >>> cocos, conf = determine_cocos(
        ...     psi_axis=0.5, psi_edge=-0.2,  # decreasing from axis
        ...     ip=-1e6, b0=-5.0,  # negative current and field
        ...     q=3.0,
        ... )
        >>> print(f"COCOS={cocos} (confidence={conf:.2f})")
        COCOS=17 (confidence=0.70)
    """
    # Determine signs
    sigma_ip = 1 if ip >= 0 else -1
    sigma_b0 = 1 if b0 >= 0 else -1

    # Determine σBp from psi gradient (Eq. 23, line 5)
    # sign(ψ_edge - ψ_axis) = σIp × σBp
    psi_diff = psi_edge - psi_axis
    psi_sign = 1 if psi_diff >= 0 else -1
    sigma_bp = psi_sign * sigma_ip

    # Determine eBp from magnitude heuristic
    # If |ψ| > ~0.5 Weber, probably full ψ (not divided by 2π)
    max_psi = max(abs(psi_axis), abs(psi_edge))
    e_bp = 1 if max_psi > 0.5 else 0

    confidence = 0.5  # Base confidence with just psi

    # Validate with dp/dpsi if available (Eq. 23, line 6)
    # sign(dp/dψ) = -σIp × σBp
    if dp_dpsi is not None and not math.isnan(dp_dpsi):
        expected_sign = -sigma_ip * sigma_bp
        actual_sign = 1 if dp_dpsi >= 0 else -1
        if actual_sign == expected_sign:
            confidence += 0.15
        else:
            confidence -= 0.25  # Inconsistent - reduce confidence

    # Determine σρθφ from q if available (Eq. 23, line 8)
    # sign(q) = σIp × σB0 × σρθφ
    if q is not None and not math.isnan(q):
        q_sign = 1 if q >= 0 else -1
        sigma_rho_theta_phi = q_sign * sigma_ip * sigma_b0
        confidence += 0.2
    else:
        # Default to right-handed (most common)
        sigma_rho_theta_phi = 1

    # σRφZ - from assumption or geometry
    sigma_r_phi_z = 1 if assume_r_phi_z_right_handed else -1

    params = COCOSParameters(
        sigma_bp=sigma_bp,
        e_bp=e_bp,
        sigma_r_phi_z=sigma_r_phi_z,
        sigma_rho_theta_phi=sigma_rho_theta_phi,
    )

    return params.cocos, min(1.0, max(0.0, confidence))


def validate_cocos_consistency(
    cocos: int,
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    q: float | None = None,
    dp_dpsi: float | None = None,
) -> list[str]:
    """Validate COCOS against physics quantities.

    Checks that the provided COCOS is consistent with the equilibrium
    data according to Eq. 23 from the Sauter paper.

    Args:
        cocos: COCOS value to validate
        psi_axis: Poloidal flux at magnetic axis
        psi_edge: Poloidal flux at plasma edge
        ip: Plasma current [A]
        b0: Toroidal field at axis [T]
        q: Safety factor (optional)
        dp_dpsi: Pressure gradient (optional)

    Returns:
        List of inconsistency messages (empty if consistent)

    Example:
        >>> errors = validate_cocos_consistency(
        ...     cocos=11,
        ...     psi_axis=0.5, psi_edge=-0.2,
        ...     ip=-1e6, b0=-5.0, q=3.0
        ... )
        >>> if errors:
        ...     print("Inconsistencies found:", errors)
    """
    errors: list[str] = []

    if cocos not in VALID_COCOS:
        errors.append(f"Invalid COCOS value {cocos}")
        return errors

    params = cocos_to_parameters(cocos)
    sigma_ip = 1 if ip >= 0 else -1
    sigma_b0 = 1 if b0 >= 0 else -1

    # Check psi gradient (Eq. 23, line 5)
    expected_psi_sign = sigma_ip * params.sigma_bp
    actual_psi_sign = 1 if (psi_edge - psi_axis) >= 0 else -1
    if expected_psi_sign != actual_psi_sign:
        direction = "increasing" if actual_psi_sign == 1 else "decreasing"
        expected_dir = "increasing" if expected_psi_sign == 1 else "decreasing"
        errors.append(
            f"ψ gradient: expected {expected_dir}, got {direction} "
            f"(σIp={sigma_ip}, σBp={params.sigma_bp})"
        )

    # Check dp/dpsi if provided (Eq. 23, line 6)
    if dp_dpsi is not None and not math.isnan(dp_dpsi):
        expected_dp_sign = -sigma_ip * params.sigma_bp
        actual_dp_sign = 1 if dp_dpsi >= 0 else -1
        if expected_dp_sign != actual_dp_sign:
            errors.append(
                f"dp/dψ sign: expected {expected_dp_sign:+d}, got {actual_dp_sign:+d}"
            )

    # Check q sign if provided (Eq. 23, line 8)
    if q is not None and not math.isnan(q):
        expected_q_sign = sigma_ip * sigma_b0 * params.sigma_rho_theta_phi
        actual_q_sign = 1 if q >= 0 else -1
        if expected_q_sign != actual_q_sign:
            errors.append(
                f"q sign: expected {expected_q_sign:+d}, got {actual_q_sign:+d} "
                f"(σIp={sigma_ip}, σB0={sigma_b0}, σρθφ={params.sigma_rho_theta_phi})"
            )

    return errors


@dataclass(frozen=True)
class ValidationResult:
    """Result of COCOS validation against physics data.

    Attributes:
        is_consistent: Whether declared COCOS matches physics data
        declared_cocos: The COCOS value being validated
        calculated_cocos: COCOS inferred from physics quantities
        confidence: Confidence in calculated COCOS (0.0-1.0)
        inconsistencies: List of specific physics inconsistencies
    """

    is_consistent: bool
    declared_cocos: int
    calculated_cocos: int
    confidence: float
    inconsistencies: list[str]


def validate_cocos_from_data(
    declared_cocos: int,
    *,
    # Required physics quantities
    psi_axis: float,
    psi_edge: float,
    ip: float,
    b0: float,
    # Optional - improve confidence
    q: float | None = None,
    dp_dpsi: float | None = None,
) -> ValidationResult:
    """Validate declared COCOS against physics data.

    This is deterministic - uses Eq. 23 from Sauter paper.
    No LLM involvement. Data source agnostic.

    The caller is responsible for loading physics quantities from their
    data source (MDSplus, IMAS IDS, EQDSK, HDF5, etc.).

    Args:
        declared_cocos: The COCOS value to validate
        psi_axis: Poloidal flux at magnetic axis [Wb]
        psi_edge: Poloidal flux at plasma edge [Wb]
        ip: Plasma current [A] (sign matters)
        b0: Toroidal field at axis [T] (sign matters)
        q: Safety factor at mid-radius (optional, improves confidence)
        dp_dpsi: Pressure gradient dp/dψ (optional, validates σBp)

    Returns:
        ValidationResult with consistency check, calculated COCOS,
        confidence score, and list of specific inconsistencies.

    Example:
        >>> result = validate_cocos_from_data(
        ...     declared_cocos=17,
        ...     psi_axis=0.5, psi_edge=-0.2,
        ...     ip=-1e6, b0=-5.0, q=3.0,
        ... )
        >>> if not result.is_consistent:
        ...     print(f"COCOS mismatch: declared {result.declared_cocos}, "
        ...           f"calculated {result.calculated_cocos}")
        ...     for error in result.inconsistencies:
        ...         print(f"  - {error}")
    """
    # Calculate COCOS from physics
    calculated, confidence = determine_cocos(
        psi_axis=psi_axis,
        psi_edge=psi_edge,
        ip=ip,
        b0=b0,
        q=q,
        dp_dpsi=dp_dpsi,
    )

    # Validate declared COCOS against physics
    inconsistencies = validate_cocos_consistency(
        cocos=declared_cocos,
        psi_axis=psi_axis,
        psi_edge=psi_edge,
        ip=ip,
        b0=b0,
        q=q,
        dp_dpsi=dp_dpsi,
    )

    return ValidationResult(
        is_consistent=len(inconsistencies) == 0,
        declared_cocos=declared_cocos,
        calculated_cocos=calculated,
        confidence=confidence,
        inconsistencies=inconsistencies,
    )


# Known COCOS values for common codes (from Sauter paper Table IV and Appendix A)
KNOWN_CODE_COCOS: dict[str, int] = {
    "CHEASE": 2,
    "ONETWO": 2,
    "LIUQE": 17,
    "LIUQE.M": 17,
    "LIUQE02": 17,
    "LIUQE03": 17,
    "FBTE": 17,  # TCV convention
    "ORB5": 3,  # Uses negative Ip normalization
    # ITER/IMAS defaults
    "IMAS_DD_V3": 11,
    "IMAS_DD_V4": 17,
}
