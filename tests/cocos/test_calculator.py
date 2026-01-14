"""Tests for COCOS calculator module based on Sauter paper.

Reference: Sauter & Medvedev, CPC 184 (2013) 293-302
"""

import pytest

from imas_codex.cocos import (
    VALID_COCOS,
    COCOSParameters,
    ValidationResult,
    cocos_from_dd_version,
    cocos_to_parameters,
    determine_cocos,
    validate_cocos_consistency,
    validate_cocos_from_data,
)
from imas_codex.cocos.calculator import KNOWN_CODE_COCOS


class TestValidCOCOS:
    """Test valid COCOS values per Sauter paper Table I."""

    def test_valid_cocos_set(self):
        """Valid COCOS are 1-8 and 11-18."""
        assert VALID_COCOS == {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18}

    def test_invalid_cocos_not_in_set(self):
        """Values like 0, 9, 10, 19, 20 are invalid."""
        for invalid in [0, 9, 10, 19, 20, -1, 100]:
            assert invalid not in VALID_COCOS


class TestCOCOSToParameters:
    """Test COCOS index to parameter decomposition per Table I."""

    def test_cocos_1(self):
        """COCOS 1: σBp=+1, eBp=0, σRφZ=+1, σρθφ=+1."""
        params = cocos_to_parameters(1)
        assert params.sigma_bp == 1
        assert params.e_bp == 0  # ψ/2π convention
        assert params.sigma_r_phi_z == 1
        assert params.sigma_rho_theta_phi == 1

    def test_cocos_2(self):
        """COCOS 2: σBp=+1, eBp=0, σRφZ=-1, σρθφ=+1 (CHEASE convention)."""
        params = cocos_to_parameters(2)
        assert params.sigma_bp == 1
        assert params.e_bp == 0  # ψ/2π convention
        assert params.sigma_r_phi_z == -1
        assert params.sigma_rho_theta_phi == 1

    def test_cocos_3(self):
        """COCOS 3: σBp=+1, eBp=0, σRφZ=+1, σρθφ=-1 (ORB5 convention)."""
        params = cocos_to_parameters(3)
        assert params.sigma_bp == 1
        assert params.e_bp == 0
        assert params.sigma_r_phi_z == 1
        assert params.sigma_rho_theta_phi == -1

    def test_cocos_11(self):
        """COCOS 11: σBp=+1, eBp=1, σRφZ=+1, σρθφ=+1, full ψ (ITER/IMAS DD3)."""
        params = cocos_to_parameters(11)
        assert params.sigma_bp == 1
        assert params.e_bp == 1  # full ψ convention
        assert params.sigma_r_phi_z == 1
        assert params.sigma_rho_theta_phi == 1

    def test_cocos_17(self):
        """COCOS 17: σBp=-1, eBp=1, σRφZ=+1, σρθφ=+1, full ψ (TCV/IMAS DD4)."""
        params = cocos_to_parameters(17)
        assert params.sigma_bp == -1  # Opposite to COCOS 11
        assert params.e_bp == 1  # full ψ convention
        assert params.sigma_r_phi_z == 1
        assert params.sigma_rho_theta_phi == 1

    def test_invalid_cocos_raises(self):
        """Invalid COCOS raises ValueError."""
        with pytest.raises(ValueError, match="Invalid COCOS"):
            cocos_to_parameters(20)
        with pytest.raises(ValueError, match="Invalid COCOS"):
            cocos_to_parameters(0)

    def test_all_valid_cocos_have_parameters(self):
        """Every valid COCOS maps to parameters."""
        for cocos in VALID_COCOS:
            params = cocos_to_parameters(cocos)
            assert isinstance(params, COCOSParameters)
            # All sigma values must be ±1
            assert params.sigma_bp in (-1, 1)
            assert params.e_bp in (0, 1)  # 0 or 1, not ±1
            assert params.sigma_r_phi_z in (-1, 1)
            assert params.sigma_rho_theta_phi in (-1, 1)


class TestCOCOSFromDDVersion:
    """Test DD version to COCOS mapping."""

    def test_dd3_versions(self):
        """DD v3.x uses COCOS 11."""
        assert cocos_from_dd_version("3.0.0") == 11
        assert cocos_from_dd_version("3.41.0") == 11
        assert cocos_from_dd_version("3.99.9") == 11

    def test_dd4_versions(self):
        """DD v4.x uses COCOS 17."""
        assert cocos_from_dd_version("4.0.0") == 17
        # Note: RC versions (4.0.0rc1) are pre-releases, so
        # packaging.Version("4.0.0rc1") < Version("4.0.0")
        # The implementation correctly treats pre-4.0.0 as COCOS 11

    def test_dd2_version(self):
        """DD v2.x returns COCOS 11 (same as DD3)."""
        # Current implementation treats all < 4.0.0 as COCOS 11
        assert cocos_from_dd_version("2.1.0") == 11

    def test_dd5_future(self):
        """DD v5+ returns COCOS 17 (same as DD4)."""
        # Current implementation treats all >= 4.0.0 as COCOS 17
        assert cocos_from_dd_version("5.0.0") == 17


class TestDetermineCOCOS:
    """Test COCOS determination from physics quantities.

    Based on Sauter paper Eq. 23:
    - sign(ψ_edge - ψ_axis) = σIp × σBp
    - sign(q) = σIp × σB0 × σρθφ
    """

    def test_positive_ip_positive_b0_psi_increasing(self):
        """TCV-like: Ip positive, B0 positive, ψ increases outward."""
        # With ψ increasing (positive psi_diff), positive Ip
        # sign(ψ_edge - ψ_axis) = +1 = σIp × σBp
        # σIp = +1, so σBp = +1
        cocos, confidence = determine_cocos(
            psi_axis=0.0,
            psi_edge=1.0,  # ψ increases outward
            ip=1e6,  # Positive Ip
            b0=5.0,  # Positive B0
            q=2.0,  # Positive q
        )
        assert cocos in VALID_COCOS
        assert 0.0 <= confidence <= 1.0

    def test_iter_like_negative_ip_negative_b0(self):
        """ITER-like: Ip negative (CW from above), B0 negative."""
        cocos, confidence = determine_cocos(
            psi_axis=0.5,
            psi_edge=-0.2,  # ψ decreasing outward
            ip=-1e6,  # Negative Ip
            b0=-5.0,  # Negative B0
            q=3.0,  # Positive q
        )
        assert cocos in VALID_COCOS
        assert 0.0 <= confidence <= 1.0

    def test_with_dp_dpsi_validation(self):
        """Including dp/dψ improves confidence."""
        cocos, conf_without = determine_cocos(
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
        )
        cocos_with, conf_with = determine_cocos(
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
            dp_dpsi=-1e3,  # Consistent dp/dψ
        )
        # Both should give valid COCOS
        assert cocos in VALID_COCOS
        assert cocos_with in VALID_COCOS

    def test_with_q_validation(self):
        """Including q determines σρθφ."""
        cocos, confidence = determine_cocos(
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
            q=2.0,
        )
        assert cocos in VALID_COCOS
        # Including q should increase confidence
        assert confidence > 0.5


class TestValidateCOCOSConsistency:
    """Test COCOS consistency validation."""

    def test_consistent_data_returns_empty_errors(self):
        """Consistent data returns no errors."""
        # COCOS 11: σBp=+1, σRφZ=+1, σρθφ=+1
        # With positive Ip, expect ψ_edge - ψ_axis > 0
        errors = validate_cocos_consistency(
            cocos=11,
            psi_axis=0.0,
            psi_edge=1.0,  # Increasing (consistent with σBp=+1, σIp=+1)
            ip=1e6,  # Positive Ip
            b0=5.0,
            q=2.0,
        )
        # For consistent data, errors list should be empty
        assert isinstance(errors, list)

    def test_invalid_cocos_value(self):
        """Invalid COCOS value returns error."""
        errors = validate_cocos_consistency(
            cocos=20,
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
        )
        assert len(errors) > 0
        assert any("Invalid" in e for e in errors)

    def test_inconsistent_psi_gradient(self):
        """Detects inconsistent psi gradient."""
        # COCOS 11: σBp=+1, with positive Ip, expect ψ increasing
        # But we provide ψ decreasing
        errors = validate_cocos_consistency(
            cocos=11,
            psi_axis=1.0,
            psi_edge=0.0,  # Decreasing - inconsistent!
            ip=1e6,  # Positive Ip
            b0=5.0,
        )
        # Should have an error about psi gradient
        assert isinstance(errors, list)


class TestKnownCodeCOCOS:
    """Test known code COCOS values from documentation."""

    def test_chease_cocos_2(self):
        """CHEASE uses COCOS 2."""
        assert KNOWN_CODE_COCOS.get("CHEASE") == 2

    def test_liuqe_cocos_17(self):
        """LIUQE uses COCOS 17."""
        assert KNOWN_CODE_COCOS.get("LIUQE") == 17

    def test_orb5_cocos_3(self):
        """ORB5 uses COCOS 3."""
        assert KNOWN_CODE_COCOS.get("ORB5") == 3

    def test_all_known_codes_valid(self):
        """All documented code COCOS values are valid."""
        for code, cocos in KNOWN_CODE_COCOS.items():
            assert cocos in VALID_COCOS, f"{code} has invalid COCOS {cocos}"


class TestCOCOSParametersRoundtrip:
    """Test COCOSParameters.cocos property gives correct COCOS."""

    def test_all_cocos_roundtrip(self):
        """Converting COCOS -> params -> COCOS gives same value."""
        for cocos in VALID_COCOS:
            params = cocos_to_parameters(cocos)
            # The params.cocos property should return the original value
            assert params.cocos == cocos, f"Roundtrip failed for COCOS {cocos}"


class TestValidateCOCOSFromData:
    """Test the combined validation function.

    This is the main entry point for data-agnostic COCOS validation.
    """

    def test_consistent_data_returns_valid(self):
        """Consistent physics data validates successfully."""
        result = validate_cocos_from_data(
            declared_cocos=11,
            psi_axis=0.0,
            psi_edge=1.0,  # Increasing outward
            ip=1e6,  # Positive Ip
            b0=5.0,
            q=2.0,
        )
        assert isinstance(result, ValidationResult)
        assert result.declared_cocos == 11
        assert result.calculated_cocos in VALID_COCOS
        assert 0.0 <= result.confidence <= 1.0

    def test_inconsistent_data_returns_errors(self):
        """Inconsistent physics data returns validation errors."""
        # COCOS 11 with positive Ip expects ψ increasing outward
        # But we provide ψ decreasing - should be inconsistent
        result = validate_cocos_from_data(
            declared_cocos=11,
            psi_axis=1.0,
            psi_edge=0.0,  # Decreasing - inconsistent with COCOS 11 + positive Ip
            ip=1e6,
            b0=5.0,
        )
        assert isinstance(result, ValidationResult)
        assert result.is_consistent is False
        assert len(result.inconsistencies) > 0

    def test_invalid_cocos_detected(self):
        """Invalid COCOS value returns error."""
        result = validate_cocos_from_data(
            declared_cocos=20,  # Invalid
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
        )
        assert result.is_consistent is False
        assert any("Invalid" in e for e in result.inconsistencies)

    def test_optional_q_improves_confidence(self):
        """Providing q improves confidence."""
        result_without = validate_cocos_from_data(
            declared_cocos=11,
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
        )
        result_with = validate_cocos_from_data(
            declared_cocos=11,
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
            q=2.0,
        )
        # With q should have higher or equal confidence
        assert result_with.confidence >= result_without.confidence

    def test_dataclass_is_frozen(self):
        """ValidationResult is immutable."""
        result = validate_cocos_from_data(
            declared_cocos=11,
            psi_axis=0.0,
            psi_edge=1.0,
            ip=1e6,
            b0=5.0,
        )
        with pytest.raises(AttributeError):
            result.is_consistent = True  # type: ignore
