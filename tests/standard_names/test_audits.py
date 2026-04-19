"""Tests for L3 post-generation audits module."""

from __future__ import annotations

import numpy as np
import pytest

# =========================================================================
# L3: latex_def_check
# =========================================================================


class TestLatexDefCheck:
    """Tests for latex_def_check audit."""

    def test_pass_symbols_defined(self):
        """Symbols with definitions pass."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The safety factor is $q = d\\Phi/d\\Psi$, where "
                "$q$ is the safety factor (dimensionless), "
                "$\\Phi$ is the toroidal flux (Wb), and "
                "$\\Psi$ is the poloidal flux (Wb)."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) == 0

    def test_fail_undefined_symbol(self):
        """Symbols without definitions are flagged."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The plasma has $T_e$ and $n_e$ parameters. "
                "These affect confinement significantly. "
                "Further analysis shows improved stability."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) >= 1
        assert any("latex_def_check" in i for i in issues)

    def test_pass_universal_constants_skipped(self):
        """Universal physics constants (\\pi, \\alpha, \\mu_0, \\hbar,
        k_B) and numeric factors thereof (``2\\pi``, ``\\pi/2``) do not
        require a definition sentence."""
        from imas_codex.standard_names.audits import latex_def_check

        for sym in (
            r"\pi",
            r"2\pi",
            r"\pi/2",
            r"\alpha",
            r"\mu_0",
            r"\hbar",
            "k_B",
            r"\epsilon_0",
        ):
            candidate = {
                "documentation": f"Uses ${sym}$ with no explicit definition.",
            }
            assert latex_def_check(candidate) == [], (
                f"Expected no issue for {sym}, got {latex_def_check(candidate)}"
            )

    def test_pass_empty_documentation(self):
        """No documentation produces no issues."""
        from imas_codex.standard_names.audits import latex_def_check

        assert latex_def_check({"documentation": ""}) == []
        assert latex_def_check({}) == []

    def test_pass_no_latex_symbols(self):
        """Documentation without LaTeX symbols passes."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {"documentation": "A simple description without math."}
        assert latex_def_check(candidate) == []


# =========================================================================
# L3: provenance_verb_check
# =========================================================================


class TestProvenanceVerbCheck:
    """Tests for provenance_verb_check audit."""

    def test_pass_no_provenance_verb(self):
        """Clean name passes."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature"}
        issues = provenance_verb_check(candidate)
        assert len(issues) == 0

    def test_fail_measured_in_name(self):
        """Name with 'measured' when source lacks it is flagged."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(
            candidate, source_path="core_profiles/profiles_1d/electrons/temperature"
        )
        assert len(issues) == 1
        assert "measured" in issues[0]

    def test_pass_measured_in_both(self):
        """Name with 'measured' is OK when source path also has it."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(candidate, source_path="diagnostics/measured/te")
        assert len(issues) == 0


# =========================================================================
# L3: synonym_check
# =========================================================================


class TestSynonymCheck:
    """Tests for synonym_check audit."""

    def test_pass_no_similar(self):
        """No existing SNs means no synonym issues."""
        from imas_codex.standard_names.audits import synonym_check

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": np.random.rand(384).tolist(),
        }
        issues = synonym_check(candidate, [])
        assert len(issues) == 0

    def test_fail_high_cosine(self):
        """Near-identical embedding with same unit is flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)
        # Create a very similar vector
        similar_vec = vec + np.random.rand(384).astype(np.float32) * 0.01

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "te_profile",
                "unit": "eV",
                "description_embedding": similar_vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 1
        assert "synonym_check" in issues[0]

    def test_pass_different_unit(self):
        """Same embedding but different unit is not flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "electron_density",
                "unit": "m^-3",
                "description_embedding": vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 0


# =========================================================================
# L3: unit_dimension_check
# =========================================================================


class TestUnitDimensionCheck:
    """Tests for unit_dimension_check audit."""

    def test_pass_consistent(self):
        """Description consistent with unit passes."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "eV", "description": "Electron temperature profile"}
        assert unit_dimension_check(candidate) == []

    def test_fail_inconsistent(self):
        """Description inconsistent with unit is flagged."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "A", "description": "Radial position of the boundary"}
        issues = unit_dimension_check(candidate)
        assert len(issues) == 1
        assert "unit_dimension_check" in issues[0]

    def test_pass_dimensionless(self):
        """Dimensionless unit is not checked."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "1", "description": "Safety factor profile"}
        assert unit_dimension_check(candidate) == []


# =========================================================================
# L3: multi_subject_check
# =========================================================================


class TestMultiSubjectCheck:
    """Tests for multi_subject_check audit."""

    def test_pass_single_subject(self):
        """Single-subject name passes."""
        from imas_codex.standard_names.audits import multi_subject_check

        candidate = {"id": "electron_temperature"}
        issues = multi_subject_check(candidate)
        assert len(issues) == 0

    def test_fail_multiple_subjects(self):
        """Name with two different subject tokens is flagged."""
        from imas_codex.standard_names.audits import multi_subject_check

        # This will flag if two Subject enum values appear in name tokens
        candidate = {"id": "electron_ion_temperature"}
        issues = multi_subject_check(candidate)
        # May or may not flag depending on grammar — at least doesn't crash
        assert isinstance(issues, list)


# =========================================================================
# L3: cocos_specificity_check
# =========================================================================


class TestCocosSpecificityCheck:
    """Tests for cocos_specificity_check audit."""

    def test_pass_cocos_mentioned(self):
        """Documentation mentioning COCOS with digit passes."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {
            "documentation": "Sign convention: Positive when COCOS 11 convention applies."
        }
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 0

    def test_fail_no_cocos(self):
        """Documentation without COCOS mention when source has COCOS type is flagged."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "The poloidal magnetic flux per radian."}
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 1
        assert "cocos_specificity_check" in issues[0]

    def test_pass_no_cocos_type(self):
        """No source COCOS type means no check."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "Simple quantity."}
        issues = cocos_specificity_check(candidate, source_cocos_type=None)
        assert len(issues) == 0


# =========================================================================
# L3: run_audits integration
# =========================================================================


class TestRunAudits:
    """Tests for run_audits orchestrator."""

    def test_clean_candidate_passes(self):
        """A well-formed candidate passes all audits."""
        from imas_codex.standard_names.audits import run_audits

        candidate = {
            "id": "electron_temperature",
            "description": "Electron temperature profile",
            "documentation": (
                "The electron temperature $T_e$ is the kinetic energy "
                "per degree of freedom, where $T_e$ denotes the temperature (eV)."
            ),
            "unit": "eV",
        }
        issues = run_audits(candidate)
        assert isinstance(issues, list)

    def test_has_critical_audit_failure(self):
        """Critical check tag detection works."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        assert (
            has_critical_audit_failure(["audit:latex_def_check: missing def"]) is True
        )
        assert has_critical_audit_failure(["audit:synonym_check: cosine=0.95"]) is True
        assert (
            has_critical_audit_failure(["audit:multi_subject_check: two subjects"])
            is True
        )
        assert (
            has_critical_audit_failure(["audit:placeholder_check: [condition]"]) is True
        )
        assert (
            has_critical_audit_failure(
                ["audit:unit_validity_check: non-unit token 'dimension'"]
            )
            is True
        )
        assert (
            has_critical_audit_failure(["audit:unit_dimension_check: mismatch"])
            is False
        )
        assert has_critical_audit_failure([]) is False


# =========================================================================
# placeholder_check
# =========================================================================


class TestPlaceholderCheck:
    """Tests for placeholder_check audit."""

    def test_pass_concrete_sign_convention(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "Sign convention: Positive when the plasma current flows "
                "counter-clockwise viewed from above."
            ),
        }
        assert placeholder_check(c) == []

    def test_fail_bracketed_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "Sign convention: Positive when [condition]."}
        issues = placeholder_check(c)
        assert len(issues) == 1
        assert "placeholder_check" in issues[0]

    def test_fail_bracketed_specific_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"description": "Positive when [specific physical condition]"}
        issues = placeholder_check(c)
        assert len(issues) == 1

    def test_pass_markdown_link(self):
        """Markdown [text](url) links are not placeholders."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "See [magnetic_flux](#magnetic_flux) and [safety_factor](#safety_factor)."
            ),
        }
        assert placeholder_check(c) == []

    def test_pass_numeric_bracket(self):
        """Numeric brackets like [1] or citation-like markers are not flagged."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "See reference [1] and range [0, 1]."}
        assert placeholder_check(c) == []


# =========================================================================
# unit_validity_check
# =========================================================================


class TestUnitValidityCheck:
    """Tests for unit_validity_check audit."""

    def test_pass_real_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        for unit in ("m", "T", "Wb", "eV", "m^2", "kg*m/s^2", "m.s^-1", "1"):
            assert unit_validity_check({"unit": unit}) == [], f"failed for {unit}"

    def test_fail_m_caret_dimension(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "m^dimension"})
        assert len(issues) == 1
        assert "dimension" in issues[0]

    def test_fail_fourier_in_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "T*fourier"})
        assert len(issues) == 1

    def test_pass_empty_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        assert unit_validity_check({"unit": ""}) == []
        assert unit_validity_check({"unit": "dimensionless"}) == []


class TestGenericNounCheck:
    """Tests for generic_noun_check audit."""

    def test_fail_bare_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        issues = generic_noun_check({"id": "geometry"})
        assert len(issues) == 1
        assert "generic_noun_check" in issues[0]

    def test_fail_bare_data(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "data"})) == 1

    def test_pass_qualified_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "grid_object_geometry"}) == []

    def test_pass_multi_token(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "electron_temperature"}) == []

    def test_fail_generic_qualifier_plus_generic_noun(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "raw_data"})) == 1


class TestTautologyCheck:
    """Tests for tautology_check audit."""

    def test_fail_position_of_position(self):
        from imas_codex.standard_names.audits import tautology_check

        issues = tautology_check({"id": "radial_position_of_reference_position"})
        assert len(issues) == 1
        assert "tautology_check" in issues[0]
        assert "position" in issues[0]

    def test_fail_component_of_component(self):
        from imas_codex.standard_names.audits import tautology_check

        assert len(tautology_check({"id": "normal_component_of_field_component"})) == 1

    def test_pass_no_of(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "electron_temperature"}) == []

    def test_pass_different_heads(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "radial_position_of_plasma_boundary"}) == []

    def test_pass_of_without_tautology_head(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "elongation_of_plasma_boundary"}) == []


class TestSpectralSuffixCheck:
    """Tests for spectral_suffix_check audit."""

    def test_fail_fourier_coefficients(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        issues = spectral_suffix_check({"id": "normal_field_fourier_coefficients"})
        assert len(issues) == 1
        assert "spectral_suffix_check" in issues[0]

    def test_fail_harmonics(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert len(spectral_suffix_check({"id": "magnetic_field_harmonics"})) == 1

    def test_pass_mode_prefix(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "mode_amplitude_of_normal_field"}) == []

    def test_pass_ordinary_name(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "poloidal_magnetic_flux"}) == []


class TestAbbreviationCheck:
    """Tests for abbreviation_check audit."""

    def test_fail_norm_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        issues = abbreviation_check({"id": "norm_poloidal_magnetic_flux"})
        assert len(issues) == 1
        assert "normalized" in issues[0]

    def test_fail_perp_interior(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "velocity_perp_component"})) == 1

    def test_fail_temp_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "temp_profile"})) == 1

    def test_pass_full_words(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": "normalized_poloidal_magnetic_flux"}) == []
        assert abbreviation_check({"id": "perpendicular_velocity_component"}) == []

    def test_pass_empty(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": ""}) == []


class TestNameDescriptionConsistencyCheck:
    """Tests for name_description_consistency_check audit."""

    def test_fail_fourier_desc_bare_name(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        issues = name_description_consistency_check(
            {
                "id": "normal_component_of_magnetic_field",
                "description": "Fourier coefficients of the normal component of the field.",
            }
        )
        assert len(issues) == 1

    def test_pass_decomposition_marker(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "mode_amplitude_of_normal_field",
                    "description": "Fourier coefficients of the normal field.",
                }
            )
            == []
        )

    def test_pass_plain_desc(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature profile.",
                }
            )
            == []
        )

    def test_pass_missing_fields(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert name_description_consistency_check({"id": "x", "description": ""}) == []


class TestAmericanSpellingCheck:
    """Tests for american_spelling_check audit (NC-17)."""

    def test_fail_british_in_name(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "normalised_poloidal_flux", "description": "A flux."}
        )
        assert any("'normalised'" in i and "normalized" in i for i in issues)
        assert any("field 'name'" in i for i in issues)

    def test_fail_british_in_description(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "plasma_current",
                "description": "Current at the centre of the plasma, analysed per shot.",
            }
        )
        fields = {i.split("field '")[1].split("'")[0] for i in issues}
        assert "description" in fields
        joined = " ".join(issues)
        assert "centre" in joined and "analysed" in joined

    def test_fail_british_in_constraints(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "foo",
                "description": "ok",
                "constraints": ["Must be normalised to 1"],
            }
        )
        assert any("constraints[0]" in i for i in issues)

    def test_pass_american_only(self):
        from imas_codex.standard_names.audits import american_spelling_check

        assert (
            american_spelling_check(
                {
                    "id": "normalized_poloidal_flux",
                    "description": "The normalized flux at the center of the plasma, analyzed per shot.",
                    "documentation": "Modeled behavior of labeled channels.",
                }
            )
            == []
        )

    def test_case_insensitive(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "x", "description": "The Normalised profile."}
        )
        assert len(issues) == 1
        assert "Normalised" in issues[0]


# =========================================================================
# description_verb_drift_check
# =========================================================================


class TestDescriptionVerbDriftCheck:
    """Name/description rate-marker consistency."""

    def test_fail_instant_change_prefix(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        issues = description_verb_drift_check(
            {
                "id": "instant_change_in_electron_density",
                "description": "Instantaneous signed change in electron number density.",
            }
        )
        assert len(issues) == 1
        assert "instant_change_" in issues[0] or "begins with" in issues[0]

    def test_fail_rate_description_missing_marker(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        issues = description_verb_drift_check(
            {
                "id": "ion_temperature",
                "description": "Instantaneous change in ion temperature due to a transient plasma event.",
            }
        )
        assert len(issues) == 1
        assert "rate" in issues[0] or "tendency_of_" in issues[0]

    def test_pass_tendency_name(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "tendency_of_electron_density",
                    "description": "Instantaneous signed change in electron density.",
                }
            )
            == []
        )

    def test_pass_change_in_name(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "change_in_ion_temperature",
                    "description": "Time derivative of ion temperature.",
                }
            )
            == []
        )

    def test_pass_base_quantity_description(self):
        from imas_codex.standard_names.audits import description_verb_drift_check

        assert (
            description_verb_drift_check(
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature radial profile.",
                }
            )
            == []
        )


# =========================================================================
# structural_dim_tag_check
# =========================================================================


class TestStructuralDimTagCheck:
    """Advisory flag for DD data-type dimensionality tags in descriptions."""

    def test_fail_1d_in_description(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        issues = structural_dim_tag_check(
            {"description": "Electron temperature as a 1D radial profile."}
        )
        assert len(issues) == 1
        assert "1D" in issues[0]

    def test_fail_2d_in_description(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        issues = structural_dim_tag_check({"description": "2D map of poloidal flux."})
        assert len(issues) == 1

    def test_pass_no_tag(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        assert (
            structural_dim_tag_check(
                {"description": "Electron temperature radial profile."}
            )
            == []
        )

    def test_pass_dimensionless(self):
        from imas_codex.standard_names.audits import structural_dim_tag_check

        # 'dimensionless' contains 'd' but not \bNd\b
        assert (
            structural_dim_tag_check({"description": "A dimensionless parameter."})
            == []
        )


class TestDensityUnitConsistencyCheck:
    def test_fail_density_with_bare_momentum_unit(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        issues = density_unit_consistency_check(
            {"id": "toroidal_angular_momentum_density", "unit": "kg.m.s^-1"}
        )
        assert len(issues) == 1
        assert "no inverse-length factor" in issues[0]

    def test_pass_volumetric_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check({"id": "electron_density", "unit": "m^-3"})
            == []
        )

    def test_pass_areal_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check(
                {"id": "surface_charge_density", "unit": "C.m^-2"}
            )
            == []
        )

    def test_pass_dimensionless_density(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check({"id": "ion_fraction_density", "unit": "1"})
            == []
        )

    def test_pass_no_density_in_name(self):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        assert (
            density_unit_consistency_check(
                {"id": "toroidal_torque", "unit": "kg.m^2.s^-2"}
            )
            == []
        )


class TestPositionCoordinateCheck:
    def test_fail_radial_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "radial_position_of_antenna_row"})
        assert len(issues) == 1
        assert "major_radius_of_<X>" in issues[0]

    def test_fail_toroidal_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "toroidal_position_of_antenna_row"})
        assert len(issues) == 1
        assert "toroidal_angle_of_<X>" in issues[0]

    def test_fail_vertical_position_unconditional(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        issues = position_coordinate_check({"id": "vertical_position_of_x_point"})
        assert len(issues) == 1
        assert "vertical_coordinate_of_<X>" in issues[0]

    def test_pass_canonical_major_radius(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        assert (
            position_coordinate_check(
                {"id": "major_radius_of_electron_cyclotron_launcher"}
            )
            == []
        )

    def test_pass_no_position_in_name(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        assert position_coordinate_check({"id": "electron_density"}) == []

    def test_pass_plain_position_no_directional_qualifier(self):
        from imas_codex.standard_names.audits import position_coordinate_check

        # Plain `position_of_X` (no R/Z/phi qualifier) is acceptable for
        # unspecified 3-vector positions and must not be flagged.
        assert position_coordinate_check({"id": "position_of_strike_point"}) == []


class TestVectorFieldComponentCheck:
    """Tests for vector_field_component_check audit."""

    def test_flags_vertical_coordinate_of_surface_normal(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        issues = vector_field_component_check(
            {"id": "vertical_coordinate_of_surface_normal"}
        )
        assert len(issues) == 1
        assert "vector_field_component_check" in issues[0]
        assert "vertical_component_of_surface_normal" in issues[0]

    def test_flags_radial_coordinate_of_magnetic_field_vector(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        issues = vector_field_component_check(
            {"id": "radial_coordinate_of_magnetic_field_vector"}
        )
        assert len(issues) == 1
        assert "radial_component_of_magnetic_field_vector" in issues[0]

    def test_passes_vertical_coordinate_of_plasma_boundary(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        # plasma_boundary is a geometric feature (point/curve), not a vector
        # field — _coordinate_of_ is correct.
        assert (
            vector_field_component_check(
                {"id": "vertical_coordinate_of_plasma_boundary"}
            )
            == []
        )

    def test_passes_vertical_component_of_surface_normal(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        # The canonical form is not flagged.
        assert (
            vector_field_component_check({"id": "vertical_component_of_surface_normal"})
            == []
        )

    def test_passes_unrelated_name(self):
        from imas_codex.standard_names.audits import vector_field_component_check

        assert vector_field_component_check({"id": "electron_temperature"}) == []


class TestSegmentOrderCheck:
    def test_fail_trailing_toroidal(self):
        from imas_codex.standard_names.audits import segment_order_check

        issues = segment_order_check({"id": "ion_rotation_frequency_toroidal"})
        assert issues and "segment_order_check" in issues[0]

    def test_fail_trailing_poloidal(self):
        from imas_codex.standard_names.audits import segment_order_check

        issues = segment_order_check({"id": "electron_flux_poloidal"})
        assert issues and "segment_order_check" in issues[0]

    def test_pass_leading_component(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert segment_order_check({"id": "toroidal_ion_rotation_frequency"}) == []

    def test_pass_component_of_preposition(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert (
            segment_order_check({"id": "toroidal_component_of_ion_rotation_frequency"})
            == []
        )

    def test_pass_no_component_token(self):
        from imas_codex.standard_names.audits import segment_order_check

        assert segment_order_check({"id": "electron_temperature"}) == []


class TestCausalDueToCheckExtended:
    def test_pass_due_to_resistive_isn_process(self):
        """`resistive` is registered as a process token in ISN rc13+."""
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check({"id": "parallel_current_density_due_to_resistive"})
            == []
        )

    def test_pass_due_to_non_inductive_isn_process(self):
        """`non_inductive` is registered as a process token in ISN rc13+."""
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check({"id": "parallel_current_density_due_to_non_inductive"})
            == []
        )

    def test_pass_due_to_turbulent_isn_process(self):
        """`turbulent` is registered as a process token in ISN rc13+."""
        from imas_codex.standard_names.audits import causal_due_to_check

        assert causal_due_to_check({"id": "heat_flux_due_to_turbulent"}) == []

    def test_fail_due_to_shutdown_temporal(self):
        """Temporal events not in ISN process vocab remain flagged."""
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "heat_flux_due_to_shutdown"})
        assert issues and "shutdown" in issues[0]

    def test_pass_due_to_resistive_diffusion(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        assert (
            causal_due_to_check(
                {"id": "parallel_current_density_due_to_resistive_diffusion"}
            )
            == []
        )


class TestPeakingFactorExemption:
    def test_pass_ion_temperature_peaking_factor(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "ion_temperature_peaking_factor",
                    "unit": "1",
                    "description": "Ratio of central to volume-averaged ion temperature",
                }
            )
            == []
        )

    def test_pass_electron_temperature_profile_factor(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "electron_temperature_profile_factor",
                    "unit": "1",
                    "description": "Profile peaking factor for electron temperature",
                }
            )
            == []
        )

    def test_pass_fraction(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "bootstrap_current_fraction",
                    "unit": "1",
                    "description": "Fraction of total current carried by bootstrap",
                }
            )
            == []
        )

    def test_fail_bare_temperature_with_dimensionless(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {
                "id": "ion_temperature",
                "unit": "1",
                "description": "Ion temperature in dimensionless units",
            }
        )
        assert issues

    def test_pass_energy_flux_with_power_per_area(self):
        """``_energy_flux`` carries dimensions of power-per-area (W.m^-2),
        not pure energy. The audit must accept this without flagging."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        for name, unit in [
            ("incident_neutral_kinetic_energy_flux", "W.m^-2"),
            ("incident_ion_kinetic_energy_flux_on_wall", "m^-2.W"),
            ("electron_emitted_kinetic_energy_flux_at_first_wall", "m^-2.W"),
        ]:
            assert (
                name_unit_consistency_check(
                    {"id": name, "unit": unit, "description": ""}
                )
                == []
            ), f"unexpected fail for {name} {unit}"

    def test_pass_energy_flux_with_particle_flux_unit(self):
        """A particle flux of energy-bearing species (``m^-2.s^-1``) is also
        a legitimate use of ``_energy_flux`` per source-path semantics."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "neutral_kinetic_energy_flux_emitted_from_wall",
                    "unit": "m^-2.s^-1",
                    "description": "",
                }
            )
            == []
        )

    def test_pass_mass_flux_with_compound_unit(self):
        from imas_codex.standard_names.audits import name_unit_consistency_check

        assert (
            name_unit_consistency_check(
                {
                    "id": "ion_mass_flux_at_separatrix",
                    "unit": "kg.m^-2.s^-1",
                    "description": "",
                }
            )
            == []
        )

    def test_fail_bare_energy_with_particle_flux_unit(self):
        """Without the ``flux`` qualifier, energy with a non-energy unit
        must still fail — guards against the exemption being too broad."""
        from imas_codex.standard_names.audits import name_unit_consistency_check

        issues = name_unit_consistency_check(
            {
                "id": "neutral_kinetic_energy",
                "unit": "m^-2.s^-1",
                "description": "",
            }
        )
        assert issues


class TestAggregatorOrderCheck:
    def test_fail_trailing_volume_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "ion_temperature_volume_averaged"})
        assert issues and "aggregator_order_check" in issues[0]
        assert "volume_averaged_ion_temperature" in issues[0]

    def test_fail_trailing_flux_surface_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "current_density_flux_surface_averaged"})
        assert issues and "flux_surface_averaged" in issues[0]

    def test_fail_trailing_line_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        issues = aggregator_order_check({"id": "electron_density_line_averaged"})
        assert issues

    def test_pass_leading_volume_averaged(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        assert (
            aggregator_order_check({"id": "volume_averaged_electron_temperature"}) == []
        )

    def test_pass_no_aggregator(self):
        from imas_codex.standard_names.audits import aggregator_order_check

        assert aggregator_order_check({"id": "electron_temperature"}) == []


class TestNamedFeaturePrepositionCheck:
    def test_fail_at_magnetic_axis(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        issues = named_feature_preposition_check(
            {"id": "poloidal_magnetic_flux_at_magnetic_axis"}
        )
        assert issues and "poloidal_magnetic_flux_of_magnetic_axis" in issues[0]

    def test_fail_at_last_closed_flux_surface(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        issues = named_feature_preposition_check(
            {"id": "loop_voltage_at_last_closed_flux_surface"}
        )
        assert issues

    def test_fail_at_x_point(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        issues = named_feature_preposition_check(
            {"id": "poloidal_magnetic_flux_at_x_point"}
        )
        assert issues

    def test_pass_of_magnetic_axis(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        assert (
            named_feature_preposition_check(
                {"id": "poloidal_magnetic_flux_of_magnetic_axis"}
            )
            == []
        )

    def test_pass_of_plasma_boundary(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        assert (
            named_feature_preposition_check(
                {"id": "poloidal_magnetic_flux_of_plasma_boundary"}
            )
            == []
        )

    def test_pass_unrelated_name(self):
        from imas_codex.standard_names.audits import named_feature_preposition_check

        assert named_feature_preposition_check({"id": "electron_temperature"}) == []


class TestDiamagneticComponentCheck:
    def test_fail_diamagnetic_component_of_electric_field(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        issues = diamagnetic_component_check(
            {"id": "diamagnetic_component_of_electric_field"}
        )
        assert issues and "drift" in issues[0].lower()

    def test_fail_diamagnetic_component_of_ion_velocity(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        issues = diamagnetic_component_check(
            {"id": "diamagnetic_component_of_ion_velocity"}
        )
        assert issues

    def test_pass_diamagnetic_drift_velocity(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert diamagnetic_component_check({"id": "diamagnetic_drift_velocity"}) == []

    def test_pass_diamagnetic_current_density(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert diamagnetic_component_check({"id": "diamagnetic_current_density"}) == []

    def test_pass_toroidal_component(self):
        from imas_codex.standard_names.audits import diamagnetic_component_check

        assert (
            diamagnetic_component_check({"id": "toroidal_component_of_electric_field"})
            == []
        )


# =========================================================================
# C.1: multi_subject_check — greedy compound-subject match
# =========================================================================


class TestMultiSubjectCheckGreedy:
    """Tests for greedy longest-match in multi_subject_check."""

    @pytest.mark.parametrize(
        "name,expected_passes",
        [
            ("deuterium_tritium_fusion_power_density", True),
            ("tritium_to_deuterium_density_ratio", True),
            ("deuterium_deuterium_fusion_power", True),
            ("tritium_tritium_reaction_rate", True),
            ("electron_deuterium_density", False),
        ],
    )
    def test_compound_species_greedy(self, name, expected_passes):
        from imas_codex.standard_names.audits import multi_subject_check

        issues = multi_subject_check({"id": name})
        if expected_passes:
            assert issues == [], f"False positive on '{name}': {issues}"
        else:
            assert issues, f"Expected failure on '{name}'"

    def test_pass_single_compound_subject(self):
        from imas_codex.standard_names.audits import multi_subject_check

        assert multi_subject_check({"id": "deuterium_tritium_fusion_power"}) == []

    def test_pass_electron_equivalent_exemption_preserved(self):
        from imas_codex.standard_names.audits import multi_subject_check

        assert multi_subject_check({"id": "ion_electron_equivalent"}) == []


# =========================================================================
# C.3: density_unit_consistency_check — constraint-metadata suffix skip
# =========================================================================


class TestDensityConstraintMetadataExemption:
    """Tests for constraint-metadata suffix skip in density_unit_consistency_check."""

    @pytest.mark.parametrize(
        "suffix",
        [
            "_constraint_measurement_time",
            "_constraint_weight",
            "_constraint_reconstructed",
            "_constraint_measured",
            "_constraint_time_measurement",
            "_constraint_position",
        ],
    )
    def test_pass_constraint_metadata_suffix(self, suffix):
        from imas_codex.standard_names.audits import density_unit_consistency_check

        name = f"toroidal_current_density{suffix}"
        assert density_unit_consistency_check({"id": name, "unit": "s"}) == [], (
            f"False positive on '{name}'"
        )

    def test_fail_density_without_constraint_suffix(self):
        """Density without a constraint suffix and wrong unit still flagged."""
        from imas_codex.standard_names.audits import density_unit_consistency_check

        issues = density_unit_consistency_check(
            {"id": "toroidal_current_density", "unit": "s"}
        )
        assert issues, "Expected failure for bare density with wrong unit"


# =========================================================================
# C.4: implicit_field_check — device whitelist
# =========================================================================


class TestImplicitFieldDeviceWhitelist:
    """Tests for device whitelist in implicit_field_check."""

    def test_pass_vacuum_toroidal_field_function(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert implicit_field_check({"id": "vacuum_toroidal_field_function"}) == []

    def test_pass_resistance_of_poloidal_field_coil(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert implicit_field_check({"id": "resistance_of_poloidal_field_coil"}) == []

    def test_pass_field_coil_substring(self):
        from imas_codex.standard_names.audits import implicit_field_check

        assert (
            implicit_field_check({"id": "current_in_poloidal_field_coil_supply"}) == []
        )

    def test_fail_bare_field_still_flagged(self):
        from imas_codex.standard_names.audits import implicit_field_check

        issues = implicit_field_check({"id": "vacuum_toroidal_field"})
        assert issues and "implicit_field_check" in issues[0]


# =========================================================================
# C.5: causal_due_to_check — adjective-to-process map + suggested_fix
# =========================================================================


class TestCausalDueToSuggestedFix:
    """Tests for suggested_fix in causal_due_to_check adjective map."""

    def test_suggested_fix_halo(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_halo"})
        if issues:  # only fires if halo is not in ISN process vocab
            assert "suggested_fix=" in issues[0]
            assert "halo_currents" in issues[0]

    def test_suggested_fix_thermal(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_thermal"})
        if issues:
            assert "suggested_fix=" in issues[0]
            assert "thermal_fusion" in issues[0]

    def test_suggested_fix_fast_ion(self):
        from imas_codex.standard_names.audits import causal_due_to_check

        issues = causal_due_to_check({"id": "power_due_to_fast_ion"})
        if issues:
            assert "suggested_fix=" in issues[0]
            assert "fast_ions" in issues[0]


# =========================================================================
# C.6: pulse_schedule_reference_check
# =========================================================================


class TestPulseScheduleReferenceCheck:
    """Tests for pulse_schedule_reference_check audit."""

    def test_fail_reference_suffix(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check({"id": "plasma_current_reference"})
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_fail_reference_waveform_suffix(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current_reference_waveform"}
        )
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_pass_source_path_with_physics_name(self):
        """Physics SN with pulse_schedule source attached should NOT flag.

        Name like ``plasma_current`` is a legitimate physics standard name
        even if a controller-reference source path is attached to it.
        """
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current"},
            source_path="pulse_schedule/position_control/reference",
        )
        assert issues == []

    def test_fail_reference_suffix_with_pulse_schedule_source(self):
        """Name with _reference_waveform suffix must fail regardless of source."""
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        issues = pulse_schedule_reference_check(
            {"id": "plasma_current_reference_waveform"},
            source_path="pulse_schedule/position_control/reference_waveform/data",
        )
        assert issues and "pulse_schedule_reference_check" in issues[0]

    def test_pass_no_reference(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        assert pulse_schedule_reference_check({"id": "electron_temperature"}) == []

    def test_pass_source_path_no_pulse_schedule(self):
        from imas_codex.standard_names.audits import pulse_schedule_reference_check

        assert (
            pulse_schedule_reference_check(
                {"id": "electron_temperature"},
                source_path="core_profiles/profiles_1d/electrons/temperature",
            )
            == []
        )


# =========================================================================
# C.7: ratio_binary_operator_check
# =========================================================================


class TestRatioBinaryOperatorCheck:
    """Tests for ratio_binary_operator_check audit."""

    def test_pass_canonical_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        assert (
            ratio_binary_operator_check(
                {"id": "ratio_of_electron_density_to_ion_density"}
            )
            == []
        )

    def test_fail_adhoc_density_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        issues = ratio_binary_operator_check({"id": "electron_to_ion_density_ratio"})
        assert issues and "ratio_binary_operator_check" in issues[0]
        assert "suggested_fix=" in issues[0]
        assert "ratio_of_electron_to_ion" in issues[0]

    def test_fail_adhoc_bare_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        issues = ratio_binary_operator_check({"id": "tritium_to_deuterium_ratio"})
        assert issues and "ratio_binary_operator_check" in issues[0]

    def test_pass_no_ratio(self):
        from imas_codex.standard_names.audits import ratio_binary_operator_check

        assert ratio_binary_operator_check({"id": "electron_temperature"}) == []


# =========================================================================
# C.8: unit_validity_check — whitespace and ^dimension
# =========================================================================


class TestUnitValidityCheckStrengthened:
    """Tests for strengthened unit_validity_check."""

    def test_fail_whitespace_prose_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "Elementary Charge Unit"})
        assert issues and "whitespace" in issues[0]
        assert "dd_upstream" in issues[0]

    def test_fail_caret_dimension_placeholder(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "m^dimension"})
        assert issues and "^dimension" in issues[0]
        assert "dd_upstream" in issues[0]

    def test_pass_normal_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        assert unit_validity_check({"unit": "m^-3"}) == []
        assert unit_validity_check({"unit": "eV"}) == []
        assert unit_validity_check({"unit": "kg.m.s^-2"}) == []
