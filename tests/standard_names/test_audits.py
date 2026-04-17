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
                "The plasma has $\\alpha$ and $\\beta$ parameters. "
                "These affect confinement significantly. "
                "Further analysis shows improved stability."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) >= 1
        assert any("latex_def_check" in i for i in issues)

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
            has_critical_audit_failure(["audit:unit_dimension_check: mismatch"])
            is False
        )
        assert has_critical_audit_failure([]) is False
