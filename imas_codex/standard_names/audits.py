"""Post-generation audits for standard name candidates.

Six deterministic checks run after ISN validation to catch quality issues
that grammar/pydantic validation alone cannot detect. Each check returns
tagged issue strings (``"audit:<check_name>: <detail>"``) appended to the
candidate's ``validation_issues`` list.

Critical checks (quarantine on failure): latex_def_check, synonym_check,
multi_subject_check.
Non-critical (advisory only): provenance_verb_check, unit_dimension_check,
cocos_specificity_check.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _isn_process_tokens() -> frozenset[str]:
    """Return the canonical set of process tokens registered in ISN grammar.

    Queried from ``imas_standard_names.grammar.get_grammar_context()`` at runtime
    so the audit stays aligned with whichever ISN release is installed. Any token
    in this set is a legitimate ``due_to_<token>`` target and must not be flagged
    as an adjective by :func:`causal_due_to_check`.
    """
    try:
        from imas_standard_names.grammar import get_grammar_context

        ctx = get_grammar_context()
        for section in ctx.get("vocabulary_sections", []) or []:
            if section.get("segment") == "process":
                return frozenset(section.get("tokens") or ())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not load ISN process tokens: %s", exc)
    return frozenset()


# Checks whose failure demotes to quarantined
CRITICAL_CHECKS = frozenset(
    {
        "latex_def_check",
        "synonym_check",
        "multi_subject_check",
        "placeholder_check",
        "unit_validity_check",
        "generic_noun_check",
        "tautology_check",
        "spectral_suffix_check",
        "abbreviation_check",
        "name_description_consistency_check",
        "american_spelling_check",
        "description_verb_drift_check",
        "structural_dim_tag_check",
        "name_unit_consistency_check",
        "representation_artifact_check",
        "causal_due_to_check",
        "implicit_field_check",
        "density_unit_consistency_check",
        "position_coordinate_check",
        "vector_field_component_check",
        "segment_order_check",
        "aggregator_order_check",
        "named_feature_preposition_check",
        "diamagnetic_component_check",
        "amplitude_of_prefix_check",
        "mode_number_suffix_check",
        "cumulative_prefix_check",
        "pulse_schedule_reference_check",
        "ratio_binary_operator_check",
    }
)

# Map from head-noun tokens present in a standard name to the unit(s) they
# imply. Keys are tokens that appear in names; values are sets of acceptable
# units. When the name contains the token but the declared unit is not in the
# expected set, the audit raises a critical failure.
#
# These rules are deliberately conservative — only unambiguous head nouns are
# listed. Ambiguous words (``radiation``, ``field``) are left out because they
# appear across multiple physical dimensions.
_NAME_TOKEN_UNIT_EXPECTATIONS: dict[str, set[str]] = {
    # Energy head noun must be in energy units.
    "energy": {"J", "eV", "keV", "MeV", "GeV"},
    # Power implies a rate of energy delivery.
    "power": {"W", "MW", "kW"},
    # Temperature implies thermal units.
    "temperature": {"eV", "keV", "K"},
    # Pressure implies Pa (or dimensionally equivalent J/m^3).
    "pressure": {"Pa", "kPa", "MPa", "bar", "J.m^-3"},
    # Voltage implies V.
    "voltage": {"V", "kV", "mV"},
    # Angle / rotation implies rad.
    "angle": {"rad", "deg"},
    # Mass implies kg or u.
    "mass": {"kg", "u"},
    # Frequency implies Hz.
    "frequency": {"Hz", "kHz", "MHz", "GHz", "rad.s^-1", "s^-1"},
}

# Single-token names that are too generic to be self-describing standard names.
# A standard name must convey its meaning without requiring source-path context.
_GENERIC_NOUN_NAMES = frozenset(
    {
        "geometry",
        "data",
        "value",
        "quantity",
        "parameter",
        "coefficient",
        "coefficients",
        "element",
        "elements",
        "object",
        "objects",
        "node",
        "nodes",
        "index",
        "measure",
        "type",
        "name",
        "label",
        "status",
        "flag",
        "mode",
        "state",
        "version",
        "identifier",
        "metadata",
    }
)

# Regex patterns for tautological preposition chains.
# A name like "radial_position_of_reference_position" is wrong — "position_of_*_position"
# repeats the head noun. Similarly "component_of_*_component".
_TAUTOLOGY_HEADS = (
    "position",
    "component",
    "coordinate",
    "angle",
    "distance",
    "radius",
    "height",
    "width",
    "length",
)

# Tokens that indicate an unfilled prompt placeholder leaked through.
# Matches bracketed tokens containing these words anywhere in documentation/description.
_PLACEHOLDER_TOKENS = frozenset(
    {
        "condition",
        "specific condition",
        "specific physical condition",
        "quantity",
        "value",
        "unit",
        "placeholder",
        "todo",
        "fill in",
        "tbd",
    }
)

# Non-unit tokens that indicate an invalid unit expression — these are
# shape-related or semantic labels that should never appear in a unit string.
_INVALID_UNIT_TOKENS = frozenset(
    {
        "dimension",
        "rank",
        "fourier",
        "component",
        "coefficient",
        "shape",
        "index",
        "tbd",
        "n/a",
    }
)

# Provenance verbs that should not appear in standard names
_PROVENANCE_VERBS = frozenset(
    {"measured", "reconstructed", "fitted", "computed", "calculated"}
)

# Minimal unit → expected description noun mapping
_UNIT_NOUN_MAP: dict[str, set[str]] = {
    "m": {
        "position",
        "length",
        "distance",
        "radius",
        "height",
        "width",
        "displacement",
        "coordinate",
        "separation",
        "shift",
        "offset",
        "circumference",
        "perimeter",
        "major",
        "minor",
        "elongation",
    },
    "eV": {
        "temperature",
        "energy",
        "thermal",
        "potential",
        "ionization",
        "ionisation",
        "work function",
        "binding",
    },
    "K": {
        "temperature",
        "thermal",
    },
    "A": {
        "current",
    },
    "Pa": {
        "pressure",
    },
    "T": {
        "magnetic",
        "field",
    },
    "Wb": {
        "flux",
        "magnetic",
    },
    "V": {
        "voltage",
        "potential",
        "electric",
        "loop",
    },
    "W": {
        "power",
        "heating",
        "radiation",
        "radiated",
    },
    "m^-3": {
        "density",
        "concentration",
    },
    "s": {
        "time",
        "duration",
        "confinement",
        "period",
    },
    "rad": {
        "angle",
        "phase",
        "rotation",
        "toroidal",
        "poloidal",
    },
    "m^2": {
        "area",
        "cross",
        "section",
        "surface",
    },
    "m^3": {
        "volume",
    },
}

# Definition-indicator words for latex symbol definitions.
# Single words use \b word-boundary regex; multi-word phrases use substring match.
_DEFINITION_WORDS_SINGLE = frozenset(
    {"is", "are", "denotes", "represents", "where", "defined", "being"}
)
_DEFINITION_PHRASES = frozenset({"given by", "expressed as", "known as"})

# Pre-compiled word-boundary pattern for single definition words
_DEF_WORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _DEFINITION_WORDS_SINGLE) + r")\b"
)


def latex_def_check(candidate: dict[str, Any]) -> list[str]:
    """Check that every LaTeX symbol in documentation has a definition.

    Scans ``documentation`` for ``$...$`` groups and verifies each unique
    symbol has at least one definition sentence (heuristic: a sentence
    within 2 sentences of first occurrence containing the symbol, plus
    a definition-indicator word or a unit in brackets).
    """
    issues: list[str] = []
    doc = candidate.get("documentation") or ""
    if not doc:
        return issues

    # Find all inline math groups $...$  (not display $$...$$)
    # First remove display math to avoid double-matching
    doc_no_display = re.sub(r"\$\$[^$]+\$\$", " ", doc)
    symbols = set(re.findall(r"\$([^$]+)\$", doc_no_display))

    if not symbols:
        return issues

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", doc)

    for sym in symbols:
        # Skip very short fragments like single digits or operators
        sym_stripped = sym.strip()
        if len(sym_stripped) <= 1 and not sym_stripped.isalpha():
            continue

        # Skip pure numeric LaTeX like 10^{21}, 10^{-19}, 2.5\times10^{19}
        # — these are numeric magnitudes/exponents, not symbolic variables
        # requiring definition.
        if re.fullmatch(
            r"[\d.\-+]+(?:\s*\\times\s*)?1?0?\^?\{?-?\d+\}?|\d+(?:\.\d+)?|"
            r"1?0?\^\{?-?\d+\}?|\\times|\\cdot",
            sym_stripped,
        ):
            continue
        # Also skip if it's just a number followed by an exponent group
        if re.fullmatch(r"1?0?\s*\^\s*\{[-\d]+\}", sym_stripped):
            continue
        if re.fullmatch(r"\d+\^\{?-?\d+\}?", sym_stripped):
            continue

        # Skip universal physics / math constants that are self-evident:
        # \pi, 2\pi, \pi/2, \alpha, \beta, \gamma, \mu_0, \epsilon_0,
        # \hbar, c, e, k_B (optionally with a numeric scalar prefix or
        # simple rational factor). These need no definition sentence.
        if re.fullmatch(
            r"(?:\d+(?:\.\d+)?\s*)?"  # optional numeric coefficient
            r"(?:\\pi|\\alpha|\\beta|\\gamma|\\delta|\\epsilon|\\mu|"
            r"\\lambda|\\sigma|\\tau|\\omega|\\Omega|"
            r"\\mu_0|\\mu_\{0\}|\\epsilon_0|\\epsilon_\{0\}|\\hbar|"
            r"k_B|k_\{B\})"
            r"(?:\s*/\s*\d+)?",  # optional /N divisor
            sym_stripped,
        ):
            continue

        # Find first occurrence sentence index
        first_idx = None
        for i, sent in enumerate(sentences):
            if f"${sym}$" in sent or f"${sym_stripped}$" in sent:
                first_idx = i
                break

        if first_idx is None:
            continue

        # Check nearby sentences (within 2) for definition
        found_def = False
        window = sentences[max(0, first_idx - 1) : first_idx + 3]
        for sent in window:
            sent_lower = sent.lower()
            # Check for definition words near the symbol (word-boundary match)
            has_def_word = bool(_DEF_WORD_RE.search(sent_lower)) or any(
                p in sent_lower for p in _DEFINITION_PHRASES
            )
            # Check for unit in brackets/parentheses
            has_unit_bracket = bool(
                re.search(r"\([^)]*(?:eV|m|A|T|Pa|Wb|K|rad|s|W|V)[^)]*\)", sent)
            )
            if has_def_word or has_unit_bracket:
                found_def = True
                break

        if not found_def:
            issues.append(
                f"audit:latex_def_check: symbol ${sym}$ lacks a definition sentence"
            )

    return issues


# Time-derivative / rate-of-change markers that, when present in a
# description, require the name to carry an explicit tendency/change marker.
# Otherwise the name (a base quantity) contradicts the description (a rate).
_RATE_DESC_PATTERNS = (
    "instantaneous change",
    "instantaneous signed change",
    "rate of change",
    "time derivative",
    "time rate of change",
    "signed change in",
    "temporal derivative",
    "per unit time",
)

# Name prefixes/tokens that legitimately describe a rate/change quantity.
_RATE_NAME_MARKERS = (
    "tendency_of_",
    "change_in_",
    "rate_of_change_of_",
    "time_derivative_of_",
)


def description_verb_drift_check(candidate: dict[str, Any]) -> list[str]:
    """Flag name/description verb drift on rate-type paths.

    If a description claims the quantity is a time derivative or rate of
    change but the name lacks a rate marker (``tendency_of_``,
    ``change_in_``, ``rate_of_change_of_``), the name is mis-labelled as
    a base quantity. This is a critical mismatch that invites downstream
    misuse.

    Conversely, avoid the awkward literal prefix ``instant_change_`` in
    names — prefer ``change_in_`` or ``tendency_of_``. Names starting
    with ``instant_change_`` are also flagged.
    """
    issues: list[str] = []
    name = (
        str(candidate.get("id") or candidate.get("standard_name") or "").strip().lower()
    )
    description = str(candidate.get("description") or "").lower()

    if not name or not description:
        return issues

    # Guard: names starting with "instant_change_" or "instantaneous_change_"
    # should be replaced with "change_in_" or "tendency_of_".
    if name.startswith(("instant_change_", "instantaneous_change_")):
        issues.append(
            "audit:description_verb_drift_check: name begins with "
            f"'{name.split('_')[0]}_change_'; prefer 'change_in_' or "
            "'tendency_of_'"
        )
        return issues

    has_rate_desc = any(pat in description for pat in _RATE_DESC_PATTERNS)
    if not has_rate_desc:
        return issues

    has_rate_name = any(marker in name for marker in _RATE_NAME_MARKERS)
    if not has_rate_name:
        issues.append(
            "audit:description_verb_drift_check: description implies a "
            "rate/time-derivative but name lacks 'tendency_of_', "
            "'change_in_', or 'rate_of_change_of_' marker"
        )
    return issues


# Structural dimensionality tags leaked from DD data-type metadata that
# should not appear in human-readable descriptions.
_STRUCTURAL_DIM_RE = re.compile(r"\b([0-3])[dD]\b")


def structural_dim_tag_check(candidate: dict[str, Any]) -> list[str]:
    """Flag descriptions that echo DD data-type dimensionality tags.

    Tokens like ``1D``, ``2D``, ``3D`` in a description are a leak from
    the DD data type (``FLT_1D`` etc.) rather than a physically meaningful
    descriptor. This is advisory only — descriptions should describe what
    the quantity *is*, not how it is stored.
    """
    issues: list[str] = []
    description = str(candidate.get("description") or "")
    match = _STRUCTURAL_DIM_RE.search(description)
    if match:
        issues.append(
            f"audit:structural_dim_tag_check: description contains "
            f"storage-shape tag '{match.group(0)}' (remove or rephrase "
            "in terms of the physical quantity)"
        )
    return issues


def provenance_verb_check(
    candidate: dict[str, Any], source_path: str | None = None
) -> list[str]:
    """Check that name contains no provenance verbs unless source path does too.

    Standard names should describe the physical quantity, not how it was
    obtained. Words like ``measured``, ``reconstructed`` are only allowed
    when the source DD path itself contains that word.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    tokens = set(name.split("_"))
    source_tokens = set((source_path or "").replace("/", "_").split("_"))

    for verb in _PROVENANCE_VERBS:
        if verb in tokens and verb not in source_tokens:
            issues.append(
                f"audit:provenance_verb_check: name contains '{verb}' "
                f"but source path does not"
            )

    return issues


def synonym_check(
    candidate: dict[str, Any],
    existing_sns_in_domain: list[dict[str, Any]],
) -> list[str]:
    """Flag near-duplicate names with cosine similarity > 0.92.

    Compares the candidate's description embedding against precomputed
    embeddings of existing SNs in the same domain with the same unit.
    """
    issues: list[str] = []
    if not existing_sns_in_domain:
        return issues

    cand_name = candidate.get("id") or candidate.get("standard_name") or ""
    cand_unit = candidate.get("unit") or "1"

    # Get candidate embedding
    cand_embedding = candidate.get("description_embedding")
    if cand_embedding is None:
        return issues

    cand_vec = np.array(cand_embedding, dtype=np.float32)
    cand_norm = np.linalg.norm(cand_vec)
    if cand_norm == 0:
        return issues

    for existing in existing_sns_in_domain:
        ex_name = existing.get("name") or existing.get("id") or ""
        if ex_name == cand_name:
            continue
        ex_unit = existing.get("unit") or "1"
        if ex_unit != cand_unit:
            continue
        ex_embedding = existing.get("description_embedding")
        if ex_embedding is None:
            continue

        ex_vec = np.array(ex_embedding, dtype=np.float32)
        ex_norm = np.linalg.norm(ex_vec)
        if ex_norm == 0:
            continue

        cosine = float(np.dot(cand_vec, ex_vec) / (cand_norm * ex_norm))
        if cosine > 0.92:
            issues.append(
                f"audit:synonym_check: cosine={cosine:.3f} with existing "
                f"'{ex_name}' (same unit={cand_unit})"
            )

    return issues


def placeholder_check(candidate: dict[str, Any]) -> list[str]:
    """Detect unfilled prompt placeholders leaking into name/description/documentation.

    Flags bracketed tokens like ``[condition]``, ``[specific physical condition]``,
    ``[quantity]`` — these indicate the LLM copied the prompt's placeholder
    pattern verbatim instead of substituting a concrete value.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    description = candidate.get("description") or ""
    documentation = candidate.get("documentation") or ""

    # Match [anything] where the inner text contains a placeholder token
    bracket_pattern = re.compile(r"\[([^\[\]]{1,80})\]")
    for field_name, text in (
        ("name", name),
        ("description", description),
        ("documentation", documentation),
    ):
        if not text:
            continue
        for match in bracket_pattern.finditer(text):
            inner = match.group(1).lower().strip()
            # Skip markdown links [text](url) — these have concrete link text.
            end = match.end()
            if end < len(text) and text[end] == "(":
                continue
            # Only flag when the bracketed content matches a known placeholder
            # token (single words or short phrases).
            for token in _PLACEHOLDER_TOKENS:
                if token in inner:
                    issues.append(
                        f"audit:placeholder_check: unfilled placeholder '[{match.group(1)}]' "
                        f"in {field_name}"
                    )
                    break

    return issues


def unit_validity_check(candidate: dict[str, Any]) -> list[str]:
    """Flag invented / malformed unit expressions.

    Catches units containing semantic labels (e.g. ``m^dimension``, ``Wb*fourier``)
    rather than valid SI symbols. Does not validate unit algebra (left to Pydantic /
    pint); this is a defensive sanity check against LLM hallucinations.

    Also flags DD-upstream quality issues: unit strings containing whitespace
    (prose unit names like ``"Elementary Charge Unit"``) and ``^dimension``
    placeholders that escaped the DD XML without resolution.
    """
    issues: list[str] = []
    raw_unit = (candidate.get("unit") or "").strip()
    unit = raw_unit.lower()
    if not unit or unit in ("1", "dimensionless", "-", "mixed", "none"):
        return issues

    # C.8: whitespace in unit string → dd_upstream severity
    if re.search(r"\s", raw_unit):
        issues.append(
            f"audit:unit_validity_check: unit '{raw_unit}' contains "
            f"whitespace — prose unit names are not valid SI expressions; "
            f"severity=dd_upstream"
        )
        return issues

    # C.8: ^dimension placeholder → dd_upstream severity
    if "^dimension" in unit:
        issues.append(
            f"audit:unit_validity_check: unit '{raw_unit}' contains "
            f"'^dimension' placeholder — unresolved DD variable; "
            f"severity=dd_upstream"
        )
        return issues

    # Split on unit algebra operators and check each token
    tokens = re.split(r"[\s*/.^()·×]+", unit)
    for tok in tokens:
        if not tok:
            continue
        if tok in _INVALID_UNIT_TOKENS:
            issues.append(
                f"audit:unit_validity_check: unit '{raw_unit}' "
                f"contains non-unit token '{tok}'"
            )
            break
    return issues


def unit_dimension_check(candidate: dict[str, Any]) -> list[str]:
    """Heuristic check that description nouns are consistent with unit.

    Uses a minimal unit→expected-noun-set map; flags when no noun from
    the expected set appears in the description.
    """
    issues: list[str] = []
    unit = candidate.get("unit") or ""
    description = (candidate.get("description") or "").lower()

    if not unit or unit in ("1", "dimensionless", "-", "mixed") or not description:
        return issues

    # Check all unit keys for a match
    expected_nouns = _UNIT_NOUN_MAP.get(unit)
    if expected_nouns is None:
        return issues

    # Tokenize description
    desc_words = set(re.findall(r"[a-z]+", description))
    if not desc_words & expected_nouns:
        issues.append(
            f"audit:unit_dimension_check: unit='{unit}' but description "
            f"lacks expected terms {sorted(expected_nouns)[:5]}"
        )

    return issues


def name_unit_consistency_check(candidate: dict[str, Any]) -> list[str]:
    """Check that head-noun tokens in the *name* match the declared unit.

    Operates on the name alone (compose is always name-only per ADR-1).
    Catches cases like ``heating_power_due_to_ohmic``
    with unit ``J`` (should be ``W``) or ``neutral_beam_injection_unit_energy``
    with unit ``1`` (should be ``J`` or ``eV``).

    Ignores compound-unit decorations such as ``m^-3.W`` (power density) by
    also accepting units that *contain* an expected unit as a component. The
    failure is raised only when the name asserts a head dimension but the
    declared unit contains no compatible component.
    """
    issues: list[str] = []
    name = (candidate.get("id") or candidate.get("standard_name") or "").lower()
    unit = (candidate.get("unit") or "").strip()
    if not name or not unit:
        return issues
    if unit in ("1", "dimensionless", "-", "mixed", "none"):
        dimensionless = True
    else:
        dimensionless = False

    name_tokens = set(re.findall(r"[a-z]+", name))

    for token, expected_units in _NAME_TOKEN_UNIT_EXPECTATIONS.items():
        if token not in name_tokens:
            continue
        # Skip if name also contains a qualifier that shifts the head noun
        # (e.g. ``power_density`` has token ``power`` but density shifts the
        # unit to ``m^-3.W``).
        if token == "power" and "density" in name_tokens:
            continue
        if token == "energy" and "density" in name_tokens:
            continue
        if token == "angle" and ("offset" in name_tokens or "gradient" in name_tokens):
            continue
        # ``_flux`` shifts the head-noun dimension by time and area:
        # ``energy_flux`` → W.m^-2 (power per area), not energy.
        # ``mass_flux``   → kg.m^-2.s^-1, not mass.
        # ``charge_flux`` → A.m^-2, not charge.
        # The flux variant is too ambiguous (could be particle-flux of an
        # energy-bearing species) for a hard audit; defer to dimensional
        # consistency checks at the description layer.
        if "flux" in name_tokens and token in ("energy", "mass", "voltage"):
            continue
        # ``_density`` qualifier already handled for power/energy above; also
        # exempt for mass (mass_density → kg.m^-3) and pressure (rare).
        if token == "mass" and "density" in name_tokens:
            continue
        # ``center_of_mass`` is a reference point (the barycentre), not a mass
        # quantity. The mass token is part of a compound location label, not a
        # dimensional subject. Same for similar location compounds.
        if token == "mass" and "center_of_mass" in name:
            continue
        # ``_source`` / ``_sink`` shift the head noun to a rate-per-volume:
        # ``energy_source`` → W/m^3 (volumetric power density), not energy.
        # ``particle_source`` → m^-3.s^-1, etc. Defer to description-layer
        # dimensional checks.
        if ("source" in name_tokens or "sink" in name_tokens) and token in (
            "energy",
            "power",
            "mass",
            "momentum",
        ):
            continue
        # ``_diffusivity`` / ``_conductivity`` / ``_resistivity`` describe
        # transport coefficients whose units depend on what is being
        # transported, not on the prefix token. ``ion_energy_diffusivity``
        # has m^2/s (kinematic diffusivity) regardless of the ``energy``
        # qualifier — same shape as thermal diffusivity.
        if any(
            coef in name
            for coef in (
                "_diffusivity",
                "_conductivity",
                "_resistivity",
                "_viscosity",
                "_mobility",
            )
        ):
            continue
        # _peaking_factor, _profile_factor, _ratio, _fraction names are
        # dimensionless by definition — the head noun (temperature, density)
        # describes the numerator quantity, not the declared unit.
        if any(
            suffix in name
            for suffix in (
                "_peaking_factor",
                "_profile_factor",
                "_ratio",
                "_fraction",
                "_normalized",
                "_normalised",
            )
        ):
            continue
        # Meta-descriptor names: constraint_weight_of_X, exact_flag_of_X,
        # convergence_count_of_X etc. describe META properties of a physical
        # quantity X (weight, flag, count) — the head noun is the meta token
        # (weight/flag/count), not X. Their unit is dimensionless by design.
        if any(
            marker in name
            for marker in (
                "_constraint_weight_of_",
                "_constraint_weight",
                "_exact_flag",
                "_iteration_count",
                "_convergence_count",
            )
        ):
            continue

        if dimensionless:
            issues.append(
                f"audit:name_unit_consistency_check: name contains '{token}' "
                f"but unit is dimensionless ('{unit}'); expected one of "
                f"{sorted(expected_units)}"
            )
            continue

        if unit in expected_units:
            continue
        # Accept compound units that list one expected unit as a factor or
        # exponentiated component (e.g. ``m^-3.W`` for power density is
        # already filtered above; ``N.m`` for torque is not a power issue).
        tokens_in_unit = set(re.findall(r"[A-Za-z]+", unit))
        if expected_units & tokens_in_unit:
            continue
        issues.append(
            f"audit:name_unit_consistency_check: name contains '{token}' "
            f"but unit='{unit}' is not in expected set "
            f"{sorted(expected_units)}"
        )

    return issues


def multi_subject_check(candidate: dict[str, Any]) -> list[str]:
    """Detect names combining two different subject segments.

    Uses ``parse_standard_name`` from ISN grammar to check whether
    multiple ``subject_*`` segments are detected in the name.
    """
    issues: list[str] = []
    name = candidate.get("id") or candidate.get("standard_name") or ""
    if not name:
        return issues

    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(name)
        # StandardName has subject, subject2 fields (or similar)
        # Check if multiple subject values are present
        subjects = []
        if hasattr(parsed, "subject") and parsed.subject is not None:
            subjects.append(str(parsed.subject))
        # Check for compound subject patterns in the name itself
        # (two different species/subject tokens)
        if hasattr(parsed, "binary_operator") and parsed.binary_operator is not None:
            # Binary operator implies two operands — this is legitimate
            return issues
    except Exception:
        # Parse failure is handled by grammar round-trip check
        return issues

    # Heuristic: check if name contains two subject enum values.
    # Use greedy longest-match so compound subjects like
    # ``deuterium_tritium`` consume their constituent tokens and prevent
    # ``deuterium`` + ``tritium`` from being counted as two subjects.
    try:
        from imas_standard_names.grammar import Subject

        # Sort longest-first for greedy matching
        all_subjects = sorted((s.value for s in Subject), key=len, reverse=True)
        remaining = name
        matched_subjects: list[str] = []
        for sv in all_subjects:
            # Match the subject value as a whole-word substring
            # (delimited by underscores or string boundaries)
            pattern = rf"(?:^|_){re.escape(sv)}(?:_|$)"
            if re.search(pattern, remaining):
                matched_subjects.append(sv)
                # Remove matched tokens so shorter subjects sharing the
                # same tokens (e.g. ``deuterium`` inside
                # ``deuterium_tritium``) are not double-counted.
                remaining = re.sub(pattern, "_", remaining).strip("_")

        # Exempt known unit-qualifier compounds where a subject token appears
        # as a modifier rather than a true subject. These are conventional
        # particle-count conversions in the DD, not dual subjects.
        #   - `*_electron_equivalent` — ionization-equivalent particle count
        #     (e.g. hydrogen molecule released → N electrons on full ionization)
        #   - `*_electron_temperature_equivalent` — temperature expressed as kT/e
        if name.endswith("_electron_equivalent"):
            matched_subjects = [s for s in matched_subjects if s != "electron"]

        # Exempt ratio/comparison patterns: ``{species1}_to_{species2}_…``
        # uses ``_to_`` as a conventional connector between numerator and
        # denominator species (e.g. tritium_to_deuterium_density_ratio).
        if "_to_" in name and len(matched_subjects) == 2:
            matched_subjects = []

        # Exempt metadata/flag descriptors — names ending in ``_flag``,
        # ``_index``, or containing ``_state_`` reference classification
        # attributes rather than two physical subjects. E.g.
        # ``ion_state_neutral_flag`` describes a flag on the ion-state
        # enum; ``neutral`` is an enum value, not a second subject.
        _META_TOKENS = ("_flag", "_index", "_state_", "_type_flag")
        if any(tok in name for tok in _META_TOKENS):
            matched_subjects = []

        if len(matched_subjects) >= 2:
            issues.append(
                f"audit:multi_subject_check: name contains multiple subjects: "
                f"{matched_subjects}"
            )
    except Exception:
        pass

    return issues


def cocos_specificity_check(
    candidate: dict[str, Any],
    source_cocos_type: str | None = None,
) -> list[str]:
    """If source path has COCOS transformation type, description must mention COCOS.

    Checks that the documentation contains the string ``COCOS`` and a digit
    when the source DD path is tagged with a ``cocos_transformation_type``.
    """
    issues: list[str] = []
    if not source_cocos_type:
        return issues

    doc = candidate.get("documentation") or ""
    desc = candidate.get("description") or ""
    combined = doc + " " + desc

    has_cocos = "COCOS" in combined or "cocos" in combined.lower()
    has_digit = bool(re.search(r"COCOS\s*\d", combined, re.IGNORECASE))

    if not (has_cocos and has_digit):
        issues.append(
            f"audit:cocos_specificity_check: source has cocos_transformation_type="
            f"'{source_cocos_type}' but documentation lacks 'COCOS <digit>'"
        )

    return issues


def generic_noun_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names that are bare generic nouns without physics-specific qualifiers.

    A standard name must be self-describing. Names like ``geometry``, ``data``,
    ``value``, or ``measure`` require source-path context to interpret and
    cannot be used as standalone identifiers across facilities.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    if not name:
        return []
    tokens = [t for t in name.split("_") if t]
    if len(tokens) == 0:
        return []

    if len(tokens) == 1 and tokens[0] in _GENERIC_NOUN_NAMES:
        return [
            f"audit:generic_noun_check: name '{name}' is a bare generic noun; "
            "add a physics-specific qualifier (e.g. 'grid_object_geometry' "
            "instead of 'geometry')"
        ]

    if len(tokens) == 2:
        if tokens[-1] in _GENERIC_NOUN_NAMES and tokens[0] in {
            "raw",
            "input",
            "output",
            "generic",
            "basic",
        }:
            return [
                f"audit:generic_noun_check: name '{name}' uses a generic "
                "qualifier + generic noun; specify the physical quantity"
            ]

    return []


def tautology_check(candidate: dict[str, Any]) -> list[str]:
    """Flag tautological preposition chains like 'position_of_X_position'.

    Detects patterns where the same head noun (position, component, coordinate,
    etc.) appears on both sides of an ``_of_`` connector. These names are
    stylistically awkward and signal a missing physics-specific qualifier.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    if "_of_" not in name:
        return []

    parts = name.split("_of_")
    if len(parts) < 2:
        return []

    issues: list[str] = []
    for i in range(len(parts) - 1):
        left_tokens = parts[i].split("_")
        right_tokens = parts[i + 1].split("_")
        if not left_tokens or not right_tokens:
            continue
        left_head = left_tokens[-1]
        right_head = right_tokens[-1]
        if left_head in _TAUTOLOGY_HEADS and left_head == right_head:
            issues.append(
                f"audit:tautology_check: name '{name}' repeats head noun "
                f"'{left_head}' across '_of_' (tautological chain); "
                f"replace the second occurrence with a specific qualifier"
            )
            break

    return issues


def spectral_suffix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names ending in spectral-decomposition suffixes.

    Names like ``*_fourier_coefficients``, ``*_fourier_modes``, or
    ``*_harmonics`` place the decomposition type at the end as a generic
    suffix. The preferred pattern is ``mode_<n>_of_<quantity>`` or to name
    the decomposition component explicitly (e.g. ``fourier_amplitude_of_X``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    bad_suffixes = (
        "_fourier_coefficients",
        "_fourier_coefficient",
        "_fourier_modes",
        "_fourier_mode",
        "_harmonics",
        "_harmonic_coefficients",
    )
    for suf in bad_suffixes:
        if name.endswith(suf):
            return [
                f"audit:spectral_suffix_check: name '{name}' ends with "
                f"spectral suffix '{suf}'; use a mode-prefixed or "
                "amplitude-of-quantity pattern instead"
            ]
    return []


# Abbreviation prefixes/infixes forbidden by NC-5. A standard name must spell
# the concept out in full — no truncation or contraction.
_FORBIDDEN_ABBREVIATIONS = (
    ("norm_", "normalized_"),
    ("_norm_", "_normalized_"),
    ("perp_", "perpendicular_"),
    ("_perp_", "_perpendicular_"),
    ("par_", "parallel_"),
    ("_par_", "_parallel_"),
    ("temp_", "temperature_"),
    ("_temp_", "_temperature_"),
    ("pos_", "position_"),
    ("_pos_", "_position_"),
    ("max_", "maximum_"),
    ("_max_", "_maximum_"),
    ("min_", "minimum_"),
    ("_min_", "_minimum_"),
    ("avg_", "average_"),
    ("_avg_", "_average_"),
    ("sep_", "separatrix_"),
    ("_sep_", "_separatrix_"),
    ("ec_", "electron_cyclotron_"),
    ("_ec_", "_electron_cyclotron_"),
    ("ic_", "ion_cyclotron_"),
    ("_ic_", "_ion_cyclotron_"),
    ("nbi_", "neutral_beam_injector_"),
    ("_nbi_", "_neutral_beam_injector_"),
    ("lh_", "lower_hybrid_"),
    ("_lh_", "_lower_hybrid_"),
)


def abbreviation_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names that use truncated/abbreviated concept words.

    NC-5 mandates spelled-out concept words. Common offenders that slip past
    the LLM despite the prompt rule: ``norm_``, ``perp_``, ``par_``,
    ``temp_``, ``pos_``, ``max_``, ``min_``, ``sep_``. This audit catches
    them deterministically.

    False-positive safety: ``min``, ``max``, and ``avg`` are only flagged at
    token boundaries (prefix, suffix, or between underscores). Chemical
    element symbols and unit tokens are not affected since this audit only
    inspects the standard-name identifier, not units or documentation.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    tokens = set(name.split("_"))
    issues: list[str] = []
    seen_bare: set[str] = set()
    for abbrev, full in _FORBIDDEN_ABBREVIATIONS:
        # Strict token-boundary match: the abbreviation must appear as a
        # whole token in the name, never as a letter subsequence inside
        # another word (e.g. ``ic`` must not match ``ionic``).
        bare = abbrev.strip("_")
        if bare in seen_bare:
            continue
        if bare in tokens:
            issues.append(
                f"audit:abbreviation_check: name '{name}' contains "
                f"abbreviation '{bare}'; spell as '{full.strip('_')}'"
            )
            seen_bare.add(bare)
            break  # one report per name is sufficient
    return issues


# Description tokens that indicate spectral/decomposition semantics. If the
# description claims the quantity is a Fourier coefficient/spectral mode but
# the name carries none of these markers, the name-description pair is
# inconsistent.
_SPECTRAL_DESC_PATTERNS = (
    "fourier coefficient",
    "fourier mode",
    "spectral coefficient",
    "spectral mode",
    "harmonic amplitude",
    "harmonic coefficient",
)
_SPECTRAL_NAME_MARKERS = (
    "mode_",
    "_mode_",
    "_amplitude",
    "fourier",
    "harmonic",
    "spectral",
)


_UK_TO_US_SPELLING = {
    "normalised": "normalized",
    "polarised": "polarized",
    "magnetised": "magnetized",
    "ionised": "ionized",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analysing": "analyzing",
    "organise": "organize",
    "organised": "organized",
    "organising": "organizing",
    "behaviour": "behavior",
    "colour": "color",
    "flavour": "flavor",
    "centre": "center",
    "fibre": "fiber",
    "metre": "meter",
    "metres": "meters",
    "modelled": "modeled",
    "modelling": "modeling",
    "labelled": "labeled",
    "labelling": "labeling",
    "travelled": "traveled",
    "travelling": "traveling",
    "fuelled": "fueled",
    "fuelling": "fueling",
    "channelled": "channeled",
    "channelling": "channeling",
    "signalled": "signaled",
    "signalling": "signaling",
    "catalogue": "catalog",
    "programme": "program",
}

_UK_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _UK_TO_US_SPELLING) + r")\b",
    re.IGNORECASE,
)


def american_spelling_check(candidate: dict[str, Any]) -> list[str]:
    """Flag British spellings in the name or any prose field.

    The ISN catalog uses American spelling throughout (``normalized`` not
    ``normalised``, ``polarized`` not ``polarised``). Names or
    descriptions containing British variants violate NC-17 and are
    quarantined so they can be regenerated with the canonical spelling.
    """
    issues: list[str] = []
    name = str(candidate.get("id") or candidate.get("name") or "").strip()
    fields: list[tuple[str, str]] = []
    if name:
        fields.append(("name", name.replace("_", " ")))
    for fld in ("description", "documentation", "validity_domain"):
        val = candidate.get(fld)
        if isinstance(val, str) and val.strip():
            fields.append((fld, val))
    constraints = candidate.get("constraints")
    if isinstance(constraints, list):
        for i, c in enumerate(constraints):
            if isinstance(c, str) and c.strip():
                fields.append((f"constraints[{i}]", c))

    seen: set[tuple[str, str]] = set()
    for field_name, text in fields:
        for m in _UK_WORD_RE.finditer(text):
            uk = m.group(0).lower()
            us = _UK_TO_US_SPELLING[uk]
            key = (field_name, uk)
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                f"audit:american_spelling_check: field '{field_name}' "
                f"contains British spelling '{m.group(0)}'; use '{us}' "
                f"(ISN catalog follows American convention — NC-17)"
            )
    return issues


def name_description_consistency_check(candidate: dict[str, Any]) -> list[str]:
    """Flag names whose description asserts a different concept than the name.

    Specifically detects the case where the description describes a
    Fourier/spectral decomposition but the name is simply the underlying
    quantity (e.g. ``normal_component_of_magnetic_field`` described as
    "Fourier coefficients of the normal component ..."). Either the name
    must mark the decomposition explicitly, or the description must be
    rewritten to describe the underlying quantity.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    description = str(candidate.get("description") or "").lower()
    if not name or not description:
        return []
    # Only flag when description strongly implies a decomposition
    if not any(pat in description for pat in _SPECTRAL_DESC_PATTERNS):
        return []
    # But the name carries no decomposition marker
    if any(marker in name for marker in _SPECTRAL_NAME_MARKERS):
        return []
    return [
        f"audit:name_description_consistency_check: description of '{name}' "
        "claims a spectral/Fourier decomposition but the name encodes only "
        "the underlying quantity; either add a decomposition marker to the "
        "name or rewrite the description"
    ]


_REPRESENTATION_NAME_RE = re.compile(
    r"(?:"
    # Generic coefficient / basis / spline / ggd / fourier suffixes
    r"_(?:coefficients|ggd|basis|spline|fourier_modes|harmonics_coefficients)"
    # GGD-coefficient variants
    r"|_ggd_coefficients"
    r"|_coefficient_on_ggd"
    # Interpolation coefficient variants (singular/plural, optional _on_ggd)
    r"|_interpolation_coefficients?(?:_on_ggd)?"
    # Finite-element coefficient variants (base + real/imaginary split)
    r"|_finite_element(?:_interpolation)?_coefficients?"
    r"|_finite_element_coefficients_(?:real|imaginary)_part"
    r")"
    r"(?:_|$)"
)

#: Heuristic regex: bare ``_on_ggd$`` suffix — flagged only when the
#: DD source path carries a GGD marker (``/ggd/`` or ``/grids_ggd/``).
#: Avoids false positives on legitimate ``_on_<other>`` suffixes.
_ON_GGD_SUFFIX_RE = re.compile(r"_on_ggd$")


def representation_artifact_check(
    candidate: dict[str, Any], source_path: str | None = None
) -> list[str]:
    """Flag names whose final tokens describe a basis-function representation.

    Names ending in ``_coefficients``, ``_ggd_coefficients``,
    ``_coefficient_on_ggd``, ``_interpolation_coefficient(s)(_on_ggd)``,
    ``_finite_element_coefficients_(real|imaginary)_part``, ``_basis``,
    ``_spline``, ``_fourier_modes`` etc. are storage representations of an
    underlying physical field, not standalone physics concepts.  They should
    be quarantined and the corresponding source path skipped at classification.

    A bare ``_on_ggd$`` suffix is flagged only when *source_path* carries a
    GGD marker (``/ggd/`` or ``/grids_ggd/``) — this avoids false positives
    on legitimate ``_on_<other>`` compound terms.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    if _REPRESENTATION_NAME_RE.search(name):
        return [
            f"audit:representation_artifact_check: name '{name}' encodes a "
            "basis-function or grid representation; the underlying physical "
            "quantity already has a standard name on the sibling path — "
            "this path should have been classified as skip"
        ]
    # Heuristic _on_ggd$ — only fire when DD source path carries a GGD marker
    if (
        _ON_GGD_SUFFIX_RE.search(name)
        and source_path
        and ("/ggd/" in source_path or "/grids_ggd/" in source_path)
    ):
        return [
            f"audit:representation_artifact_check: name '{name}' ends in "
            "'_on_ggd' and its DD source path contains a GGD marker — this "
            "is a grid-representation storage node, not a physics concept; "
            "the source path should have been classified as skip"
        ]
    return []


# Verbs/processes that are mis-used with the ``due_to_<X>`` template.  These
# fall into two classes:
#  - ``during_X`` would be more accurate (the X is a temporal event, not a
#    physical cause): disruption, breakdown, ramp_up, ramp_down, flat_top
#  - ``due_to_X_<verb>`` is required (X is an adjective, not a process noun):
#    ohmic → ohmic_dissipation/ohmic_heating
_DURE_TO_TEMPORAL = {
    "disruption",
    "breakdown",
    "ramp_up",
    "ramp_down",
    "flat_top",
    "shutdown",
    "startup",
}
_DURE_TO_ADJECTIVE = {
    "ohmic": "ohmic_dissipation or ohmic_heating",
    "neutral_beam": "neutral_beam_injection",
    "wave": "wave_heating",
    "halo": "halo_currents",
    "runaway": "runaway_electrons",
    "fast_ion": "fast_ions",
    "alpha": "alpha_particle_heating",
    "resistive": "resistive_dissipation or resistive_diffusion",
    "non_inductive": "non_inductive_drive or non_inductive_current_drive",
    "inductive": "inductive_drive",
    "turbulent": "turbulent_transport",
    "neoclassical": "neoclassical_transport",
    "anomalous": "anomalous_transport",
    "thermal": "thermal_fusion",
}


def causal_due_to_check(candidate: dict[str, Any]) -> list[str]:
    """Flag misuse of the ``due_to_<process>`` grammatical template.

    The ``due_to_`` template asserts a causal physical process.  It is wrong
    when the trailing token is a temporal event (use ``during_<event>``) or
    a bare adjective (use ``due_to_<adjective>_<process_noun>``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if "_due_to_" not in name:
        return []
    issues: list[str] = []
    suffix = name.split("_due_to_", 1)[1]

    isn_processes = _isn_process_tokens()
    # Exempt any ISN-registered process token — these are canonically valid
    # targets of the ``due_to_`` template regardless of English part-of-speech.
    if suffix in isn_processes or suffix.split("_", 1)[0] in isn_processes:
        return []

    for event in _DURE_TO_TEMPORAL:
        if event in isn_processes:
            continue
        if suffix == event or suffix.startswith(event + "_"):
            issues.append(
                f"audit:causal_due_to_check: name '{name}' uses 'due_to_{event}' — "
                f"'{event}' is a temporal event, not a physical process; use "
                f"'during_{event}' instead"
            )
            break
    for adj, suggestion in _DURE_TO_ADJECTIVE.items():
        if adj in isn_processes:
            continue
        if suffix == adj or suffix == adj + "_":
            issues.append(
                f"audit:causal_due_to_check: name '{name}' uses 'due_to_{adj}' — "
                f"'{adj}' is an adjective, not a process noun; "
                f"suggested_fix=due_to_{suggestion}"
            )
            break
    return issues


FIELD_DEVICE_WHITELIST = {
    "vacuum_toroidal_field_function",
    "vacuum_toroidal_field_flux_function",
    "resistance_of_poloidal_field_coil",
}

_FIELD_QUALIFIERS = (
    "magnetic",
    "electric",
    "radiation",
    "displacement",
    "velocity",
    "temperature",
    "pressure",
    "density",
    "flow",
    "vector",
)


def implicit_field_check(candidate: dict[str, Any]) -> list[str]:
    """Flag bare ``_field`` token without a qualifier (e.g. ``vacuum_toroidal_field``).

    The IMAS catalog and ISN convention require explicit ``magnetic_field``,
    ``electric_field``, etc. — never the colloquial bare ``field``.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    # Device whitelist: names referring to physical devices (field coils,
    # toroidal-field machines) use "field" as a device qualifier, not a
    # physics-field concept.
    if name in FIELD_DEVICE_WHITELIST or "_field_coil" in name:
        return []
    tokens = name.split("_")
    issues: list[str] = []
    for i, tok in enumerate(tokens):
        if tok != "field":
            continue
        prev = tokens[i - 1] if i > 0 else ""
        if prev not in _FIELD_QUALIFIERS:
            issues.append(
                f"audit:implicit_field_check: name '{name}' contains bare '_field' "
                f"after '{prev or '<start>'}'; qualify as 'magnetic_field', "
                f"'electric_field', etc."
            )
            break
    return issues


def density_unit_consistency_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_density`` suffix when the declared unit lacks an inverse-length factor.

    A "density" in physics is a quantity per unit volume / area / length. The
    declared unit must therefore include ``m^-3`` (volumetric), ``m^-2``
    (areal), or ``m^-1`` (linear). Names ending in ``_density`` whose unit is
    a bare extensive quantity (e.g. ``kg.m.s^-1`` for momentum) are misnamed —
    drop ``_density`` or rename to reflect the actual quantity.

    Examples flagged:
    - ``toroidal_angular_momentum_density`` with unit ``kg.m.s^-1`` (linear
      momentum, not density).
    - ``electron_pressure_density`` with unit ``Pa`` (pressure already has
      energy-per-volume dimensions; ``_density`` is redundant).
    """
    name = candidate.get("id", "")
    unit = (candidate.get("unit") or "").strip()
    if not name or not unit:
        return []
    if "_density" not in name and not name.endswith("_density"):
        return []
    # Skip constraint-metadata suffixes: names like
    # ``toroidal_current_density_constraint_measurement_time`` carry
    # ``_density`` in the base quantity, not in the metadata suffix.
    # The unit refers to the suffix semantics (e.g. ``s`` for time).
    _CONSTRAINT_SUFFIXES = (
        "_constraint_measurement_time",
        "_constraint_weight",
        "_constraint_reconstructed",
        "_constraint_measured",
        "_constraint_time_measurement",
        "_constraint_position",
        "_constraint",
    )
    for suffix in _CONSTRAINT_SUFFIXES:
        if name.endswith(suffix):
            return []
    # Meta-prefix patterns: ``measurement_time_of_X_density_constraint``,
    # ``time_of_X_density_*`` describe a meta property of the constraint,
    # not the density itself. Unit refers to the meta property (s, weight).
    _META_PREFIXES = (
        "measurement_time_of_",
        "time_of_",
        "position_of_",
        "weight_of_",
        "exact_flag_of_",
    )
    for prefix in _META_PREFIXES:
        if name.startswith(prefix):
            return []
    # Acceptable density unit factors: any negative power of m.
    if "m^-" in unit or "m**-" in unit:
        return []
    # Special case: dimensionless density (rare but valid for fractions/probabilities)
    # is not flagged — declared unit "1" is allowed.
    if unit in {"1", ""}:
        return []
    return [
        f"audit:density_unit_consistency_check: name '{name}' ends with "
        f"'_density' but declared unit '{unit}' has no inverse-length factor "
        f"(expected m^-1, m^-2, or m^-3). Either drop '_density' or correct "
        f"the unit."
    ]


def vector_field_component_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_coordinate_of_<vector_field>`` and recommend ``_component_of_<vector_field>``.

    The cylindrical-coordinate vocabulary (``radial``, ``vertical``, ``toroidal``,
    ``major_radius``, ``z_coordinate``) describes a *point in space*. When the
    target ``X`` is itself a vector field (surface normal, magnetic field vector,
    velocity vector), the correct usage is ``<axis>_component_of_<X>`` — you
    project the vector onto an axis, you do not extract a coordinate.

    Caught from equilibrium iteration:
    ``vertical_coordinate_of_surface_normal`` should be
    ``vertical_component_of_surface_normal``.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    vector_field_tails = (
        "surface_normal",
        "magnetic_field_vector",
        "electric_field_vector",
        "velocity_vector",
        "current_density_vector",
        "poynting_vector",
    )
    issues: list[str] = []
    for axis in ("radial", "vertical", "toroidal"):
        bad = f"{axis}_coordinate_of_"
        if bad not in name:
            continue
        for tail in vector_field_tails:
            if name.endswith(bad + tail) or (bad + tail + "_") in name:
                issues.append(
                    f"audit:vector_field_component_check: name '{name}' applies "
                    f"'_coordinate_of_' to vector field '{tail}'; rename to "
                    f"'{axis}_component_of_{tail}' (vectors have components, "
                    f"points have coordinates)."
                )
    return issues


def position_coordinate_check(candidate: dict[str, Any]) -> list[str]:
    """Flag colloquial ``_position_of_X`` and recommend canonical coordinate vocabulary.

    Names like ``vertical_position_of_antenna``, ``radial_position_of_X`` and
    ``toroidal_position_of_X`` should use the canonical coordinate vocabulary
    that aligns with cylindrical (R, φ, Z) tokamak conventions:

    - ``radial_position_of_X`` → ``major_radius_of_X``.
    - ``toroidal_position_of_X`` → ``toroidal_angle_of_X``.
    - ``vertical_position_of_X`` → ``vertical_coordinate_of_X`` or
      ``z_coordinate_of_X``.

    The check fires unconditionally on the colloquial name pattern (it does not
    require a confirming description) because the canonical vocabulary already
    covers every R/Z/φ-coordinate use case in IMAS. Without this check,
    both forms would leak into the catalog as unintended synonyms.
    """
    name = candidate.get("id", "")
    if not name:
        return []
    issues: list[str] = []
    patterns = (
        ("radial_position_of_", "major_radius_of_<X>"),
        ("toroidal_position_of_", "toroidal_angle_of_<X>"),
        (
            "vertical_position_of_",
            "vertical_coordinate_of_<X> or z_coordinate_of_<X>",
        ),
    )
    for prefix, suggested in patterns:
        if prefix in name:
            issues.append(
                f"audit:position_coordinate_check: name '{name}' uses "
                f"colloquial '{prefix.rstrip('_')}_' form; rename to "
                f"{suggested} (cylindrical-coordinate canonical vocabulary)."
            )
    return issues


def segment_order_check(candidate: dict[str, Any]) -> list[str]:
    """Flag Component tokens appearing as a trailing suffix instead of a prefix.

    ISN grammar places Component segments (``toroidal``, ``poloidal``, ``radial``,
    ``parallel``, ``perpendicular``, ``vertical``, ``diamagnetic``) either as a
    leading prefix or via the ``<axis>_component_of_<quantity>`` preposition. A
    trailing ``_<component>`` suffix after the quantity reverses segment order.

    Caught from transport iteration:
    ``ion_rotation_frequency_toroidal`` → ``toroidal_ion_rotation_frequency`` or
    ``toroidal_component_of_ion_rotation_frequency``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    component_suffixes = (
        "toroidal",
        "poloidal",
        "radial",
        "parallel",
        "perpendicular",
        "vertical",
        "diamagnetic",
    )
    issues: list[str] = []
    for comp in component_suffixes:
        if not name.endswith(f"_{comp}"):
            continue
        stem = name[: -(len(comp) + 1)]
        # Only flag when the stem is a substantive quantity (has at least two
        # tokens AND contains no other component token as a prefix already).
        stem_tokens = stem.split("_")
        if len(stem_tokens) < 2:
            continue
        if stem_tokens[0] in component_suffixes:
            continue
        issues.append(
            f"audit:segment_order_check: name '{name}' ends with component "
            f"token '_{comp}'; Component segments must precede the Subject or "
            f"use '<axis>_component_of_<quantity>'. Rename to "
            f"'{comp}_{stem}' or '{comp}_component_of_{stem}'."
        )
    return issues


_AGGREGATOR_SUFFIXES = (
    "volume_averaged",
    "flux_surface_averaged",
    "surface_averaged",
    "line_averaged",
    "density_averaged",
    "time_averaged",
)


def aggregator_order_check(candidate: dict[str, Any]) -> list[str]:
    """Flag Aggregator tokens appearing as a trailing suffix instead of a prefix.

    ISN grammar places Aggregator segments (``volume_averaged``,
    ``flux_surface_averaged``, ``line_averaged``, ``time_averaged``, etc.) as a
    prefix before the physical base, not as a trailing suffix after it.

    Caught from transport iteration:
    ``ion_temperature_volume_averaged`` → ``volume_averaged_ion_temperature``
    (matches pattern already used by ``volume_averaged_electron_temperature``).
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    issues: list[str] = []
    for agg in _AGGREGATOR_SUFFIXES:
        suffix = f"_{agg}"
        if not name.endswith(suffix):
            continue
        stem = name[: -len(suffix)]
        # Skip if the aggregator is immediately after another aggregator prefix
        # (defensive — unlikely in practice).
        if not stem:
            continue
        issues.append(
            f"audit:aggregator_order_check: name '{name}' ends with aggregator "
            f"token '_{agg}'; Aggregator segments must precede the Subject/Base. "
            f"Rename to '{agg}_{stem}'."
        )
    return issues


_NAMED_FEATURE_TOKENS = (
    "magnetic_axis",
    "plasma_boundary",
    "last_closed_flux_surface",
    "separatrix",
    "x_point",
    "o_point",
    "strike_point",
    "inner_strike_point",
    "outer_strike_point",
    "stagnation_point",
)


def named_feature_preposition_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_at_<named_feature>`` when ``_of_<named_feature>`` is canonical.

    When a scalar property is evaluated at a named geometric feature (magnetic
    axis, x-point, plasma boundary, separatrix, strike point), the possessive
    ``_of_`` form is canonical and prevents silent synonym pairs such as
    ``poloidal_magnetic_flux_at_magnetic_axis`` vs
    ``poloidal_magnetic_flux_of_magnetic_axis``.

    Caught from transport iteration:
    ``poloidal_magnetic_flux_at_magnetic_axis`` →
    ``poloidal_magnetic_flux_of_magnetic_axis``.
    ``loop_voltage_at_last_closed_flux_surface`` →
    ``loop_voltage_of_last_closed_flux_surface``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if not name:
        return []
    issues: list[str] = []
    for feat in _NAMED_FEATURE_TOKENS:
        at_pattern = f"_at_{feat}"
        if name.endswith(at_pattern) or at_pattern + "_" in name:
            suggested = name.replace(at_pattern, f"_of_{feat}")
            issues.append(
                f"audit:named_feature_preposition_check: name '{name}' uses "
                f"'_at_{feat}'; named geometric features take the possessive "
                f"'_of_' form. Rename to '{suggested}'."
            )
    return issues


_DIAMAGNETIC_COMPONENT_PATTERN = "diamagnetic_component_of_"


def diamagnetic_component_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``diamagnetic_component_of_*`` — diamagnetic is a drift, not a component.

    ``diamagnetic`` labels a specific drift velocity ``v_dia = B × ∇p / (qnB²)``,
    not a spatial projection axis like ``toroidal`` or ``poloidal``. Using
    ``diamagnetic_component_of_<X>`` therefore either:

    - Makes no physical sense for scalars and projected fields (e.g.
      ``diamagnetic_component_of_electric_field``), or
    - Is redundant for a drift velocity (``v_dia`` IS the diamagnetic drift,
      not a component of something else).

    Canonical constructions:
    - For the drift velocity itself → ``diamagnetic_drift_velocity`` (no
      ``_component_of_``).
    - For a flux driven by the diamagnetic drift → ``diamagnetic_<base>`` or
      ``<base>_due_to_diamagnetic_drift``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    if _DIAMAGNETIC_COMPONENT_PATTERN not in name:
        return []
    tail = name.split(_DIAMAGNETIC_COMPONENT_PATTERN, 1)[1]
    return [
        f"audit:diamagnetic_component_check: name '{name}' uses "
        f"'diamagnetic_component_of_{tail}' — 'diamagnetic' labels a drift "
        f"(v_dia = B × ∇p / (qnB²)), not a spatial projection axis. Use "
        f"'diamagnetic_drift_velocity' for the drift itself, or "
        f"'<base>_due_to_diamagnetic_drift' for a flux driven by it."
    ]


def amplitude_of_prefix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``amplitude_of_<X>`` / ``phase_of_<X>`` / ``magnitude_of_<X>`` prefix forms.

    For the amplitude, phase, magnitude, real part or imaginary part of a
    quantity ``<X>``, the canonical ISN form is the noun-suffix construction
    ``<X>_amplitude``, ``<X>_phase``, ``<X>_magnitude``, ``<X>_real_part``,
    ``<X>_imaginary_part``. The prefix form ``amplitude_of_<X>`` and
    siblings break the grammar when ``<X>`` contains a ``_of_`` or
    ``component_of_`` chain (e.g. ``amplitude_of_parallel_component_of_*``
    fails the vocabulary consistency check because ``amplitude_of_parallel``
    is not a Component token). Use the noun-suffix form consistently.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    prefixes = (
        "amplitude_of_",
        "phase_of_",
        "magnitude_of_",
        "real_part_of_",
        "imaginary_part_of_",
        "modulus_of_",
    )
    for prefix in prefixes:
        if name.startswith(prefix):
            noun = prefix[:-4]  # strip trailing "_of_"
            tail = name[len(prefix) :]
            return [
                f"audit:amplitude_of_prefix_check: name '{name}' uses "
                f"'{prefix}<X>' prefix — canonical ISN form is the "
                f"noun-suffix '{tail}_{noun}'. Prefix forms break grammar "
                f"when <X> contains '_of_' or 'component_of_' chains."
            ]
    return []


def mode_number_suffix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ``_per_<axis>_mode_number`` — canonical suffix drops ``_number``.

    The spectral qualifier is ``_per_toroidal_mode`` or ``_per_poloidal_mode``;
    the ``_number`` token is redundant because the mode index is implicit.
    Within a batch the spelling must be consistent: never emit both
    ``_per_toroidal_mode`` and ``_per_toroidal_mode_number``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    bad_suffixes = (
        "_per_toroidal_mode_number",
        "_per_poloidal_mode_number",
    )
    for suffix in bad_suffixes:
        if name.endswith(suffix):
            canonical = suffix.rsplit("_number", 1)[0]
            return [
                f"audit:mode_number_suffix_check: name '{name}' ends with "
                f"'{suffix}' — canonical suffix is '{canonical}' (drop "
                f"'_number'; the mode index is implicit)."
            ]
    return []


def cumulative_prefix_check(candidate: dict[str, Any]) -> list[str]:
    """Flag spatial-integration misnomers like ``cumulative_``/``integrated_``/``running_``.

    DD leaf names ending in ``_inside`` (e.g. ``power_inside_thermal_n_tor``,
    ``current_tor_inside``) denote a quantity integrated inside the enclosing
    flux surface. The canonical ISN suffix is ``_inside_flux_surface`` placed
    directly after the quantity; ``cumulative_`` / ``integrated_`` / ``running_``
    lose the geometric meaning and are not part of the ISN grammar vocabulary.

    NOTE: ``accumulated_`` is NOT flagged here — DD gas-injection and coil-charge
    paths use ``accumulated_`` to denote a running total over time, which is a
    distinct physical concept from spatial flux-surface integration. Time
    accumulation is handled via the ISN process / transformation vocabulary.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    # Meta-descriptor prefixes refer to a property OF a named quantity, not to a
    # cumulative/integrated quantity itself. The "integrated_" inside is part of
    # the inner quantity's name (e.g. "line_integrated_electron_density"), so this
    # audit must not fire.
    _META_PREFIXES = (
        "measurement_time_of_",
        "time_of_",
        "position_of_",
        "weight_of_",
        "exact_flag_of_",
    )
    # Also skip meta-flag suffixes that wrap a quantity name (e.g. *_exact_flag,
    # *_iteration_count, *_convergence_count, *_constraint_weight*).
    _META_SUFFIXES = (
        "_exact_flag",
        "_iteration_count",
        "_convergence_count",
        "_constraint_weight",
    )
    if any(name.startswith(p) for p in _META_PREFIXES):
        return []
    if any(s in name for s in _META_SUFFIXES):
        return []
    bad_tokens = ("cumulative_", "integrated_", "running_")
    tokens = name.split("_")
    for bad in bad_tokens:
        stem = bad.rstrip("_")
        if stem in tokens:
            return [
                f"audit:cumulative_prefix_check: name '{name}' contains "
                f"'{stem}_' — for DD `_inside`-style quantities use the "
                f"suffix `_inside_flux_surface` placed after the quantity "
                f"instead of prefixing with `{stem}_`."
            ]
    return []


# ---- Regex for ad-hoc ratio patterns (C.7) --------------------------------
_ADHOC_RATIO_RE = re.compile(r"^(.+?)_to_(.+?)_ratio$")


def pulse_schedule_reference_check(
    candidate: dict[str, Any],
    source_path: str | None = None,
) -> list[str]:
    """Flag reference/reference-waveform sentinels from pulse_schedule IDS.

    Controller reference targets live under ``pulse_schedule/.../reference``
    or ``pulse_schedule/.../reference_waveform`` and are not physics standard
    name candidates.  Severity: critical.

    Triggers when:
    1. ``source_path`` matches ``pulse_schedule/.+/reference(_waveform)?``
       (including deeper children like ``.../reference_waveform/data``).
    2. Name ends with ``_reference`` or ``_reference_waveform``.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()
    issues: list[str] = []

    # Check source path first
    if source_path:
        if re.match(r"pulse_schedule/.+/reference(_waveform)?(/|$)", source_path):
            issues.append(
                f"audit:pulse_schedule_reference_check: source path "
                f"'{source_path}' is a controller reference target; "
                f"not a physics SN candidate; severity=critical"
            )
            return issues

    # Check name suffix
    if name.endswith("_reference") or name.endswith("_reference_waveform"):
        issues.append(
            f"audit:pulse_schedule_reference_check: name '{name}' ends with "
            f"a reference/reference_waveform suffix — likely a controller "
            f"reference target, not a physics SN candidate; severity=critical"
        )

    return issues


def ratio_binary_operator_check(candidate: dict[str, Any]) -> list[str]:
    """Flag ad-hoc ratio naming patterns; enforce ``ratio_of_<A>_to_<B>`` form.

    The ISN canonical form for ratios is ``ratio_of_<A>_to_<B>`` (ISN-10).
    Ad-hoc patterns like ``<A>_to_<B>_density_ratio`` or ``<A>_to_<B>_ratio``
    are rejected with a suggested rewrite.
    """
    name = str(candidate.get("id") or candidate.get("name") or "").strip().lower()

    # Accept canonical form
    if name.startswith("ratio_of_") and "_to_" in name:
        return []

    # Detect ad-hoc ``<A>_to_<B>_ratio`` or ``<A>_to_<B>_<noun>_ratio``
    match = _ADHOC_RATIO_RE.match(name)
    if match:
        a_part = match.group(1)
        b_part = match.group(2)
        suggested = f"ratio_of_{a_part}_to_{b_part}"
        return [
            f"audit:ratio_binary_operator_check: name '{name}' uses ad-hoc "
            f"ratio form; canonical ISN form is 'ratio_of_<A>_to_<B>'; "
            f"suggested_fix={suggested}"
        ]

    return []


def run_audits(
    candidate: dict[str, Any],
    existing_sns_in_domain: list[dict[str, Any]] | None = None,
    source_path: str | None = None,
    source_cocos_type: str | None = None,
) -> list[str]:
    """Run all audits on a candidate and return tagged issue strings.

    Each returned string has the format ``"audit:<check_name>: <detail>"``.

    Args:
        candidate: Standard name candidate dict (must include ``id``,
            ``description``, ``documentation``, ``unit`` at minimum).
        existing_sns_in_domain: Precomputed list of existing SNs in the
            same domain for synonym checking. Each dict needs ``name``,
            ``description_embedding``, ``unit``.
        source_path: The original source DD path for provenance verb check.
        source_cocos_type: COCOS transformation type from the source path.

    Returns:
        List of tagged issue strings.
    """
    all_issues: list[str] = []

    all_issues.extend(latex_def_check(candidate))
    all_issues.extend(placeholder_check(candidate))
    all_issues.extend(unit_validity_check(candidate))
    all_issues.extend(generic_noun_check(candidate))
    all_issues.extend(tautology_check(candidate))
    all_issues.extend(spectral_suffix_check(candidate))
    all_issues.extend(abbreviation_check(candidate))
    all_issues.extend(american_spelling_check(candidate))
    all_issues.extend(name_description_consistency_check(candidate))
    all_issues.extend(description_verb_drift_check(candidate))
    all_issues.extend(structural_dim_tag_check(candidate))
    all_issues.extend(provenance_verb_check(candidate, source_path))
    all_issues.extend(synonym_check(candidate, existing_sns_in_domain or []))
    all_issues.extend(unit_dimension_check(candidate))
    all_issues.extend(name_unit_consistency_check(candidate))
    all_issues.extend(multi_subject_check(candidate))
    all_issues.extend(cocos_specificity_check(candidate, source_cocos_type))
    all_issues.extend(representation_artifact_check(candidate, source_path))
    all_issues.extend(causal_due_to_check(candidate))
    all_issues.extend(implicit_field_check(candidate))
    all_issues.extend(density_unit_consistency_check(candidate))
    all_issues.extend(position_coordinate_check(candidate))
    all_issues.extend(vector_field_component_check(candidate))
    all_issues.extend(segment_order_check(candidate))
    all_issues.extend(aggregator_order_check(candidate))
    all_issues.extend(named_feature_preposition_check(candidate))
    all_issues.extend(diamagnetic_component_check(candidate))
    all_issues.extend(amplitude_of_prefix_check(candidate))
    all_issues.extend(mode_number_suffix_check(candidate))
    all_issues.extend(cumulative_prefix_check(candidate))
    all_issues.extend(pulse_schedule_reference_check(candidate, source_path))
    all_issues.extend(ratio_binary_operator_check(candidate))

    return all_issues


def has_critical_audit_failure(issues: list[str]) -> bool:
    """Return True if any issue is from a critical audit check."""
    for issue in issues:
        for check in CRITICAL_CHECKS:
            if f"audit:{check}:" in issue:
                return True
    return False
