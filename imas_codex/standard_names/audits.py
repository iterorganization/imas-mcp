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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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
    }
)

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
    """
    issues: list[str] = []
    unit = (candidate.get("unit") or "").lower()
    if not unit or unit in ("1", "dimensionless", "-", "mixed", "none"):
        return issues

    # Split on unit algebra operators and check each token
    tokens = re.split(r"[\s*/.^()·×]+", unit)
    for tok in tokens:
        if not tok:
            continue
        if tok in _INVALID_UNIT_TOKENS:
            issues.append(
                f"audit:unit_validity_check: unit '{candidate.get('unit')}' "
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

    # Heuristic: check if name contains two subject enum values
    try:
        from imas_standard_names.grammar import Subject

        name_tokens = set(name.split("_"))
        matched_subjects = [s.value for s in Subject if s.value in name_tokens]
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
    issues: list[str] = []
    for abbrev, full in _FORBIDDEN_ABBREVIATIONS:
        # Boundary-sensitive match: leading or interior token only.
        if name.startswith(abbrev) or abbrev in f"_{name}_":
            issues.append(
                f"audit:abbreviation_check: name '{name}' contains "
                f"abbreviation '{abbrev.strip('_')}'; spell as '{full.strip('_')}'"
            )
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
    all_issues.extend(multi_subject_check(candidate))
    all_issues.extend(cocos_specificity_check(candidate, source_cocos_type))

    return all_issues


def has_critical_audit_failure(issues: list[str]) -> bool:
    """Return True if any issue is from a critical audit check."""
    for issue in issues:
        for check in CRITICAL_CHECKS:
            if f"audit:{check}:" in issue:
                return True
    return False
