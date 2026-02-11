"""Physical unit and convention extraction from text content.

Extracted from discovery/wiki/scraper.py to be shared across all ingestion
pipelines. Detects physical units (eV, Tesla, etc.) and sign/coordinate
conventions (COCOS, positive clockwise, etc.) in any text.
"""

import re

# Physical units (common in fusion data)
UNIT_PATTERN = re.compile(
    r"\b(eV|keV|MeV|GeV|"  # Energy
    r"m\^?-?[123]|cm\^?-?[123]|mm|"  # Length/density
    r"Tesla|T|Wb|Weber|"  # Magnetic
    r"Amp(?:ere)?s?|A|MA|kA|"  # Current
    r"Ohm|ohm|Ω|"  # Resistance
    r"Volt|V|kV|"  # Voltage
    r"Watt|W|MW|kW|"  # Power
    r"m/s|m\.s\^-1|"  # Velocity
    r"rad|radian|degrees?|°|"  # Angle
    r"seconds?|s|ms|μs|ns|"  # Time
    r"Hz|kHz|MHz|GHz|"  # Frequency
    r"Pa|kPa|MPa|bar|mbar|Torr)\b",
    re.IGNORECASE,
)

# COCOS convention patterns
COCOS_PATTERN = re.compile(
    r"COCOS\s*(\d{1,2})|"  # COCOS 11, COCOS 3
    r"cocos\s*=\s*(\d{1,2})|"  # cocos=11
    r"coordinate\s+convention[s]?\s+(\d{1,2})",
    re.IGNORECASE,
)

# Sign convention patterns
SIGN_CONVENTION_PATTERN = re.compile(
    r"(positive|negative)\s+(clockwise|counter-?clockwise|outward|inward|"
    r"upward|downward|toroidal|poloidal|radial)|"
    r"(sign\s+convention)|"
    r"(I_?[pP]\s*[><]=?\s*0)|"  # Ip > 0
    r"(B_?[tTφ]\s*[><]=?\s*0)",  # Bt > 0
    re.IGNORECASE,
)


def extract_units(text: str) -> list[str]:
    """Extract physical units from text.

    Args:
        text: Raw text content

    Returns:
        Deduplicated list of unit symbols found
    """
    matches = UNIT_PATTERN.findall(text)
    return sorted(set(matches))


def extract_conventions(text: str) -> list[dict[str, str]]:
    """Extract sign and coordinate conventions from text.

    Args:
        text: Raw text content

    Returns:
        List of convention dicts with type, name, and context
    """
    conventions = []

    for match in COCOS_PATTERN.finditer(text):
        cocos_num = match.group(1) or match.group(2) or match.group(3)
        if cocos_num:
            conventions.append(
                {
                    "type": "cocos",
                    "name": f"COCOS {cocos_num}",
                    "cocos_index": int(cocos_num),
                    "context": text[max(0, match.start() - 50) : match.end() + 50],
                }
            )

    for match in SIGN_CONVENTION_PATTERN.finditer(text):
        matched_text = match.group(0)
        conventions.append(
            {
                "type": "sign",
                "name": matched_text.strip(),
                "context": text[max(0, match.start() - 50) : match.end() + 50],
            }
        )

    return conventions


__all__ = [
    "COCOS_PATTERN",
    "SIGN_CONVENTION_PATTERN",
    "UNIT_PATTERN",
    "extract_conventions",
    "extract_units",
]
