"""
Physics Context Integration for IMAS MCP Tools

This module integrates the physics context engine with existing IMAS MCP tools
to provide physics-aware search and explanation capabilities.
"""

from typing import Dict, List, Any
from .physics_context import (
    get_physics_engine,
    PhysicsContextEngine,
    PhysicsQuantity,
    PhysicsDomain,
    concept_to_imas_paths,
    search_physics_concepts,
)


class PhysicsSearch:
    """Search that combines IMAS paths with physics concepts."""

    def __init__(self, engine: PhysicsContextEngine):
        self.engine = engine

    def search_with_physics_context(self, query: str) -> Dict[str, Any]:
        """Search IMAS data with physics context enhancement."""
        results = {
            "query": query,
            "physics_matches": [],
            "concept_suggestions": [],
            "unit_suggestions": [],
            "symbol_suggestions": [],
            "imas_path_suggestions": [],
        }

        # Search for physics concepts
        concept_matches = search_physics_concepts(query)
        for concept, quantity in concept_matches[:5]:  # Top 5 matches
            physics_match = {
                "concept": concept,
                "quantity_name": quantity.name,
                "symbol": quantity.symbol,
                "units": quantity.units,
                "description": quantity.description,
                "imas_paths": quantity.imas_paths[:3],  # Top 3 paths
                "domain": quantity.physics_domain.value,
                "relevance_score": self._calculate_relevance(query, concept, quantity),
            }
            results["physics_matches"].append(physics_match)

        # Generate suggestions based on query
        results["concept_suggestions"] = self._generate_concept_suggestions(query)
        results["unit_suggestions"] = self._generate_unit_suggestions(query)
        results["symbol_suggestions"] = self._generate_symbol_suggestions(query)
        results["imas_path_suggestions"] = self._generate_path_suggestions(query)

        return results

    def _calculate_relevance(
        self, query: str, concept: str, quantity: PhysicsQuantity
    ) -> float:
        """Calculate relevance score for a physics concept match."""
        query_lower = query.lower()
        concept_lower = concept.lower()

        # Exact match gets highest score
        if query_lower == concept_lower:
            return 1.0

        # Check if query is in concept
        if query_lower in concept_lower:
            return 0.8

        # Check alternative names
        for alt_name in quantity.alternative_names:
            if query_lower == alt_name.lower():
                return 0.9
            if query_lower in alt_name.lower():
                return 0.7

        # Check symbol match
        if query_lower == quantity.symbol.lower():
            return 0.9

        # Partial matches in description
        if query_lower in quantity.description.lower():
            return 0.5

        return 0.3

    def _generate_concept_suggestions(self, query: str) -> List[str]:
        """Generate concept suggestions based on query."""
        suggestions = []
        query_lower = query.lower()

        # Physics concept keywords
        concept_keywords = {
            "flux": ["poloidal flux", "toroidal flux", "magnetic flux"],
            "temp": ["electron temperature", "ion temperature", "temperature"],
            "density": ["electron density", "ion density", "particle density"],
            "pressure": ["electron pressure", "ion pressure", "plasma pressure"],
            "current": ["plasma current", "current density", "bootstrap current"],
            "beta": ["toroidal beta", "poloidal beta", "normalized beta"],
            "magnetic": ["safety factor", "magnetic field", "magnetic axis"],
            "energy": ["energy confinement time", "thermal energy"],
            "heating": ["NBI power", "EC power", "IC power", "auxiliary heating"],
        }

        for keyword, concepts in concept_keywords.items():
            if keyword in query_lower:
                suggestions.extend(concepts)

        return list(set(suggestions))[:5]  # Unique, top 5

    def _generate_unit_suggestions(self, query: str) -> List[Dict[str, Any]]:
        """Generate unit-based suggestions."""
        suggestions = []

        # Common physics units in fusion
        unit_mappings = {
            "eV": "energy/temperature",
            "Wb": "magnetic flux",
            "Pa": "pressure",
            "A": "current",
            "m^-3": "density",
            "T": "magnetic field",
            "W": "power",
            "s": "time",
            "m": "length/radius",
            "1": "dimensionless/normalized",
        }

        for unit, description in unit_mappings.items():
            quantities = self.engine.find_quantities_by_units(unit)
            if quantities:
                suggestions.append(
                    {
                        "unit": unit,
                        "description": description,
                        "example_quantities": [q.concept for q in quantities[:3]],
                    }
                )

        return suggestions

    def _generate_symbol_suggestions(self, query: str) -> List[Dict[str, str]]:
        """Generate symbol-based suggestions."""
        # Common physics symbols
        symbol_mappings = {
            "psi": "poloidal flux",
            "Te": "electron temperature",
            "Ti": "ion temperature",
            "ne": "electron density",
            "ni": "ion density",
            "pe": "electron pressure",
            "pi": "ion pressure",
            "Ip": "plasma current",
            "q": "safety factor",
            "beta": "plasma beta",
            "tau_E": "energy confinement time",
        }

        suggestions = []
        query_lower = query.lower()

        for symbol, concept in symbol_mappings.items():
            if query_lower in symbol.lower() or symbol.lower() in query_lower:
                suggestions.append(
                    {
                        "symbol": symbol,
                        "concept": concept,
                        "imas_paths": concept_to_imas_paths(concept)[:2],
                    }
                )

        return suggestions

    def _generate_path_suggestions(self, query: str) -> List[str]:
        """Generate IMAS path suggestions based on query."""
        # This would integrate with existing IMAS search functionality
        # For now, return common path patterns

        path_patterns = []
        query_lower = query.lower()

        if "profile" in query_lower:
            path_patterns.extend(
                [
                    "core_profiles/profiles_1d/",
                    "edge_profiles/profiles_1d/",
                    "equilibrium/time_slice/profiles_1d/",
                ]
            )

        if "electron" in query_lower:
            path_patterns.extend(
                [
                    "core_profiles/profiles_1d/electrons/",
                    "edge_profiles/profiles_1d/electrons/",
                ]
            )

        if "equilibrium" in query_lower:
            path_patterns.extend(
                [
                    "equilibrium/time_slice/",
                    "equilibrium/time_slice/global_quantities/",
                    "equilibrium/time_slice/profiles_1d/",
                ]
            )

        return path_patterns[:5]


class PhysicsConceptExplainer:
    """Explains physics concepts with IMAS context."""

    def __init__(self, engine: PhysicsContextEngine):
        self.engine = engine

    def explain_concept(
        self, concept: str, detail_level: str = "intermediate"
    ) -> Dict[str, Any]:
        """Explain a physics concept with IMAS context."""
        quantity = self.engine.find_quantity_by_concept(concept)

        if not quantity:
            return {
                "error": f"Physics concept '{concept}' not found",
                "suggestions": [c for c, _ in search_physics_concepts(concept)[:5]],
            }

        explanation = {
            "concept": concept,
            "quantity": {
                "name": quantity.name,
                "symbol": quantity.symbol,
                "units": quantity.units,
                "description": quantity.description,
                "domain": quantity.physics_domain.value,
                "coordinate_type": quantity.coordinate_type,
            },
            "imas_integration": {
                "primary_paths": quantity.imas_paths[:5],
                "total_paths": len(quantity.imas_paths),
                "coordinate_type": quantity.coordinate_type,
            },
            "alternative_names": quantity.alternative_names,
            "typical_ranges": quantity.typical_ranges or {},
            "physics_context": self._get_physics_context(quantity, detail_level),
            "related_quantities": [
                {
                    "name": rq.name,
                    "symbol": rq.symbol,
                    "units": rq.units,
                    "concept": rq.concept,
                }
                for rq in self.engine.get_related_quantities(concept)
            ],
        }

        return explanation

    def _get_physics_context(
        self, quantity: PhysicsQuantity, detail_level: str
    ) -> Dict[str, Any]:
        """Get physics context for a quantity based on detail level."""
        context = {
            "domain_description": self._get_domain_description(quantity.physics_domain),
            "measurement_context": self._get_measurement_context(quantity),
            "significance": self._get_physics_significance(quantity),
        }

        if detail_level in ["advanced", "detailed"]:
            advanced_context = {
                "mathematical_definition": self._get_mathematical_definition(quantity),
                "calculation_methods": self._get_calculation_methods(quantity),
                "typical_profiles": self._get_typical_profiles(quantity),
            }
            context.update(advanced_context)

        return context

    def _get_domain_description(self, domain: PhysicsDomain) -> str:
        """Get description of physics domain."""
        descriptions = {
            PhysicsDomain.EQUILIBRIUM: "Magnetohydrodynamic equilibrium describes the force balance in plasma",
            PhysicsDomain.TRANSPORT: "Transport phenomena govern particle, momentum and energy transport",
            PhysicsDomain.HEATING: "Auxiliary heating systems for plasma heating and current drive",
            PhysicsDomain.CONFINEMENT: "Confinement properties determine plasma performance",
            PhysicsDomain.INSTABILITIES: "Plasma instabilities and disruptions",
            PhysicsDomain.GEOMETRY: "Geometric coordinates and plasma shape",
            PhysicsDomain.PROFILES: "Radial profiles of plasma parameters",
            PhysicsDomain.RADIATION: "Radiation losses and spectroscopy",
            PhysicsDomain.DIAGNOSTICS: "Diagnostic measurements and data",
            PhysicsDomain.CONTROL: "Plasma control systems and actuators",
        }
        return descriptions.get(domain, "Physics domain description not available")

    def _get_measurement_context(self, quantity: PhysicsQuantity) -> str:
        """Get measurement context for a quantity."""
        measurement_info = {
            "electron_temperature": "Measured by Thomson scattering, ECE, or charge exchange",
            "electron_density": "Measured by interferometry, Thomson scattering, or reflectometry",
            "poloidal_flux": "Calculated from magnetic measurements and equilibrium reconstruction",
            "safety_factor": "Derived from equilibrium reconstruction or MSE diagnostics",
            "plasma_current": "Measured by Rogowski coils or magnetic loops",
            "toroidal_beta": "Calculated from pressure profiles and magnetic field",
        }

        return measurement_info.get(
            quantity.name, "Measurement method depends on specific diagnostic setup"
        )

    def _get_physics_significance(self, quantity: PhysicsQuantity) -> str:
        """Get physics significance of a quantity."""
        significance = {
            "electron_temperature": "Determines plasma thermal content and transport rates",
            "electron_density": "Controls plasma collisionality and fusion reaction rate",
            "poloidal_flux": "Defines magnetic flux surfaces and plasma confinement geometry",
            "safety_factor": "Determines MHD stability and confinement quality",
            "plasma_current": "Drives magnetic field and determines plasma stability",
            "toroidal_beta": "Measures plasma pressure relative to magnetic pressure",
        }

        return significance.get(
            quantity.name, "Important parameter for plasma physics analysis"
        )

    def _get_mathematical_definition(self, quantity: PhysicsQuantity) -> str:
        """Get mathematical definition for advanced detail level."""
        definitions = {
            "poloidal_flux": "ψ = ∫∫ B⃗ · dA⃗ over poloidal cross-section",
            "safety_factor": "q = (dΦ/dψ)/(2π) where Φ is toroidal flux",
            "toroidal_beta": "βₜ = 2μ₀⟨p⟩/B₀² where ⟨p⟩ is volume-averaged pressure",
            "electron_pressure": "pₑ = nₑTₑ (ideal gas law for electrons)",
        }

        return definitions.get(quantity.name, "Mathematical definition not available")

    def _get_calculation_methods(self, quantity: PhysicsQuantity) -> List[str]:
        """Get calculation methods for advanced detail level."""
        methods = {
            "poloidal_flux": [
                "Grad-Shafranov equation solution",
                "Magnetic reconstruction",
            ],
            "safety_factor": [
                "Equilibrium reconstruction",
                "Direct measurement via MSE",
            ],
            "toroidal_beta": [
                "Pressure profile integration",
                "Diamagnetic measurement",
            ],
        }

        return methods.get(quantity.name, ["Standard plasma physics calculations"])

    def _get_typical_profiles(self, quantity: PhysicsQuantity) -> Dict[str, str]:
        """Get typical profile shapes for advanced detail level."""
        profiles = {
            "electron_temperature": {
                "shape": "Peaked on-axis",
                "typical": "10-30 keV central, 0.1-1 keV edge",
            },
            "electron_density": {
                "shape": "Peaked or flat",
                "typical": "1-5×10²⁰ m⁻³ central",
            },
            "safety_factor": {
                "shape": "Monotonic increasing",
                "typical": "q₀=0.8-1.2, q₉₅=3-8",
            },
        }

        return profiles.get(
            quantity.name, {"shape": "Variable", "typical": "Depends on scenario"}
        )


class IMASPhysicsIntegration:
    """Main integration class for physics-enhanced IMAS tools."""

    def __init__(self):
        self.engine = get_physics_engine()
        self.searcher = PhysicsSearch(self.engine)
        self.explainer = PhysicsConceptExplainer(self.engine)

    def search(self, query: str) -> Dict[str, Any]:
        """Perform physics search."""
        return self.searcher.search_with_physics_context(query)

    def explain_physics_concept(
        self, concept: str, detail_level: str = "intermediate"
    ) -> Dict[str, Any]:
        """Explain a physics concept with IMAS integration."""
        return self.explainer.explain_concept(concept, detail_level)

    def concept_to_imas_mapping(self, concept: str) -> Dict[str, Any]:
        """Get complete concept to IMAS mapping."""
        quantity = self.engine.find_quantity_by_concept(concept)

        if not quantity:
            return {"error": f"Concept '{concept}' not found"}

        return {
            "concept": concept,
            "symbol": quantity.symbol,
            "units": quantity.units,
            "imas_paths": quantity.imas_paths,
            "description": quantity.description,
            "domain": quantity.physics_domain.value,
            "alternative_names": quantity.alternative_names,
            "usage_examples": self._generate_usage_examples(quantity),
        }

    def _generate_usage_examples(
        self, quantity: PhysicsQuantity
    ) -> List[Dict[str, str]]:
        """Generate usage examples for IMAS paths."""
        examples = []

        if quantity.imas_paths:
            primary_path = quantity.imas_paths[0]

            examples.append(
                {
                    "scenario": "Python access",
                    "code": f"# Access {quantity.concept}\nvalue = ids.{primary_path}",
                    "description": f"Get {quantity.concept} from IMAS {primary_path.split('/')[0]} IDS",
                }
            )

            if quantity.coordinate_type == "flux":
                examples.append(
                    {
                        "scenario": "Profile access",
                        "code": f"# Access radial profile\nprofile = ids.{primary_path}[time_idx, :]",
                        "description": f"Get radial profile of {quantity.concept} at specific time",
                    }
                )

            if quantity.coordinate_type == "time":
                examples.append(
                    {
                        "scenario": "Time evolution",
                        "code": f"# Access time evolution\ntime_trace = ids.{primary_path}[:]",
                        "description": f"Get time evolution of {quantity.concept}",
                    }
                )

        return examples

    def get_domain_overview(self, domain: PhysicsDomain) -> Dict[str, Any]:
        """Get overview of a physics domain."""
        quantities = self.engine.find_quantities_by_domain(domain)

        return {
            "domain": domain.value,
            "description": self.explainer._get_domain_description(domain),
            "quantity_count": len(quantities),
            "key_quantities": [
                {
                    "concept": q.concept,
                    "symbol": q.symbol,
                    "units": q.units,
                    "path_count": len(q.imas_paths),
                }
                for q in quantities
            ],
            "common_units": list(set(q.units for q in quantities)),
            "imas_coverage": {
                "total_paths": sum(len(q.imas_paths) for q in quantities),
                "unique_ids": len(
                    set(path.split("/")[0] for q in quantities for path in q.imas_paths)
                ),
            },
        }

    def validate_physics_query(self, query: str) -> Dict[str, Any]:
        """Validate and suggest improvements for physics queries."""
        validation = {
            "query": query,
            "is_valid_physics_concept": False,
            "suggestions": [],
            "corrections": [],
            "alternative_queries": [],
        }

        # Check if query matches known concepts
        concept_matches = search_physics_concepts(query)
        if concept_matches:
            validation["is_valid_physics_concept"] = True
            validation["best_match"] = {
                "concept": concept_matches[0][0],
                "quantity": concept_matches[0][1].name,
                "confidence": "high"
                if concept_matches[0][0].lower() == query.lower()
                else "medium",
            }

        # Generate suggestions
        if not concept_matches:
            validation["suggestions"] = self.searcher._generate_concept_suggestions(
                query
            )
            validation["alternative_queries"] = self._suggest_alternative_queries(query)

        return validation

    def _suggest_alternative_queries(self, query: str) -> List[str]:
        """Suggest alternative queries for unsuccessful searches."""
        suggestions = []
        query_lower = query.lower()

        # Common misspellings and alternatives
        alternatives = {
            "temp": "temperature",
            "dens": "density",
            "curr": "current",
            "mag": "magnetic",
            "elect": "electron",
            "pol": "poloidal",
            "tor": "toroidal",
            "norm": "normalized",
        }

        for abbrev, full in alternatives.items():
            if abbrev in query_lower:
                suggestions.append(query_lower.replace(abbrev, full))

        return suggestions[:3]


# Create global integration instance
_physics_integration = None


def get_physics_integration() -> IMASPhysicsIntegration:
    """Get global physics integration instance."""
    global _physics_integration
    if _physics_integration is None:
        _physics_integration = IMASPhysicsIntegration()
    return _physics_integration


# Convenience functions for easy access
def physics_search(query: str) -> Dict[str, Any]:
    """Perform physics search."""
    integration = get_physics_integration()
    return integration.search(query)


def explain_physics_concept(
    concept: str, detail_level: str = "intermediate"
) -> Dict[str, Any]:
    """Explain physics concept with IMAS integration."""
    integration = get_physics_integration()
    return integration.explain_physics_concept(concept, detail_level)


def get_concept_imas_mapping(concept: str) -> Dict[str, Any]:
    """Get complete concept to IMAS mapping."""
    integration = get_physics_integration()
    return integration.concept_to_imas_mapping(concept)


if __name__ == "__main__":
    # Example usage and testing
    integration = get_physics_integration()

    print("=== Physics Integration Testing ===")

    # Test search
    print("\n1. Search for 'poloidal flux':")
    search_result = physics_search("poloidal flux")
    print(f"   Found {len(search_result['physics_matches'])} physics matches")
    if search_result["physics_matches"]:
        match = search_result["physics_matches"][0]
        print(
            f"   Best match: {match['concept']} -> {match['symbol']} [{match['units']}]"
        )

    # Test concept explanation
    print("\n2. Explain 'electron temperature':")
    explanation = explain_physics_concept("electron temperature")
    if "error" not in explanation:
        print(f"   Symbol: {explanation['quantity']['symbol']}")
        print(f"   Units: {explanation['quantity']['units']}")
        print(f"   IMAS paths: {len(explanation['imas_integration']['primary_paths'])}")

    # Test concept mapping
    print("\n3. Concept to IMAS mapping for 'safety factor':")
    mapping = get_concept_imas_mapping("safety factor")
    if "error" not in mapping:
        print(f"   Symbol: {mapping['symbol']}")
        print(f"   Units: {mapping['units']}")
        print(f"   Paths: {mapping['imas_paths']}")

    # Test domain overview
    print("\n4. Equilibrium domain overview:")
    domain_overview = integration.get_domain_overview(PhysicsDomain.EQUILIBRIUM)
    print(f"   Quantities: {domain_overview['quantity_count']}")
    print(f"   Total IMAS paths: {domain_overview['imas_coverage']['total_paths']}")
    print(f"   Unique IDS: {domain_overview['imas_coverage']['unique_ids']}")
