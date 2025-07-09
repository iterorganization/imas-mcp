"""
Physics Context Module for IMAS MCP Tools

This module provides a comprehensive mapping between physics concepts and IMAS attributes.
It enables semantic understanding of physics quantities and their relationships within
the IMAS data dictionary.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class PhysicsDomain(Enum):
    """Physics domains in fusion plasma physics."""

    EQUILIBRIUM = "equilibrium"
    TRANSPORT = "transport"
    HEATING = "heating"
    CONFINEMENT = "confinement"
    INSTABILITIES = "instabilities"
    GEOMETRY = "geometry"
    PROFILES = "profiles"
    RADIATION = "radiation"
    DIAGNOSTICS = "diagnostics"
    CONTROL = "control"


@dataclass
class PhysicsQuantity:
    """Represents a physics quantity with its IMAS attributes."""

    name: str
    concept: str
    description: str
    units: str
    symbol: str
    imas_paths: List[str] = field(default_factory=list)
    alternative_names: List[str] = field(default_factory=list)
    related_quantities: List[str] = field(default_factory=list)
    physics_domain: PhysicsDomain = PhysicsDomain.PROFILES
    coordinate_type: Optional[str] = None
    typical_ranges: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Validate and normalize the quantity."""
        if not self.symbol:
            self.symbol = self.name.lower()


@dataclass
class PhysicsContext:
    """Container for physics context information."""

    quantities: Dict[str, PhysicsQuantity] = field(default_factory=dict)
    concept_to_quantity: Dict[str, str] = field(default_factory=dict)
    units_to_quantities: Dict[str, List[str]] = field(default_factory=dict)
    domain_to_quantities: Dict[PhysicsDomain, List[str]] = field(default_factory=dict)

    def add_quantity(self, quantity: PhysicsQuantity):
        """Add a physics quantity to the context."""
        self.quantities[quantity.name] = quantity

        # Index by concept
        self.concept_to_quantity[quantity.concept.lower()] = quantity.name

        # Index by units
        if quantity.units not in self.units_to_quantities:
            self.units_to_quantities[quantity.units] = []
        self.units_to_quantities[quantity.units].append(quantity.name)

        # Index by domain
        if quantity.physics_domain not in self.domain_to_quantities:
            self.domain_to_quantities[quantity.physics_domain] = []
        self.domain_to_quantities[quantity.physics_domain].append(quantity.name)

        # Index alternative names
        for alt_name in quantity.alternative_names:
            self.concept_to_quantity[alt_name.lower()] = quantity.name


class PhysicsContextBuilder:
    """Builder class for creating physics context from IMAS data dictionary."""

    def __init__(self):
        self.context = PhysicsContext()
        self._build_comprehensive_context()

    def _build_comprehensive_context(self):
        """Build comprehensive physics context based on IMAS data dictionary."""

        # Define core physics quantities based on IMAS search results
        physics_quantities = [
            # === MAGNETIC QUANTITIES ===
            PhysicsQuantity(
                name="poloidal_flux",
                concept="poloidal flux",
                description="Poloidal magnetic flux. Integral of magnetic field passing through a contour defined by the intersection of a flux surface passing through the point of interest and a Z=constant plane.",
                units="Wb",
                symbol="psi",
                imas_paths=[
                    "equilibrium/time_slice/profiles_1d/psi",
                    "equilibrium/time_slice/profiles_2d/psi",
                    "core_profiles/profiles_1d/grid/psi",
                    "edge_profiles/profiles_1d/grid/psi",
                    "plasma_profiles/profiles_1d/grid/psi",
                    "core_transport/model/profiles_1d/grid_flux/psi",
                    "distributions/distribution/profiles_1d/grid/psi",
                ],
                alternative_names=["psi", "magnetic flux", "flux function"],
                physics_domain=PhysicsDomain.EQUILIBRIUM,
                coordinate_type="flux",
                typical_ranges={"tokamak": "0.1-10 Wb"},
            ),
            PhysicsQuantity(
                name="safety_factor",
                concept="safety factor",
                description="Safety factor q - ratio of toroidal to poloidal magnetic field components multiplied by geometric factors",
                units="1",
                symbol="q",
                imas_paths=[
                    "equilibrium/time_slice/profiles_1d/q",
                    "equilibrium/time_slice/global_quantities/q_min/value",
                    "equilibrium/time_slice/global_quantities/q_95",
                ],
                alternative_names=["q", "q factor", "magnetic safety factor"],
                physics_domain=PhysicsDomain.EQUILIBRIUM,
                coordinate_type="flux",
                typical_ranges={"tokamak": "0.8-8"},
            ),
            PhysicsQuantity(
                name="normalized_toroidal_flux",
                concept="normalized toroidal flux coordinate",
                description="Normalized toroidal flux coordinate. The normalizing value is the toroidal flux coordinate at the equilibrium boundary (LCFS).",
                units="1",
                symbol="rho_tor_norm",
                imas_paths=[
                    "equilibrium/time_slice/profiles_1d/rho_tor_norm",
                    "core_profiles/profiles_1d/grid/rho_tor_norm",
                    "edge_profiles/profiles_1d/grid/rho_tor_norm",
                    "plasma_profiles/profiles_1d/grid/rho_tor_norm",
                    "distributions/distribution/profiles_1d/grid/rho_tor_norm",
                ],
                alternative_names=[
                    "rho_tor_norm",
                    "normalized rho toroidal",
                    "flux coordinate",
                ],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="flux",
                typical_ranges={"tokamak": "0.0-1.0"},
            ),
            # === PLASMA PARAMETERS ===
            PhysicsQuantity(
                name="electron_temperature",
                concept="electron temperature",
                description="Electron temperature in the plasma",
                units="eV",
                symbol="Te",
                imas_paths=[
                    "core_profiles/profiles_1d/electrons/temperature",
                    "edge_profiles/profiles_1d/electrons/temperature",
                    "core_instant_changes/change/profiles_1d/electrons/temperature",
                ],
                alternative_names=["Te", "electron temp", "temperature electron"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "10-30000 eV"},
            ),
            PhysicsQuantity(
                name="ion_temperature",
                concept="ion temperature",
                description="Ion temperature in the plasma (average over charge states when multiple charge states are considered)",
                units="eV",
                symbol="Ti",
                imas_paths=[
                    "core_profiles/profiles_1d/ion/temperature",
                    "core_profiles/profiles_1d/ion/state/temperature",
                    "edge_profiles/profiles_1d/ion/temperature",
                    "core_instant_changes/change/profiles_1d/ion/temperature",
                ],
                alternative_names=["Ti", "ion temp", "temperature ion"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "10-30000 eV"},
            ),
            PhysicsQuantity(
                name="electron_density",
                concept="electron density",
                description="Electron density in the plasma (thermal + non-thermal)",
                units="m^-3",
                symbol="ne",
                imas_paths=[
                    "core_profiles/profiles_1d/electrons/density",
                    "edge_profiles/profiles_1d/electrons/density",
                    "core_instant_changes/change/profiles_1d/electrons/density",
                ],
                alternative_names=["ne", "electron density", "density electron"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e18-1e21 m^-3"},
            ),
            PhysicsQuantity(
                name="ion_density",
                concept="ion density",
                description="Ion density in the plasma (thermal + non-thermal, sum over charge states)",
                units="m^-3",
                symbol="ni",
                imas_paths=[
                    "core_profiles/profiles_1d/ion/density",
                    "core_profiles/profiles_1d/ion/state/density",
                    "edge_profiles/profiles_1d/ion/density",
                    "core_instant_changes/change/profiles_1d/ion/density",
                ],
                alternative_names=["ni", "ion density", "density ion"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e18-1e21 m^-3"},
            ),
            PhysicsQuantity(
                name="electron_pressure",
                concept="electron pressure",
                description="Electron pressure (thermal + non-thermal)",
                units="Pa",
                symbol="pe",
                imas_paths=[
                    "core_profiles/profiles_1d/electrons/pressure",
                    "edge_profiles/profiles_1d/electrons/pressure",
                    "core_instant_changes/change/profiles_1d/electrons/pressure",
                ],
                alternative_names=["pe", "electron pressure", "pressure electron"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e3-1e6 Pa"},
            ),
            PhysicsQuantity(
                name="ion_pressure",
                concept="ion pressure",
                description="Ion pressure (thermal + non-thermal, sum over charge states)",
                units="Pa",
                symbol="pi",
                imas_paths=[
                    "core_profiles/profiles_1d/ion/pressure",
                    "core_profiles/profiles_1d/ion/state/pressure",
                    "edge_profiles/profiles_1d/ion/pressure",
                    "core_instant_changes/change/profiles_1d/ion/pressure",
                ],
                alternative_names=["pi", "ion pressure", "pressure ion"],
                physics_domain=PhysicsDomain.PROFILES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e3-1e6 Pa"},
            ),
            # === CURRENTS ===
            PhysicsQuantity(
                name="plasma_current",
                concept="plasma current",
                description="Total plasma current",
                units="A",
                symbol="Ip",
                imas_paths=[
                    "summary/global_quantities/ip",
                    "equilibrium/time_slice/global_quantities/ip",
                    "magnetics/ip",
                ],
                alternative_names=["Ip", "plasma current", "toroidal current"],
                physics_domain=PhysicsDomain.EQUILIBRIUM,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.1-20 MA"},
            ),
            PhysicsQuantity(
                name="current_density",
                concept="current density",
                description="Current density profiles in the plasma",
                units="A.m^-2",
                symbol="j",
                imas_paths=[
                    "equilibrium/time_slice/profiles_1d/j_tor",
                    "equilibrium/time_slice/profiles_1d/j_non_inductive",
                    "equilibrium/time_slice/profiles_1d/j_bootstrap",
                ],
                alternative_names=["j", "current density", "toroidal current density"],
                physics_domain=PhysicsDomain.EQUILIBRIUM,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e4-1e7 A/m^2"},
            ),
            # === BETA PARAMETERS ===
            PhysicsQuantity(
                name="toroidal_beta",
                concept="toroidal beta",
                description="Toroidal beta - ratio of plasma pressure to magnetic pressure",
                units="1",
                symbol="beta_tor",
                imas_paths=[
                    "summary/global_quantities/beta_tor",
                    "core_profiles/global_quantities/beta_tor",
                ],
                alternative_names=["beta_tor", "toroidal beta", "beta toroidal"],
                physics_domain=PhysicsDomain.CONFINEMENT,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.01-0.1"},
            ),
            PhysicsQuantity(
                name="poloidal_beta",
                concept="poloidal beta",
                description="Poloidal beta - normalized pressure relative to poloidal magnetic field",
                units="1",
                symbol="beta_pol",
                imas_paths=[
                    "summary/global_quantities/beta_pol",
                    "summary/global_quantities/beta_pol_mhd",
                ],
                alternative_names=["beta_pol", "poloidal beta", "beta poloidal"],
                physics_domain=PhysicsDomain.CONFINEMENT,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.5-3.0"},
            ),
            PhysicsQuantity(
                name="normalized_beta",
                concept="normalized beta",
                description="Normalized beta - toroidal beta normalized by engineering parameters",
                units="1",
                symbol="beta_N",
                imas_paths=[
                    "summary/global_quantities/beta_tor_norm",
                    "core_profiles/global_quantities/beta_tor_norm",
                    "pulse_schedule/flux_control/beta_tor_norm",
                ],
                alternative_names=["beta_N", "normalized beta", "beta normalized"],
                physics_domain=PhysicsDomain.CONFINEMENT,
                coordinate_type="global",
                typical_ranges={"tokamak": "1.0-4.0"},
            ),
            # === GEOMETRY ===
            PhysicsQuantity(
                name="major_radius",
                concept="major radius",
                description="Major radius of the plasma",
                units="m",
                symbol="R",
                imas_paths=[
                    "equilibrium/time_slice/global_quantities/magnetic_axis/r",
                    "equilibrium/time_slice/profiles_1d/r_outboard",
                    "equilibrium/time_slice/profiles_2d/r",
                ],
                alternative_names=["R", "major radius", "radius major"],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="spatial",
                typical_ranges={"tokamak": "0.5-10 m"},
            ),
            PhysicsQuantity(
                name="minor_radius",
                concept="minor radius",
                description="Minor radius of the plasma",
                units="m",
                symbol="a",
                imas_paths=[
                    "equilibrium/time_slice/global_quantities/a_minor",
                    "summary/global_quantities/a_minor",
                ],
                alternative_names=["a", "minor radius", "radius minor"],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="spatial",
                typical_ranges={"tokamak": "0.1-3 m"},
            ),
            PhysicsQuantity(
                name="vertical_position",
                concept="vertical position",
                description="Vertical position coordinate (Z)",
                units="m",
                symbol="Z",
                imas_paths=[
                    "equilibrium/time_slice/global_quantities/magnetic_axis/z",
                    "equilibrium/time_slice/profiles_2d/z",
                ],
                alternative_names=["Z", "vertical position", "height"],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="spatial",
                typical_ranges={"tokamak": "-5 to 5 m"},
            ),
            # === CONFINEMENT ===
            PhysicsQuantity(
                name="energy_confinement_time",
                concept="energy confinement time",
                description="Global energy confinement time",
                units="s",
                symbol="tau_E",
                imas_paths=[
                    "summary/global_quantities/tau_energy",
                    "core_profiles/global_quantities/tau_energy",
                ],
                alternative_names=["tau_E", "confinement time", "energy confinement"],
                physics_domain=PhysicsDomain.CONFINEMENT,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.01-5 s"},
            ),
            # === HEATING ===
            PhysicsQuantity(
                name="nbi_power",
                concept="neutral beam injection power",
                description="Power delivered by neutral beam injection",
                units="W",
                symbol="P_NBI",
                imas_paths=[
                    "summary/heating_current_drive/nbi/power_launched",
                    "nbi/unit/power_launched",
                ],
                alternative_names=["P_NBI", "NBI power", "neutral beam power"],
                physics_domain=PhysicsDomain.HEATING,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.1-50 MW"},
            ),
            PhysicsQuantity(
                name="ec_power",
                concept="electron cyclotron power",
                description="Power delivered by electron cyclotron heating",
                units="W",
                symbol="P_EC",
                imas_paths=[
                    "summary/heating_current_drive/ec/power_launched",
                    "ec_launchers/launcher/power_launched",
                ],
                alternative_names=["P_EC", "EC power", "electron cyclotron power"],
                physics_domain=PhysicsDomain.HEATING,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.1-10 MW"},
            ),
            PhysicsQuantity(
                name="ic_power",
                concept="ion cyclotron power",
                description="Power delivered by ion cyclotron heating",
                units="W",
                symbol="P_IC",
                imas_paths=[
                    "summary/heating_current_drive/ic/power_launched",
                    "ic_antennas/antenna/power_launched",
                ],
                alternative_names=["P_IC", "IC power", "ion cyclotron power"],
                physics_domain=PhysicsDomain.HEATING,
                coordinate_type="global",
                typical_ranges={"tokamak": "0.1-20 MW"},
            ),
            # === RUNAWAY ELECTRONS ===
            PhysicsQuantity(
                name="runaway_density",
                concept="runaway electron density",
                description="Density of runaway electrons",
                units="m^-3",
                symbol="n_re",
                imas_paths=[
                    "runaway_electrons/profiles_1d/density",
                    "runaway_electrons/ggd_fluid/density",
                ],
                alternative_names=["n_re", "runaway density", "RE density"],
                physics_domain=PhysicsDomain.INSTABILITIES,
                coordinate_type="flux",
                typical_ranges={"tokamak": "1e15-1e19 m^-3"},
            ),
            # === DIAGNOSTICS ===
            PhysicsQuantity(
                name="line_integrated_density",
                concept="line integrated density",
                description="Line integrated electron density measured by interferometry",
                units="m^-2",
                symbol="n_e_line",
                imas_paths=[
                    "interferometer/channel/n_e_line",
                    "interferometer/channel/n_e_line_average",
                ],
                alternative_names=["n_e_line", "line density", "interferometry"],
                physics_domain=PhysicsDomain.DIAGNOSTICS,
                coordinate_type="line_of_sight",
                typical_ranges={"tokamak": "1e18-1e21 m^-2"},
            ),
            # === ADDITIONAL COORDINATES ===
            PhysicsQuantity(
                name="normalized_poloidal_flux",
                concept="normalized poloidal flux",
                description="Normalized poloidal flux coordinate",
                units="1",
                symbol="psi_norm",
                imas_paths=[
                    "equilibrium/time_slice/profiles_1d/psi_norm",
                    "core_profiles/profiles_1d/grid/psi_norm",
                ],
                alternative_names=["psi_norm", "normalized psi", "flux coordinate"],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="flux",
                typical_ranges={"tokamak": "0.0-1.0"},
            ),
            PhysicsQuantity(
                name="time",
                concept="time",
                description="Time coordinate",
                units="s",
                symbol="t",
                imas_paths=[
                    "equilibrium/time",
                    "core_profiles/time",
                    "magnetics/time",
                ],
                alternative_names=["t", "time", "temporal"],
                physics_domain=PhysicsDomain.GEOMETRY,
                coordinate_type="time",
                typical_ranges={"tokamak": "0-1000 s"},
            ),
        ]

        # Add all quantities to context
        for quantity in physics_quantities:
            self.context.add_quantity(quantity)

    def get_context(self) -> PhysicsContext:
        """Get the built physics context."""
        return self.context


class PhysicsContextEngine:
    """Engine for physics context queries and mappings."""

    def __init__(self, context: PhysicsContext):
        self.context = context

    def find_quantity_by_concept(self, concept: str) -> Optional[PhysicsQuantity]:
        """Find a physics quantity by concept name."""
        concept_lower = concept.lower()

        # Direct lookup
        if concept_lower in self.context.concept_to_quantity:
            quantity_name = self.context.concept_to_quantity[concept_lower]
            return self.context.quantities[quantity_name]

        # Fuzzy matching
        for mapped_concept, quantity_name in self.context.concept_to_quantity.items():
            if concept_lower in mapped_concept or mapped_concept in concept_lower:
                return self.context.quantities[quantity_name]

        return None

    def find_quantities_by_units(self, units: str) -> List[PhysicsQuantity]:
        """Find all physics quantities with given units."""
        if units in self.context.units_to_quantities:
            quantity_names = self.context.units_to_quantities[units]
            return [self.context.quantities[name] for name in quantity_names]
        return []

    def find_quantities_by_domain(self, domain: PhysicsDomain) -> List[PhysicsQuantity]:
        """Find all physics quantities in a given domain."""
        if domain in self.context.domain_to_quantities:
            quantity_names = self.context.domain_to_quantities[domain]
            return [self.context.quantities[name] for name in quantity_names]
        return []

    def get_imas_paths_for_concept(self, concept: str) -> List[str]:
        """Get IMAS paths for a physics concept."""
        quantity = self.find_quantity_by_concept(concept)
        if quantity:
            return quantity.imas_paths
        return []

    def get_units_for_concept(self, concept: str) -> Optional[str]:
        """Get units for a physics concept."""
        quantity = self.find_quantity_by_concept(concept)
        if quantity:
            return quantity.units
        return None

    def get_symbol_for_concept(self, concept: str) -> Optional[str]:
        """Get symbol for a physics concept."""
        quantity = self.find_quantity_by_concept(concept)
        if quantity:
            return quantity.symbol
        return None

    def search_concepts(self, query: str) -> List[Tuple[str, PhysicsQuantity]]:
        """Search for physics concepts matching query."""
        results = []
        query_lower = query.lower()

        for concept, quantity_name in self.context.concept_to_quantity.items():
            if query_lower in concept:
                quantity = self.context.quantities[quantity_name]
                results.append((concept, quantity))

        # Sort by relevance (exact matches first)
        results.sort(key=lambda x: (0 if x[0] == query_lower else 1, len(x[0])))

        return results

    def get_related_quantities(self, concept: str) -> List[PhysicsQuantity]:
        """Get related physics quantities for a concept."""
        quantity = self.find_quantity_by_concept(concept)
        if not quantity:
            return []

        related = []
        for related_name in quantity.related_quantities:
            if related_name in self.context.quantities:
                related.append(self.context.quantities[related_name])

        return related

    def get_all_concepts(self) -> List[str]:
        """Get all physics concepts."""
        return list(self.context.concept_to_quantity.keys())

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the physics context."""
        return {
            "total_quantities": len(self.context.quantities),
            "total_concepts": len(self.context.concept_to_quantity),
            "unique_units": len(self.context.units_to_quantities),
            "physics_domains": len(self.context.domain_to_quantities),
            "quantities_by_domain": {
                domain.value: len(quantities)
                for domain, quantities in self.context.domain_to_quantities.items()
            },
        }


# Global physics context instance
_physics_context = None
_physics_engine = None


def get_physics_context() -> PhysicsContext:
    """Get the global physics context instance."""
    global _physics_context
    if _physics_context is None:
        builder = PhysicsContextBuilder()
        _physics_context = builder.get_context()
    return _physics_context


def get_physics_engine() -> PhysicsContextEngine:
    """Get the global physics engine instance."""
    global _physics_engine
    if _physics_engine is None:
        context = get_physics_context()
        _physics_engine = PhysicsContextEngine(context)
    return _physics_engine


# Convenience functions for common operations
def concept_to_imas_paths(concept: str) -> List[str]:
    """Convert physics concept to IMAS paths."""
    engine = get_physics_engine()
    return engine.get_imas_paths_for_concept(concept)


def concept_to_units(concept: str) -> Optional[str]:
    """Convert physics concept to units."""
    engine = get_physics_engine()
    return engine.get_units_for_concept(concept)


def concept_to_symbol(concept: str) -> Optional[str]:
    """Convert physics concept to symbol."""
    engine = get_physics_engine()
    return engine.get_symbol_for_concept(concept)


def search_physics_concepts(query: str) -> List[Tuple[str, PhysicsQuantity]]:
    """Search for physics concepts."""
    engine = get_physics_engine()
    return engine.search_concepts(query)


def get_quantities_by_units(units: str) -> List[PhysicsQuantity]:
    """Get all quantities with given units."""
    engine = get_physics_engine()
    return engine.find_quantities_by_units(units)


def get_quantities_by_domain(domain: PhysicsDomain) -> List[PhysicsQuantity]:
    """Get all quantities in a physics domain."""
    engine = get_physics_engine()
    return engine.find_quantities_by_domain(domain)


if __name__ == "__main__":
    # Example usage
    engine = get_physics_engine()

    # Test concept lookup
    print("=== Testing Physics Context Engine ===")

    # Test poloidal flux
    print("\n1. Poloidal flux:")
    psi_paths = concept_to_imas_paths("poloidal flux")
    psi_units = concept_to_units("poloidal flux")
    psi_symbol = concept_to_symbol("poloidal flux")
    print(f"   Paths: {psi_paths[:3]}...")  # Show first 3
    print(f"   Units: {psi_units}")
    print(f"   Symbol: {psi_symbol}")

    # Test temperature
    print("\n2. Electron temperature:")
    te_paths = concept_to_imas_paths("electron temperature")
    te_units = concept_to_units("electron temperature")
    te_symbol = concept_to_symbol("electron temperature")
    print(f"   Paths: {te_paths}")
    print(f"   Units: {te_units}")
    print(f"   Symbol: {te_symbol}")

    # Test search
    print("\n3. Search for 'temperature':")
    results = search_physics_concepts("temperature")
    for concept, quantity in results[:3]:
        print(f"   {concept}: {quantity.symbol} [{quantity.units}]")

    # Test units lookup
    print("\n4. Quantities with units 'eV':")
    ev_quantities = get_quantities_by_units("eV")
    for q in ev_quantities:
        print(f"   {q.concept}: {q.symbol}")

    # Test domain lookup
    print("\n5. Equilibrium domain quantities:")
    eq_quantities = get_quantities_by_domain(PhysicsDomain.EQUILIBRIUM)
    for q in eq_quantities:
        print(f"   {q.concept}: {q.symbol} [{q.units}]")

    # Summary stats
    print("\n6. Summary statistics:")
    stats = engine.get_summary_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
