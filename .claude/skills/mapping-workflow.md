# IMAS Mapping Workflow

How mappings connect facility signals to IMAS fields.

## Architecture

Three symmetric levels:

| Level | Source Side | Target Side |
|-------|-----------|-------------|
| Individual | `SignalNode` | `IMASNode` leaf (e.g., FLT_0D) |
| Group | `SignalGroup` | `IMASNode` STRUCT_ARRAY |
| Collection | `IMASMapping` | `IDS` |

## Graph Structure

```
IMASMapping ──[:USES_SIGNAL_GROUP]──▶ SignalGroup ──[:MAPS_TO_IMAS {field props}]──▶ IMASNode
    │                                     ▲
    │                                [:MEMBER_OF]
    │                                     │
    │                              SignalNode / FacilitySignal
    │
    └──[:POPULATES {assembly props}]──▶ IMASNode (STRUCT_ARRAY root)
```

## Key Relationships

| Relationship | From → To | Purpose |
|---|---|---|
| `MAPS_TO_IMAS` | SignalGroup/SignalNode → IMASNode | Field-level mapping with transform |
| `POPULATES` | IMASMapping → IMASNode (STRUCT_ARRAY) | Assembly section target with config |
| `USES_SIGNAL_GROUP` | IMASMapping → SignalGroup | Assembly sources |
| `MEMBER_OF` | SignalNode/FacilitySignal → SignalGroup | Group membership |
| `HAS_EVIDENCE` | SignalGroup → MappingEvidence | Evidence for group's mapping |

## MAPS_TO_IMAS Properties

| Property | Type | Purpose |
|----------|------|---------|
| `source_property` | string | Field on SignalNode to extract (e.g., "r", "z") |
| `transform_code` | string | Python expression with `value` as input |
| `units_in` | string | Source units |
| `units_out` | string | Target units |
| `status` | string | "validated" or "proposed" |
| `confidence` | float | 0.0-1.0 |
| `cocos_source` | integer | Source COCOS convention |
| `cocos_target` | integer | Target COCOS convention |

## Full Traversal Query

```cypher
MATCH (m:IMASMapping {id: $mapping_id})
      -[t:POPULATES]->(root:IMASNode)
MATCH (root)<-[:HAS_PARENT*]-(leaf:IMASNode)
      <-[map:MAPS_TO_IMAS]-(sg:SignalGroup)
      <-[:MEMBER_OF]-(n:SignalNode)
WHERE (m)-[:USES_SIGNAL_GROUP]->(sg)
RETURN leaf.id, map.source_property, map.transform_code, n.path
```

## Key Checks

- **Units**: Must be pint-compatible. Use `pint.UnitRegistry()` to verify conversion.
- **Sign convention**: Check COCOS. TCV uses COCOS 17 for equilibrium.
- **Array shape**: IMAS expects specific dimension ordering (time last in many cases).
- **NaN handling**: Check for NaN/inf in source data before mapping.
