# Enhanced Relationships Tool Output Structure

## New Top-Level Summary Section

The tool now provides immediate context at the top level through a `summary` section in `relationship_insights`:

```json
{
  "relationship_insights": {
    "summary": {
      "query_path": "core_profiles/profiles_1d/electrons/density",
      "total_found": 8,
      "intra_ids_similar": [
        "core_profiles/profiles_1d/electrons/density_fast",
        "core_profiles/profiles_1d/electrons/density_thermal"
      ],
      "cross_ids_similar": [
        "core_profiles/profiles_1d/electrons/temperature",
        "core_profiles/profiles_1d/electrons/temperature_fit",
        "core_profiles/profiles_1d/electrons/temperature_validity"
      ],
      "strongest_relationships": [
        {"path": "core_profiles/profiles_1d/electrons/density_thermal", "strength": 0.95, "type": "cluster_intra_ids"},
        {"path": "core_profiles/profiles_1d/electrons/density_fast", "strength": 0.92, "type": "cluster_intra_ids"},
        {"path": "core_profiles/profiles_1d/electrons/temperature", "strength": 0.78, "type": "cluster_cross_ids"}
      ],
      "ids_involved": ["core_profiles", "edge_profiles"],
      "primary_physics_domain": "transport"
    },
    "total_relationships": 19,
    "avg_strength": 0.763,
    "clusters": {...},
    "strength_distribution": {...}
  }
}
```

## Benefits for LLM Consumption

### **1. Immediate Context (Summary Section)**

✅ **Query Understanding**: LLMs immediately see what was searched  
✅ **Scope Overview**: Total relationships found and strength distribution  
✅ **Categorized Relationships**: Clear separation of intra-IDS vs cross-IDS  
✅ **Top Relationships**: Strongest connections highlighted upfront  
✅ **Physics Context**: Primary domain identified immediately

### **2. Structured Relationship Types**

- **Intra-IDS Similar**: Related quantities within the same IDS (structural relationships)
- **Cross-IDS Similar**: Related quantities across different IDS (semantic relationships)
- **Strongest Relationships**: Most relevant connections with explicit strength scores

### **3. LLM Processing Advantages**

1. **Quick Scanning**: Summary provides overview before detailed analysis
2. **Relevance Filtering**: LLMs can focus on strongest relationships first
3. **Context Awareness**: Physics domain and IDS scope established upfront
4. **Structured Data**: Consistent format for parsing and reasoning

### **4. Example Use Cases**

- **Research Questions**: "What are the strongest related quantities?" → Check `strongest_relationships`
- **Domain Analysis**: "What physics domain is this in?" → Check `primary_physics_domain`
- **Scope Understanding**: "Is this cross-IDS or intra-IDS?" → Check relationship categorization
- **Data Discovery**: "What similar measurements exist?" → Scan `intra_ids_similar` and `cross_ids_similar`

## Implementation Features

1. **Smart Limiting**: Top 10 intra/cross-IDS paths, top 5 strongest relationships
2. **Strength-Based Prioritization**: Strongest relationships highlighted first
3. **Physics Domain Context**: Primary domain extracted from analysis
4. **IDS Coverage**: All involved IDS listed for scope understanding
5. **Type Categorization**: Clear separation of relationship types

This enhancement transforms the tool from a detailed dump of relationships into a **contextually organized, LLM-friendly interface** that provides immediate understanding before diving into specifics.
