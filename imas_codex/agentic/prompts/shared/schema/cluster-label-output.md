## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON — no markdown fences, no commentary.

**Schema derived from Pydantic model:**

```json
{{ cluster_label_schema_example }}
```

### Field Requirements

{{ cluster_label_schema_fields }}

### Critical Rules

1. Return ONE result per input cluster, using the cluster's `id`
2. All labels must be unique across the batch
3. Ensure valid JSON — no trailing commas, proper quoting
4. Include ALL fields even if empty lists or defaults
5. Do NOT include any text outside the JSON array
