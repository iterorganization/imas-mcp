## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON.

**Schema derived from Pydantic model:**

```json
{{ wiki_scoring_schema_example }}
```

### Field Requirements

{{ wiki_scoring_schema_fields }}

### Critical Rules

1. Return ONE result per input page, in the same order as input
2. All scores must be 0.0-1.0
3. Use 2 decimal places for scores (e.g., 0.75 not 0.7534)
4. Ensure valid JSON - no trailing commas, proper quoting
5. Include ALL fields even if null/default
6. Do NOT include any text outside the JSON object
7. The `id` field MUST exactly match the input page ID
