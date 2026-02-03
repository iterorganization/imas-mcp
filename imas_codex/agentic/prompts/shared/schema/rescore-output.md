## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON.

**Schema derived from Pydantic model:**

```json
{{ rescore_schema_example }}
```

### Field Requirements

{{ rescore_schema_fields }}

### Critical Rules

1. Return ONE result per input directory, in the same order
2. All dimension scores must be 0.0-0.95 (NEVER 1.0) or null to keep original
3. new_score can be 0.0-1.50 for exceptional directories
4. Use 2 decimal places for scores
5. Ensure valid JSON - no trailing commas, proper quoting
6. Set dimension to null if no enrichment evidence supports changing it
7. Do NOT include any text outside the JSON object

