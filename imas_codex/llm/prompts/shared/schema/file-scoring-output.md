## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON.

**Schema derived from Pydantic model:**

```json
{{ file_scoring_schema_example }}
```

### Field Requirements

{{ file_scoring_schema_fields }}

### Critical Rules

1. Return ONE result per input file, in the same order
2. All scores must be 0.0-1.0
3. Use 2 decimal places for scores
4. Ensure valid JSON - no trailing commas, proper quoting
5. Include ALL fields even if null/default
6. Do NOT include any text outside the JSON object
