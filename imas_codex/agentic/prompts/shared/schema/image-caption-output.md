## Required Output Format (CRITICAL)

You MUST return valid JSON matching this EXACT structure. The response MUST be parseable JSON.

**Schema derived from Pydantic model:**

```json
{{ image_caption_schema_example }}
```

### Field Requirements

{{ image_caption_schema_fields }}

### Critical Rules

1. Return ONE result per input image, in the same order as input
2. The `id` field MUST exactly match the input image ID
3. `caption` MUST describe the **physics content**, not just visual appearance
4. Include specific quantities, paths, diagnostics, or conventions shown
5. If text is visible in the image (axis labels, legends, titles), transcribe it into `ocr_text`
6. Ensure valid JSON - no trailing commas, proper quoting
7. Do NOT include any text outside the JSON object
