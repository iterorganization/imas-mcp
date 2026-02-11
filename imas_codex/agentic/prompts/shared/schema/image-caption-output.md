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
6. `description` is a brief 1-2 sentence summary (max 150 chars) — this gets embedded for search
7. All six `score_*` dimensions MUST be between 0.0 and 1.0
8. `should_ingest` — set true if image has any fusion physics value
9. `purpose` — classify using same categories as wiki pages
10. Ensure valid JSON — no trailing commas, proper quoting
11. Do NOT include any text outside the JSON object
