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
3. `description` MUST describe the **physics content**, not just visual appearance
4. Include specific quantities, paths, diagnostics, or conventions shown
5. If text is visible in the image (axis labels, legends, titles), transcribe it into `ocr_text`
6. `description` should be a substantial paragraph (4-8 sentences) for complex images like schematics, multi-panel plots, or data flow diagrams. Simple images need only 1-2 sentences.
7. For schematics and block diagrams, include a mermaid diagram in the `mermaid_diagram` field
9. All six `score_*` dimensions MUST be between 0.0 and 1.0
10. `should_ingest` — set true if image has any fusion physics value
11. `purpose` — classify using same categories as wiki pages
12. Ensure valid JSON — no trailing commas, proper quoting
13. Do NOT include any text outside the JSON object
