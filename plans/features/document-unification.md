# Document Unification: WikiDocument + Document → Document

**Status: COMPLETED**

Merged `WikiDocument` (40,126 nodes) and `Document` (0 nodes, unused) into a single source-agnostic `Document` node, following the `Image` pattern with a `source_type` discriminator. Also renamed `ArtifactType` → `DocumentType` with the old `document` value becoming `text_document`.

## What Changed

### Schema
- **Deleted** old `Document` class and old `DocumentType` enum from `facility.yaml`
- **Renamed** `WikiDocument` → `Document` with new `source_type`, `path`, `in_directory` attributes
- **Renamed** `ArtifactType` → `DocumentType` — merged values from both old enums; `document` → `text_document`
- **Added** `DocumentSourceType` enum: wiki, filesystem, repository
- **Renamed** `WikiDocumentStatus` → `DocumentStatus` in `common.yaml`

### Code (~20 files)
- All `WikiDocument` → `Document`, `WikiDocumentStatus` → `DocumentStatus`
- All `ArtifactType` → `DocumentType`, `artifact_type` → `document_type`, `artifact_purpose` → `document_purpose`
- All `_get_artifact_type` methods/functions → `_get_document_type`
- `_ARTIFACT_EXTENSIONS` → `_DOCUMENT_EXTENSIONS`, `SCORABLE_ARTIFACT_TYPES` → `SCORABLE_DOCUMENT_TYPES`
- Old enum value `"document"` → `"text_document"` everywhere (adapters, handlers, tests)
- Prompt templates renamed: `artifact-scorer.md` → `document-scorer.md`

### Graph Migration
- 40,126 `WikiDocument` → `Document` nodes relabeled
- `artifact_type` → `document_type` property renamed on all nodes
- `artifact_purpose` → `document_purpose` property renamed on 20,240 nodes
- 792 `document_type: "document"` → `"text_document"`
- `source_type: "wiki"` set on all existing nodes
- Stale `WikiArtifact` indexes dropped
- `document_desc_embedding` vector index created

### Enum Summary

| Enum | Status |
|------|--------|
| `DocumentType` | **NEW** — merged ArtifactType + DocumentType; `document` → `text_document` |
| `DocumentSourceType` | **NEW** — wiki, filesystem, repository |
| `DocumentStatus` | **RENAMED** from WikiDocumentStatus |
| `ArtifactType` | **DELETED** — replaced by DocumentType |
| `DocumentType` (old) | **DELETED** — merged into new DocumentType |
| `WikiDocumentStatus` | **DELETED** — renamed to DocumentStatus |
