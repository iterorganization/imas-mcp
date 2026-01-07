# Rich Progress Monitoring in Wiki Commands

This document tracks which CLI commands use Rich progress monitoring.

## Commands with Rich Progress

| Command | Rich Progress | Progress Features |
|---------|---------------|-------------------|
| `wiki ingest` | ✅ Yes | Multi-stage progress bars, live statistics panel, content preview with MDSplus paths |
| `wiki discover` | ❌ No | Simple console output |
| `wiki status` | ❌ No | Table output only |

## Implementation Details

### `wiki ingest`

Uses `WikiProgressMonitor` with live updating panels showing:

- **Progress Bar**: Current page processing with spinner
- **Statistics Panel**: 
  - Chunks created
  - TreeNodes linked (cyan)
  - IMAS paths linked (cyan)
  - Conventions found
  - Processing rate (pages/sec)
- **Content Preview Panel** (per page):
  - Page title
  - First 200 chars of extracted content
  - MDSplus paths found (up to 5, green text)

**Example output:**
```
⠦ Scraping: CXRS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3/3 0:00:21 → 0:00:00
╭──────────────────────────────────── Statistics ─────────────────────────────────────╮
│ Chunks:                    18                                                       │
│ TreeNodes:                 12                                                       │
│ IMAS paths:                5                                                        │
│ Conventions:               2                                                        │
│ Rate:         0.14 pages/sec                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────── CXRS ────────────────────────────────────────╮
│ Content:      Factsheet INSTALLED Full name Charge eXchange Recombination           │
│               Spectroscopy (LFS system) Abbreviation CXRS Measures Density...       │
│ MDSplus:      \RESULTS::CXRS, \ACQUISITION::CXRS                                   │
╰─────────────────────────────────────────────────────────────────────────────────────╯
```

## Future Enhancements

Potential Rich progress additions:

- `wiki discover`: Add progress bar when scanning large page lists
- `wiki status`: Add sparklines for ingestion trends over time
- `ingest run`: Already has Rich progress via code ingestion pipeline

## Related Files

- `imas_codex/wiki/progress.py` - WikiProgressMonitor implementation
- `imas_codex/wiki/pipeline.py` - Pipeline that uses the monitor
- `imas_codex/cli.py` - CLI command implementations
