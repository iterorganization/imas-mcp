# Rich Progress Monitoring in Wiki Commands

This document tracks which CLI commands use Rich progress monitoring.

## Commands with Rich Progress

| Command | Rich Progress | Progress Features |
|---------|---------------|-------------------|
| `wiki crawl` | ✅ Yes | Live progress bar with running totals (discovered, frontier, depth) |
| `wiki ingest` | ✅ Yes | Multi-stage progress bars, live statistics panel, content preview with MDSplus paths |
| `wiki discover` | ✅ Yes | Uses CrawlProgressMonitor for phase 1, standard progress for phase 2 |
| `wiki status` | ❌ No | Table output only |

## Implementation Details

### `wiki crawl`

Uses `CrawlProgressMonitor` with live updating display showing:

- **Progress Bar**: Dynamic percentage (crawled / (crawled + frontier))
- **Statistics Grid** (compact 2-column layout):
  - Crawled (pages processed) / Frontier (queue size)
  - Current depth / Max depth reached
  - Skipped pages / Processing rate
- **Current Page**: Shows page being crawled

Graph-driven: restarts resume from existing state. Already-crawled
pages are loaded from the graph; pending pages form the frontier.

**Example output:**
```
⠧ Crawling wiki ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  23.4%
Crawled:           188    Frontier:         612
Depth:               6    Max depth:          6
Rate:           4.2/s
→ Thomson_scattering

✓ Crawled 188 pages in 45.2s
```

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

- `imas_codex/wiki/progress.py` - WikiProgressMonitor and CrawlProgressMonitor implementations
- `imas_codex/wiki/pipeline.py` - Pipeline that uses the monitor
- `imas_codex/wiki/discovery.py` - Discovery pipeline using CrawlProgressMonitor
- `imas_codex/cli.py` - CLI command implementations
