# Performance Benchmarking with GitHub Actions

This document explains how to set up continuous performance benchmarking for the IMAS MCP project using GitHub Actions and GitHub Pages.

## Overview

The benchmarking system provides:

- **Automated benchmark execution** on every push and PR
- **Historical data persistence** using GitHub Artifacts
- **Web-hosted results** via GitHub Pages
- **Performance regression detection** with configurable thresholds
- **PR comments** with benchmark summaries

## Setup Instructions

### 1. Enable GitHub Pages

1. Go to your repository settings
2. Navigate to "Pages" section
3. Set source to "GitHub Actions"
4. The benchmark results will be available at:
   `https://your-username.github.io/your-repo-name/benchmarks/`

### 2. Repository Configuration

The setup includes these key files:

```
.github/workflows/benchmarks.yml  # GitHub Actions workflow
asv.conf.json                     # ASV configuration (enhanced)
scripts/run_performance_baseline.py  # CI-aware benchmark runner
benchmarks/                       # Benchmark definitions
```

### 3. Workflow Features

#### Automatic Triggers

- **Push to main/develop**: Runs full benchmark suite
- **Pull Requests**: Runs benchmarks and comments on PR
- **Daily Schedule**: 2 AM UTC benchmark runs
- **Manual Trigger**: Run specific benchmark subsets

#### Data Persistence

- **GitHub Artifacts**: Store `.asv` directory for 90 days
- **Artifact Download**: Restores previous results before each run
- **Cumulative History**: Builds performance history over time

#### GitHub Pages Deployment

- **HTML Reports**: Generated ASV reports deployed automatically
- **Main Branch Only**: Only deploys from main branch
- **Clean Deployment**: Replaces previous reports

## Usage

### Running Benchmarks Locally

```bash
# Full benchmark suite
python scripts/run_performance_baseline.py

# With environment detection (CI vs local)
CI=true python scripts/run_performance_baseline.py
```

### Manual GitHub Actions

1. Go to "Actions" tab in your repository
2. Select "Performance Benchmarks" workflow
3. Click "Run workflow"
4. Optionally specify benchmark filter (e.g., "SearchBenchmarks")

### Viewing Results

- **GitHub Pages**: `https://your-username.github.io/your-repo-name/benchmarks/`
- **Local Server**: `python -m http.server 8000 --directory .asv/html`
- **Direct File**: Open `.asv/html/index.html` in browser

## Configuration

### ASV Settings (`asv.conf.json`)

```json
{
  "regressions_thresholds": {
    "regress": 1.5, // 50% performance degradation triggers alert
    "improve": 0.8 // 20% improvement marks optimization
  }
}
```

### Environment Variables

- `CI`: Detected automatically, switches to simple progress output
- `GITHUB_ACTIONS`: Enables GitHub-specific features
- `GITHUB_TOKEN`: Required for Pages deployment (auto-provided)

## Performance Data

### Data Structure

```
.asv/
├── results/           # Raw benchmark data (persisted via artifacts)
│   ├── benchmarks.json
│   └── machine-name/
├── html/             # Generated reports (deployed to Pages)
│   ├── index.html
│   ├── graphs/
│   └── ...
└── env/              # Virtual environments (not persisted)
```

### Retention Policy

- **GitHub Artifacts**: 90 days
- **GitHub Pages**: Persistent (until next deployment)
- **Local Results**: Until manual cleanup

## Monitoring

### Performance Regression Alerts

- Configured in `asv.conf.json`
- Automatically detected in HTML reports
- Can be enhanced with GitHub notifications

### PR Integration

- Benchmark results commented on PRs
- Links to full reports
- Performance comparison with main branch

## Troubleshooting

### Common Issues

1. **Artifacts not found**: Normal for first run
2. **Pages deployment fails**: Check repository settings
3. **Benchmark timeouts**: Adjust ASV configuration
4. **Missing dependencies**: Verify `uv sync --extra bench`

### Debug Commands

```bash
# Verify ASV setup
asv check

# Test benchmark locally
asv dev

# Check results format
asv show latest
```

## Advanced Features

### Custom Benchmark Filters

```yaml
# In workflow dispatch
benchmark_filter: "SearchBenchmarks.time_search_imas_basic"
```

### Multiple Python Versions

```json
// In asv.conf.json
"pythons": ["3.11", "3.12"]
```

### Performance Baselines

The system automatically maintains baselines for:

- Current commit vs previous commits
- PR branches vs main branch
- Historical trend analysis

## Security Notes

- Uses minimal permissions
- Only writes to Pages and Artifacts
- No external dependencies
- Runs in isolated GitHub Actions environment
