# Docker Container Setup

This document describes how to build, run, and deploy the IMAS Codex Server container.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f imas-codex

# Stop the container
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t imas-codex .

# Run the container
docker run -d \
  --name imas-codex \
  -p 8000:8000 \
  -v ./index:/app/index:ro \
  imas-codex
```

## GitHub Container Registry

The container is automatically built and pushed to GitHub Container Registry on tagged releases.

### Pull from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/iterorganization/imas-codex:latest

# Pull a specific version
docker pull ghcr.io/iterorganization/imas-codex:v1.0.0

# Run the pulled image
docker run -d \
  --name imas-codex \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-codex:latest
```

## Available Tags

- `latest` - Latest build from main branch
- `main` - Latest build from main branch (same as latest)
- `v*` - Tagged releases (e.g., `v1.0.0`, `v1.1.0`)
- `pr-*` - Pull request builds

## Environment Variables

| Variable     | Description               | Default |
| ------------ | ------------------------- | ------- |
| `PYTHONPATH` | Python path               | `/app`  |
| `PORT`       | Port to run the server on | `8000`  |

## Volume Mounts

| Path         | Description                                |
| ------------ | ------------------------------------------ |
| `/app/index` | Index files directory (mount as read-only) |
| `/app/logs`  | Application logs (optional)                |

## Health Check

The container includes a health check that verifies the server is responding correctly. The server uses `streamable-http` transport by default, which exposes a dedicated health endpoint that checks both server availability and search index functionality. The server runs in stateful mode to support MCP sampling functionality:

```bash
# Check container health status
docker ps
# Look for "healthy" status in the STATUS column

# Manual health check using the dedicated endpoint
curl -f http://localhost:8000/health
# Example health response
{
  "status": "healthy",
  "service": "imas-codex-server",
  "version": "4.0.1.dev164",
  "index_stats": {
    "total_paths": 15420,
    "index_name": "lexicographic_4.0.1.dev164"
  },
  "transport": "streamable-http"
}
```

### Health Check Configuration

The health check is configured in `docker-compose.yml`:

```yaml
healthcheck:
  test:
    [
      "CMD",
      "python",
      "-c",
      "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')",
    ]
  interval: 30s # Check every 30 seconds
  timeout: 10s # 10 second timeout per check
  retries: 3 # Mark unhealthy after 3 consecutive failures
  start_period: 40s # Wait 40 seconds before starting checks
```

**Note**: The health endpoint is available when using `streamable-http` transport (default). For other transports (`stdio`, `sse`), the health check will verify port connectivity only.

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Use the production profile
docker-compose --profile production up -d
```

This will start both the IMAS Codex Server and an Nginx reverse proxy.

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: imas-codex
spec:
  replicas: 2
  selector:
    matchLabels:
      app: imas-codex
  template:
    metadata:
      labels:
        app: imas-codex
    spec:
      containers:
        - name: imas-codex
          image: ghcr.io/iterorganization/imas-codex:latest
          ports:
            - containerPort: 8000
          env:
            - name: PYTHONPATH
              value: "/app"
          volumeMounts:
            - name: index-data
              mountPath: /app/index
              readOnly: true
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
      volumes:
        - name: index-data
          persistentVolumeClaim:
            claimName: imas-index-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: imas-codex-service
spec:
  selector:
    app: imas-codex
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Development

### Building locally

```bash
# Build the image
docker build -t imas-codex:dev .

# Run with development settings
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  imas-codex:dev
```

### Debugging

```bash
# Run with interactive shell
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  ghcr.io/iterorganization/imas-codex:latest \
  /bin/bash

# View logs
docker logs -f imas-codex
```

## Troubleshooting

### Common Issues

1. **Container fails to start**

   - Check that port 8000 is available
   - Verify index files are properly mounted
   - Check logs: `docker-compose logs imas-codex`

2. **Index files not found**

   - Ensure the index directory exists and contains the necessary files
   - Check volume mount permissions
   - Verify the index files were built correctly

3. **Memory issues**
   - The container may need more memory for large indexes
   - Consider using Docker's memory limits: `--memory=2g`

### Performance Tuning

```bash
# Run with increased memory
docker run -d \
  --name imas-codex \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-codex:latest
```

## CI/CD Pipeline

The project includes GitHub Actions workflows for:

1. **Testing** (`.github/workflows/test.yml`)

   - Runs on every push and PR
   - Executes linting, formatting, and tests

2. **Container Build** (`.github/workflows/docker-build-push.yml`)

   - Builds and pushes containers to GHCR
   - Supports multi-architecture builds (amd64, arm64)
   - Runs on pushes to main and tagged releases

3. **Releases** (`.github/workflows/release.yml`)
   - Creates GitHub releases for tagged versions
   - Builds and uploads Python packages

### GitHub Secrets Configuration

The `OPENAI_API_KEY` secret is **optional** for Docker builds.

**What runs locally (no API key needed):**
- Schema building (XML parsing)
- Path map generation
- Embeddings (uses local `sentence-transformers` model `all-MiniLM-L6-v2`)
- Clustering (HDBSCAN algorithm)

**What uses the API key (optional):**
- **Cluster labeling**: Generates human-readable labels for semantic clusters using an LLM via OpenRouter

**Fallback behavior without API key:**
1. Uses pre-cached labels from `imas_codex/definitions/clusters/labels.json` (version-controlled)
2. Falls back to auto-generated labels from path names if no cache exists

**Configuring the secret (optional):**
1. **Repository Settings**: Go to `Settings` → `Secrets and variables` → `Actions`
2. **Add Secret**: Create a new repository secret:
   - **Name**: `OPENAI_API_KEY`
   - **Value**: Your OpenRouter API key

**Manual Docker Build:**
```bash
# Build without API key (uses cached/fallback labels)
docker build -t imas-codex .

# Build with API key for fresh LLM-generated labels
docker build --secret id=OPENAI_API_KEY,env=OPENAI_API_KEY -t imas-codex .

# Build with minimal IDS for faster iteration
docker build --build-arg IDS_FILTER="equilibrium" -t imas-codex:test .
```

## Security

- Containers run as non-root user
- No sensitive data stored in container
- Regular security updates via base image updates
- Signed container images with attestations
