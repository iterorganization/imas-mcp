# Docker Container Setup

This document describes how to build, run, and deploy the IMAS MCP Server container.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up -d

# View logs
docker-compose logs -f imas-mcp-server

# Stop the container
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t imas-mcp-server .

# Run the container
docker run -d \
  --name imas-mcp-server \
  -p 8000:8000 \
  -v ./index:/app/index:ro \
  imas-mcp-server
```

## GitHub Container Registry

The container is automatically built and pushed to GitHub Container Registry on every push to `main` and on tagged releases.

### Pull from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/iterorganization/imas-mcp-server:latest

# Pull a specific version
docker pull ghcr.io/iterorganization/imas-mcp-server:v1.0.0

# Run the pulled image
docker run -d \
  --name imas-mcp-server \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-mcp-server:latest
```

## Available Tags

- `latest` - Latest build from main branch
- `main` - Latest build from main branch (same as latest)
- `v*` - Tagged releases (e.g., `v1.0.0`, `v1.1.0`)
- `pr-*` - Pull request builds

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python path | `/app` |
| `PORT` | Port to run the server on | `8000` |

## Volume Mounts

| Path | Description |
|------|-------------|
| `/app/index` | Index files directory (mount as read-only) |
| `/app/logs` | Application logs (optional) |

## Health Check

The container includes a health check that verifies the server is responding:

```bash
# Check container health
docker ps
# Look for "healthy" status

# Manual health check
curl -f http://localhost:8000/health
```

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Use the production profile
docker-compose --profile production up -d
```

This will start both the IMAS MCP Server and an Nginx reverse proxy.

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: imas-mcp-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: imas-mcp-server
  template:
    metadata:
      labels:
        app: imas-mcp-server
    spec:
      containers:
      - name: imas-mcp-server
        image: ghcr.io/iterorganization/imas-mcp-server:latest
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
  name: imas-mcp-server-service
spec:
  selector:
    app: imas-mcp-server
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
docker build -t imas-mcp-server:dev .

# Run with development settings
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e PYTHONPATH=/app \
  imas-mcp-server:dev
```

### Debugging

```bash
# Run with interactive shell
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  ghcr.io/iterorganization/imas-mcp-server:latest \
  /bin/bash

# View logs
docker logs -f imas-mcp-server
```

## Troubleshooting

### Common Issues

1. **Container fails to start**
   - Check that port 8000 is available
   - Verify index files are properly mounted
   - Check logs: `docker-compose logs imas-mcp-server`

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
  --name imas-mcp-server \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  ghcr.io/iterorganization/imas-mcp-server:latest
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

## Security

- Containers run as non-root user
- No sensitive data stored in container
- Regular security updates via base image updates
- Signed container images with attestations
