# Docker Setup Guide

## Prerequisites
- Docker Desktop installed ([download](https://www.docker.com/products/docker-desktop))
- At least 4 GB RAM allocated to Docker

## Quick Start

### Build the Image
```bash
docker build -t financial-analytics:latest .
```

### Run the Analytics Pipeline
```bash
docker run -v $(pwd)/artifacts:/app/artifacts financial-analytics:latest
```
Outputs land in your local `artifacts/` directory.

### Run the API Server
```bash
docker-compose up api
```
Visit the interactive docs at **http://localhost:8000/docs**.

### Run Both (Pipeline then API)
```bash
docker-compose up
```

## Production Deployment

### Push to a Container Registry
```bash
docker tag financial-analytics:latest your-registry/financial-analytics:latest
docker push your-registry/financial-analytics:latest
```

### Run in Detached Mode
```bash
docker run --detach --restart unless-stopped \
  -v /data/artifacts:/app/artifacts \
  financial-analytics:latest
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `prophet` build fails | Ensure `build-essential` is in Dockerfile (it is by default) |
| Out of memory | Increase Docker memory limit to 4+ GB |
| Artifact dir empty | Check volume mount path matches `$(pwd)/artifacts` |
| Port 8000 in use | Change the port mapping in `docker-compose.yml` |
