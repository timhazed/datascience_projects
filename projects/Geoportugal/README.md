# GeoPortugal API

Production-ready geospatial API serving hierarchical Portugal location data from the GeoNames dataset.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Frontend (Next.js 15)                          │
│                         React 19 + Apollo Client + Leaflet                  │
└─────────────────────────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Backend                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   REST API      │    │    GraphQL      │    │   Monitoring    │         │
│  │   /api/v1/*     │    │    /graphql     │    │   /health/*     │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                   │
│           └──────────────────────┴──────────────────────┘                   │
│                                  │                                          │
│                                  ▼                                          │
│                        ┌─────────────────┐                                  │
│                        │ LocationService │                                  │
│                        └────────┬────────┘                                  │
│                                 │                                           │
│              ┌──────────────────┼──────────────────┐                        │
│              ▼                  ▼                  ▼                        │
│    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│    │  Repositories   │ │  CacheService   │ │    Schemas      │              │
│    │  (SQLAlchemy)   │ │    (Redis)      │ │   (Pydantic)    │              │
│    └────────┬────────┘ └────────┬────────┘ └─────────────────┘              │
└─────────────┼───────────────────┼───────────────────────────────────────────┘
              │                   │
              ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐
    │   PostgreSQL    │ │     Redis       │
    │    + PostGIS    │ │     Cache       │
    └─────────────────┘ └─────────────────┘
```

## Features

- Hierarchical data: Districts → Municipalities → Localities
- Full-text search with Redis caching
- Geospatial queries with PostGIS
- REST API + GraphQL endpoint
- 75%+ test coverage

## Quick Start

```bash
git clone <repository-url>
cd geoportugal
make setup
make dev-up

# Backend API: http://localhost:8000/docs
# GraphQL:     http://localhost:8000/graphql
# Frontend:    http://localhost:3000
```

## API Endpoints

### REST (`/api/v1/`)

| Endpoint | Description |
|----------|-------------|
| `GET /districts` | List all districts |
| `GET /districts/{id}` | District details |
| `GET /districts/{id}/municipalities` | Municipalities in district |
| `GET /municipalities/{id}` | Municipality details |
| `GET /municipalities/{id}/localities` | Localities in municipality |
| `GET /localities/{id}` | Locality details |
| `GET /search?q={term}` | Full-text search |
| `GET /nearby?lat={lat}&lng={lng}&radius_km={r}` | Geospatial search |

### GraphQL (`/graphql`)
Interactive playground with schema introspection.

### Monitoring

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health check |
| `GET /health/detailed` | System status with DB/Redis checks |
| `GET /ready` | Kubernetes readiness probe |
| `GET /live` | Kubernetes liveness probe |
| `GET /metrics` | Prometheus metrics (auth required) |

## Technology Stack

**Backend**: FastAPI, SQLAlchemy 2.0, PostgreSQL/PostGIS, Redis, Strawberry GraphQL

**Frontend**: Next.js 15, React 19, Apollo Client, Leaflet, TailwindCSS

## Development

```bash
make help              # Show all commands
make setup             # Install dependencies
make dev-up            # Start full stack
make dev-down          # Stop services
make backend-test      # Run tests
make backend-format    # Format code
make lint              # Run linting
```

## Configuration

External `.env` file: checks `<project-root>/.env` first, falls back to `~/src/python/geoportugal/.env`

Required variables:
- `SECRET_KEY` - Minimum 32 characters
- `DATABASE_URL` - PostgreSQL connection string

## Project Structure

```
backend/
├── app/
│   ├── api/routes/     # REST endpoints
│   ├── graphql/        # GraphQL resolvers
│   ├── core/           # Config, logging
│   ├── db/             # Models, repositories
│   ├── services/       # Business logic
│   └── schemas/        # Pydantic models
├── tests/
├── migrations/         # Alembic
└── pyproject.toml      # Poetry

frontend/
├── src/app/            # Next.js app router
└── package.json
```
