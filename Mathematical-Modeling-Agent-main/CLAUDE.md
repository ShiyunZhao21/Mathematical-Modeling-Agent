# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MathModelAgent is a mathematical modeling automation system designed specifically for math competitions. It automates the entire process of mathematical modeling, code generation, and paper writing to produce competition-ready submissions.

## Common Commands

### Backend Development
```bash
cd backend
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
venv\Scripts\activate.bat # Windows

# Start development server
ENV=DEV uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-ping-interval 60 --ws-ping-timeout 120 --reload

# Lint and format (using ruff)
ruff check .
ruff format .
```

### Frontend Development
```bash
cd frontend
# Install dependencies
pnpm i

# Start development server
pnpm run dev

# Build for production
pnpm run build
```

### Docker Deployment
```bash
# Start all services
docker-compose up

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

## Architecture Overview

### Project Structure
- **frontend/**: Vue 3 + TypeScript + Vite web application
- **backend/**: FastAPI-based Python backend with mathematical modeling logic
- **docker-compose.yml**: Container orchestration for development and deployment

### Core Components

#### Backend Architecture
- **FastAPI Application**: Async web framework with WebSocket support
- **Multi-Agent System**: 
  - `ModelerAgent`: Analyzes problems and creates mathematical models
  - `CoderAgent`: Generates and executes Python code using Jupyter notebooks
  - `WriterAgent`: Composes academic papers from modeling results
- **Code Interpreters**: 
  - Local Jupyter-based interpreter (saves notebooks for editing)
  - Cloud interpreters (E2B, Daytona) for remote execution
- **Task Management**: Redis-based queuing and status tracking
- **LLM Integration**: LiteLLM for multi-model support across different agents

#### Frontend Architecture
- **Vue 3 Composition API** with TypeScript
- **Pinia** for state management
- **Tailwind CSS** for styling
- **WebSocket Client**: Real-time task progress updates
- **Multi-page Application**: Chat interface, task details, and configuration

### Key Workflow
1. User uploads data files and provides problem description
2. System creates modeling task with unique task_id
3. Agents work sequentially:
   - Modeler analyzes problem and proposes mathematical approach
   - Coder implements solution in Jupyter notebooks
   - Writer generates formatted academic paper
4. Real-time progress updates via WebSocket
5. Results saved to `backend/project/work_dir/{task_id}/`

### Configuration Files
- **Backend**: `pyproject.toml` for Python dependencies, `.env.dev` for environment
- **Frontend**: `package.json` for Node.js dependencies, `.env.development` for config
- **Docker**: `docker-compose.yml` orchestrates Redis, backend, and frontend services

### Development Environment Requirements
- Python 3.12+ (with uv for package management)
- Node.js with pnpm
- Redis server
- Docker (optional, for containerized deployment)

## Important Development Notes

### Backend Development
- All new API routes should go in `backend/app/routers/` and be registered in `main.py`
- Core business logic belongs in `backend/app/core/`
- Utility functions should be placed in `backend/app/utils/`
- Use async/await patterns throughout for FastAPI compatibility
- Redis is used for task state management and WebSocket message broadcasting

### Frontend Development  
- Page components in `frontend/src/pages/` with `index.vue` as entry point
- Reusable components in `frontend/src/components/`
- API calls through `frontend/src/utils/request.ts` (axios wrapper)
- Use Vue 3 Composition API with `<script setup lang="ts">` syntax
- WebSocket connection for real-time task updates

### Testing and Quality
- Backend: Use ruff for linting and formatting (configured in pyproject.toml)
- Frontend: TypeScript compilation and Biome for linting
- Generated files are stored in `backend/project/work_dir/` for inspection