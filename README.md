# Yubu Code - Hackathon Project

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15.4+-black.svg)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.1+-blue.svg)](https://reactjs.org/)
[![Docker](https://img.shields.io/badge/Docker-3.9+-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hackathon](https://img.shields.io/badge/Hackathon-Hack%20Nation-orange.svg)](https://github.com/)

## Project Description

**Yubu Code** is an **AgentOps Replay** system built during the Hack Nation hackathon for the **VC Big Bets (Agents)** track. This project addresses the critical enterprise need for AI agent transparency, compliance, and debugging capabilities.

### Motivation & Goal

Enterprises need confidence that AI agents follow policy, use tools correctly, and make auditable decisions. When something goes wrong, teams must replay what happened, step through decisions, and prove compliance.

**Our Goal:** Build a simulation and replay arena for AI agents that:
- Captures structured traces of each step (prompts, tool calls, retrieved docs, model I/O)
- Visualizes the workflow and decisions
- Replays any session deterministically for debugging and compliance review

### Why This Matters

Without clear logs and replay tools, agent behavior is a black box—making it hard to debug failures, audit decisions, or prove compliance. This system empowers teams to turn opaque AI workflows into transparent, reproducible processes, enabling safer deployment and faster iteration in enterprise environments.

### Core Features (MVP)

- **Universal Agent Logger**: Records prompts, tool calls, retrievals, outputs, parameters, and timestamps
- **Visualization UI**: Graph + timeline view of agent actions with click-through inspection of each step
- **Deterministic Replay**: Sandbox mode to re-run sessions using recorded data

### Stretch Goals

- **Compliance Pack**: Policy violation checks and audit report export
- **Counterfactual Replay**: Change prompts or parameters and compare outcomes

### Compliance Process

Our system implements a **BART-based compliance engine** using the `facebook/bart-large-mnli` model for zero-shot classification:

- **Zero-shot Classification**: Automatically detects harmful, toxic, unethical, or biased content without pre-training
- **Multi-label Detection**: Evaluates text against multiple compliance categories simultaneously
- **Configurable Thresholds**: Adjustable violation detection sensitivity (default: 0.8)
- **Real-time Analysis**: Processes text during agent interactions for immediate compliance feedback
- **Audit Trail**: Maintains detailed logs of all compliance checks for regulatory review

### Agent Type

We've developed a **Resume Analysis Agent** as our example workflow. Our system demonstrates:
- Resume parsing and skill extraction
- Skill-to-job requirement matching
- Automated candidate evaluation


### Architecture

The system consists of three main components:

1. **Backend API** (FastAPI): Core API for managing runs, tasks, steps, and compliance checks
2. **Logger Service**: LangGraph-based agent wrapper with comprehensive logging capabilities
3. **Web Dashboard**: Next.js frontend with React Flow for visualizing agent workflows

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local development)
- Node.js 18+ (for local development)

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/valfvo/hackathon-yubu-code
   cd hack-nation-hackathon
   ```

2. **Start all services**
   ```bash
   docker-compose up --build
   ```

3. **Access the services**
   - Backend API: http://localhost:8000
   - Web Dashboard: http://localhost:3000 (when frontend is implemented)
   - Logger Service: http://localhost:8001

### Local Development Setup

#### Backend Development

1. **Navigate to Backend directory**
   ```bash
   cd Backend
   ```

2. **Install dependencies using Pixi**
   ```bash
   pixi install
   ```

3. **Run the backend service**
   ```bash
   pixi run backend
   ```

#### Logger Service Development

1. **Navigate to Logger directory**
   ```bash
   cd Logger
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the logger api**
   ```bash
   python api.py
   ```

#### Frontend Development

1. **Navigate to webapp directory**
   ```bash
   cd webapp
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

## Evaluation Criteria

Our system is designed to meet the hackathon evaluation criteria:

- **Coverage**: Logs all relevant agent steps (prompts, tool calls, retrievals, outputs, parameters, timestamps)
- **Replay Fidelity**: Matches original outputs or behavior through deterministic replay
- **UX Clarity**: Easy to navigate timeline and graph with click-through inspection
- **Compliance**: Policy checks and audit reports for enterprise requirements

## Dependencies & Environment Files

### Backend Dependencies
- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI applications
- **Pydantic**: Data validation using Python type annotations
- **Transformers**: Hugging Face transformers library for ML models
- **PyTorch**: Deep learning framework
- **Pixi**: Dependency management for Python

### Logger Service Dependencies
- **FastAPI**: Web framework
- **LangChain**: Framework for developing applications with LLMs
- **LangGraph**: Library for building stateful, multi-actor applications
- **LangChain OpenAI**: OpenAI integration for LangChain
- **Pydantic**: Data validation
- **PyPDF**: PDF processing library
- **Uvicorn**: ASGI server

### Frontend Dependencies
- **Next.js 15**: React framework for production
- **React 19**: UI library
- **React Flow**: Node-based editor for React
- **Tailwind CSS**: Utility-first CSS framework
- **TypeScript**: Typed JavaScript

### Environment Configuration

The system uses the following environment variables:

```bash
# CORS Configuration
CORS_ORIGINS=http://localhost:3000,*

# OpenAI API Key (for LangGraph agent)
OPENAI_API_KEY=your_openai_api_key_here
```

## API Endpoints

### Core Operations
- `POST /run` - Create a new agent run
- `POST /task` - Create a new task within a run
- `POST /step` - Add a step to a task
- `GET /runs` - List all runs
- `GET /run/{run_id}` - Get details of a specific run

### Compliance & Auditing
- `POST /compliance/check` - Check compliance for a run/task
- `GET /compliance/audit/{run_id}.csv` - Export compliance audit as CSV

### Replay & Debugging
- `POST /replay` - Replay and modify a specific step

## Team Member Credits

- Bryan Chen
- Jules Decaestecker
- Alice Devilder 
- Valentin Fontaine

### Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests to improve the project.

## Frameworks & Technologies

Our implementation leverages the recommended hackathon resources:

- **Agent Framework**: LangChain + LangGraph for agent orchestration and tracing hooks
- **Visualization**: React/Next.js + React Flow for interactive workflow graphs
- **Backend**: FastAPI for robust API development
- **Storage**: JSON-based traces with local file system for artifacts
- **Containerization**: Docker for easy deployment and scaling

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Built with ❤️ during Hack Nation Hackathon - VC Big Bets (Agents) Track**
 hack-nation-hackathon-2025
