## AI Desktop Agent

A small Electron + Flask project that runs local AI agents to perform tasks. The repo contains two main parts:

- `agent-backend/` — Flask-based Python server exposing an `/execute_task` endpoint and unit tests.
- `agent-frontend/` — Electron + React desktop client that talks to the backend.

This README explains how to install, run, and test both parts locally for development.

## AI Desktop Agent

A small Electron + Flask project that runs local AI agents to perform tasks. The repo contains two main parts:

- `agent-backend/` — Flask-based Python server exposing an `/execute_task` endpoint and unit tests.
- `agent-frontend/` — Electron + React desktop client that talks to the backend.

This README explains how to install, run, and test both parts locally for development.

## Prerequisites

- Node.js and npm (for the frontend)
- Python 3.10+ and pip (for the backend)
- Optional: an available LLM provider configured if you intend to run real agent workflows.

## Backend (agent-backend)

Quick steps to run the backend server for development:

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r agent-backend/requirements.txt
```

3. Run the server:

```bash
python agent-backend/app.py
```

The server listens on port 5000 by default and exposes POST /execute_task which expects a JSON body with `prompt`.

Run the unit tests:

```bash
cd agent-backend
pytest -q
```

Notes:
- The backend contains test-friendly factories and placeholders so tests can run in environments without Flask or LLM clients installed.
- If your environment lacks configured LLM clients, the server may fall back to diagnostics or Ollama if present.

API example (quick):

```bash
curl -s -X POST http://localhost:5000/execute_task \
	-H "Content-Type: application/json" \
	-d '{"prompt":"Summarize the current tasks."}'
```

## Frontend (agent-frontend)

Run the Electron + React client locally:

1. Install Node dependencies:

```bash
cd agent-frontend
npm install
```

2. Start the React development server (for UI development):

```bash
npm run react-start
```

3. Start the Electron app (desktop build/dev):

```bash
npm start
```

The frontend expects the backend to be reachable at the API paths used in the code (see `agent-frontend/src` for the exact URLs). If you run the Flask backend locally, ensure any CORS/origin settings match (the backend sets CORS in `agent-backend/app.py`).

## Configuration / Environment

- You can keep LLM credentials or provider toggles in environment variables. A simple `.env.example` might include:

```text
# .env.example
OPENAI_API_KEY=
OLLAMA_URL=
```

Place a real `.env` in `agent-backend/` (or configure your OS env) before running if you need real LLM access.

## Project structure

- `agent-backend/` — Flask app, agent orchestration, and tests
- `agent-frontend/` — Electron + React app

## Troubleshooting

- If tests fail due to missing packages, ensure the backend virtualenv is active and `pip install -r agent-backend/requirements.txt` has completed.
- For LLM-related errors, consult logs from the backend — the server includes extra diagnostics for LLM configuration and fallbacks.

## Contributing

Contributions are welcome. Open an issue or submit a pull request with a short description of your change.

## License

This project is licensed under the MIT License — see the `LICENSE` file in the repository root.

---

If you'd like, I can also add a brief API reference section, an `.env.example` file, or a small developer quick-start script.
