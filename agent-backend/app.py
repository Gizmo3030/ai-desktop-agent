try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except Exception:
    # Allow the module to be imported in test environments where Flask isn't installed.
    Flask = None
    request = None
    jsonify = None
    CORS = None

try:
    from crewai import Crew, Process, Task
except Exception:
    # Provide placeholders so tests can monkeypatch `app.Crew` before use.
    Crew = None
    Process = None
    Task = None

from agents import supervisor_agent, vision_agent, general_agent, llm_manager
try:
    import litellm
except Exception:
    # Provide a minimal placeholder so tests can monkeypatch litellm.BadRequestError
    class _LitellmPlaceholder:
        class BadRequestError(Exception):
            pass
    litellm = _LitellmPlaceholder
from tasks import AgentTasks


def create_app():
    """Application factory for test-friendly imports.

    If Flask is not installed, this function will raise at call time. Tests can import
    this module without Flask by not calling create_app(), which lets monkeypatches run
    against Crew or other symbols.
    """
    if Flask is None:
        raise RuntimeError("Flask is not installed in this environment")

    app = Flask(__name__)
    CORS(app)  # Allow requests from the Electron app

    import logging
    logging.basicConfig(level=logging.INFO)

    # Attach the route to the app instance
    @app.route('/execute_task', methods=['POST'])
    def execute_task_route():
        return _execute_task()

    return app


# Keep a module-level convenience `app` when Flask is available so running the file
# directly behaves the same as before. If Flask isn't installed, `app` remains None.
app = None
if Flask is not None:
    app = create_app()

def _execute_task():
    # Use Flask request when available. This function is split out so tests can
    # call it directly by providing a fake `request` object if needed.
    if request is None:
        raise RuntimeError("Flask request object is not available in this environment")

    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required"}), 400

    prompt = data['prompt']
    
    tasks = AgentTasks()
    
    # Define the crew with the supervisor as the manager
    crew = Crew(
        agents=[supervisor_agent, vision_agent, general_agent],
        tasks=[
            Task(
                description=prompt,
                expected_output="The final answer from the most suitable specialist agent.",
                agent=supervisor_agent
            )
        ],
        process=Process.hierarchical,
        manager_llm=supervisor_agent.llm,
        verbose=True # CORRECTED: Changed from 2 to True
    )
    
    try:
        # TEST HOOK: if a client sends this exact prompt, force the ValueError path
        # so we can reproduce the "Invalid response from LLM call - None or empty." diagnostics.
        if prompt == "__force_value_error__":
            raise ValueError("Invalid response from LLM call - None or empty.")
        result = crew.kickoff()
        return jsonify({"result": str(result)})
    except litellm.BadRequestError as e:
        # If litellm complains about provider/model formats, retry with Ollama fallback.
        app.logger.exception("litellm.BadRequestError encountered: %s", e)
        try:
            # Build a fresh Crew for the retry to avoid mutated state from the failed run.
            # Temporarily swap each agent's llm to Ollama and clear manager tools (crew enforces manager must not have tools).
            orig_supervisor_tools = getattr(supervisor_agent, 'tools', None)
            orig_supervisor_llm = getattr(supervisor_agent, 'llm', None)
            orig_vision_llm = getattr(vision_agent, 'llm', None)
            orig_general_llm = getattr(general_agent, 'llm', None)
            try:
                supervisor_agent.tools = []
                supervisor_agent.llm = llm_manager.ollama_llm
                vision_agent.llm = llm_manager.ollama_llm
                general_agent.llm = llm_manager.ollama_llm

                retry_crew = Crew(
                    agents=[supervisor_agent, vision_agent, general_agent],
                    tasks=[
                        Task(
                            description=prompt,
                            expected_output="The final answer from the most suitable specialist agent.",
                            agent=supervisor_agent
                        )
                    ],
                    process=Process.hierarchical,
                    manager_llm=llm_manager.ollama_llm,
                    verbose=True
                )

                result = retry_crew.kickoff()
                return jsonify({"result": str(result), "note": "Retried with Ollama fallback"})
            finally:
                # Restore original agent state
                if orig_supervisor_tools is None:
                    delattr(supervisor_agent, 'tools')
                else:
                    supervisor_agent.tools = orig_supervisor_tools
                supervisor_agent.llm = orig_supervisor_llm
                vision_agent.llm = orig_vision_llm
                general_agent.llm = orig_general_llm
        except Exception as ee:
            app.logger.exception("Retry with Ollama also failed: %s", ee)
            return jsonify({"error": str(ee)}), 500
    except ValueError as e:
        # Provide extra diagnostics when crewai reports an invalid/empty LLM response
        app.logger.exception("ValueError during Crew.kickoff or LLM response processing: %s", e)
        try:
            # Inspect which LLMs are configured on the manager for debugging
            mgr = llm_manager
            info = {
                "has_openai": bool(getattr(mgr, 'openai_llm', None)),
                "has_gemini": bool(getattr(mgr, 'gemini_llm', None)),
                "has_ollama": bool(getattr(mgr, 'ollama_llm', None)),
            }
            app.logger.info("LLM manager configuration: %s", info)

            # Attempt a safe, synchronous health check on available LLMs (best-effort). Do not raise.
            health = {}
            for name in ("openai_llm", "gemini_llm", "ollama_llm"):
                llm_obj = getattr(mgr, name, None)
                if not llm_obj:
                    health[name] = "not_configured"
                    continue
                try:
                    # Try multiple call patterns safely
                    resp = None
                    if hasattr(llm_obj, 'call'):
                        # crewai-style call expects messages list
                        resp = llm_obj.call([{"role": "user", "content": "Health check: respond with OK"}], callbacks=[])
                    elif callable(llm_obj):
                        resp = llm_obj("Health check: respond with OK")
                    elif hasattr(llm_obj, 'generate'):
                        resp = llm_obj.generate([{"role": "user", "content": "Health check: respond with OK"}], callbacks=[])
                    health[name] = {
                        "ok": bool(resp),
                        "sample": str(resp)[:200] if resp else None,
                    }
                except Exception as he:
                    health[name] = {"error": str(he)}

            app.logger.info("LLM health check: %s", health)
        except Exception as diag_e:
            app.logger.exception("Diagnostics failed: %s", diag_e)

        # Return original message and the diagnostic summary to the caller for faster debugging
        return jsonify({"error": str(e), "diagnostics": {"llm_info": info, "llm_health": health}}), 500
    except Exception as e:
        # It's helpful to log the exception to the server logs for debugging
        app.logger.exception("Unhandled exception in /execute_task: %s", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)