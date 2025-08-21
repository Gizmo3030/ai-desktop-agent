import json
import pytest


def test_execute_task_requires_prompt(monkeypatch, tmp_path):
    from app import app

    client = app.test_client()

    # Missing prompt
    resp = client.post('/execute_task', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert 'error' in data and 'Prompt is required' in data['error']


def test_execute_task_success(monkeypatch):
    from app import app

    # Mock Crew.kickoff to return a predictable value
    class DummyCrew:
        def __init__(self, *args, **kwargs):
            pass

        def kickoff(self):
            return {'ok': True, 'answer': 'dummy'}

    monkeypatch.setattr('app.Crew', lambda *a, **k: DummyCrew())

    client = app.test_client()
    resp = client.post('/execute_task', json={'prompt': 'hello'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'result' in data


def test_ollama_fallback_retry(monkeypatch):
    """First kickoff raises litellm.BadRequestError, second kickoff (retry) succeeds and returns note."""
    import litellm
    from app import app

    # Some litellm exceptions require constructor args; replace the BadRequestError
    # in the `app` module with a simple Exception subclass for testing.
    class MyBadRequestError(Exception):
        pass

    monkeypatch.setattr('app.litellm.BadRequestError', MyBadRequestError)

    call_state = {"count": 0}

    class FirstCrew:
        def kickoff(self):
            raise litellm.BadRequestError("provider/model format error")

    class SecondCrew:
        def kickoff(self):
            return {'ok': True, 'note': 'Retried with Ollama fallback'}

    def crew_factory(*args, **kwargs):
        # First creation -> return object that fails, second -> succeeds
        call_state['count'] += 1
        if call_state['count'] == 1:
            return FirstCrew()
        return SecondCrew()

    monkeypatch.setattr('app.Crew', crew_factory)

    client = app.test_client()
    resp = client.post('/execute_task', json={'prompt': 'trigger fallback'})
    assert resp.status_code == 200
    data = resp.get_json()
    # The retry path adds a 'note' in the response per app.py
    assert 'note' in data or (isinstance(data.get('result'), str) and 'Retried' in data.get('result'))


def test_value_error_diagnostics(monkeypatch):
    """Simulate Crew.kickoff raising ValueError to trigger diagnostics and ensure diagnostics are present."""
    from app import app

    class BrokenCrew:
        def kickoff(self):
            raise ValueError("Invalid response from LLM call - None or empty.")

    monkeypatch.setattr('app.Crew', lambda *a, **k: BrokenCrew())

    client = app.test_client()
    resp = client.post('/execute_task', json={'prompt': 'cause value error'})
    assert resp.status_code == 500
    data = resp.get_json()
    assert 'error' in data
    assert 'diagnostics' in data
    diag = data['diagnostics']
    assert 'llm_info' in diag and 'llm_health' in diag
