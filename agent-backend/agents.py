import os
import requests
import time
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Tests may not have python-dotenv; that's fine.
    load_dotenv = None

try:
    from crewai import Agent
except Exception:
    Agent = None

# Make LLM client imports optional so module import works in lightweight test envs
try:
    from langchain_ollama import OllamaLLM
except Exception:
    OllamaLLM = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

from tools import ScreenCaptureTool

# Centralized LLM Manager
class LLMManager:
    def __init__(self):
        # Read keys from env
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        # Create OpenAI LLM only if a key is provided
        self.openai_llm = None
        if openai_key:
            try:
                self.openai_llm = ChatOpenAI(
                    model="gpt-5",
                    api_key=openai_key
                )
            except Exception as e:
                # If the provider wrapper fails to initialize, log and continue with fallback
                print(f"Failed to initialize OpenAI LLM: {e}")
                self.openai_llm = None

        # Create Gemini LLM only if a key is provided
        self.gemini_llm = None
        if gemini_key:
            try:
                # Allow overriding the model string via env; use a litellm-style default that
                # includes the expected provider namespace. If your litellm integration expects
                # a different format, set GEMINI_MODEL in your environment.
                gemini_model = os.getenv("GEMINI_MODEL", "models/google/gemini-2.5-flash")
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=gemini_key
                )
            except Exception as e:
                # litellm or the google wrapper may require an explicit provider or different model string.
                # Provide a more actionable error message pointing to litellm provider docs.
                print(f"Failed to initialize Gemini LLM: {e}")
                print("If using litellm, try setting GEMINI_MODEL to a provider-prefixed model string like 'models/google/gemini-2.5-flash' or consult https://docs.litellm.ai/docs/providers')")
                self.gemini_llm = None

        # Always instantiate Ollama LLM (local fallback). Use a sensible default for base_url
        # so a local Ollama at http://localhost:11434 will be used if OLLAMA_BASE_URL isn't set.
        ollama_model = os.getenv("OLLAMA_MODEL") or "llama2"
        # Some clients (litellm/langchain integrations) expect a provider-prefixed model
        # string (for example: 'ollama/gpt-oss:latest'). If the env value lacks a '/'
        # provider prefix, prepend 'ollama/' so the provider is explicit and avoids
        # litellm.BadRequestError about a missing provider.
        if "/" not in ollama_model:
            ollama_model = f"ollama/{ollama_model}"

        # Create Ollama client only if the class is available
        raw_ollama = None
        if OllamaLLM is not None:
            try:
                raw_ollama = OllamaLLM(
                    model=ollama_model,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
            except Exception as e:
                print(f"Failed to initialize Ollama LLM: {e}")
                raw_ollama = None

        # Adapter: some callers (crewai/litellm wrappers) expect certain LLM attributes
        # like `supports_stop_words`. Wrap the OllamaLLM so missing attributes don't crash.
        class OllamaAdapter:
            def __init__(self, inner):
                self._inner = inner

            def supports_stop_words(self) -> bool:
                return False

            def __getattr__(self, name):
                if self._inner is None:
                    raise AttributeError(name)
                return getattr(self._inner, name)

            def __repr__(self):
                return f"OllamaAdapter({repr(self._inner)})"

        # Sanitizer wrapper: normalize responses to plain text expected by tools.
        class SanitizingLLM:
            def __init__(self, inner):
                self._inner = inner

            def _sanitize_text(self, text: str) -> str:
                if not isinstance(text, str):
                    text = str(text)
                import re
                text = re.sub(r"(?s)Thought:.*?Final Answer:\s*", "", text)
                text = re.sub(r"(?s)Thought:.*", "", text)
                text = re.sub(r"(?s)Action:.*", "", text)
                text = re.sub(r"```.*?```", "", text)
                try:
                    import json
                    stripped = text.strip()
                    if (stripped.startswith('{') or stripped.startswith('[')):
                        parsed = json.loads(stripped)
                        for key in ('result','text','output','answer'):
                            if isinstance(parsed, dict) and key in parsed:
                                return str(parsed[key])
                        return json.dumps(parsed)
                except Exception:
                    pass
                text = re.sub(r"\s+", " ", text).strip()
                return text or ""

            def generate(self, *args, **kwargs):
                try:
                    if os.getenv('OLLAMA_FORCE_HTTP', '0') == '1':
                        prompt_text = None
                        if args:
                            first = args[0]
                            if isinstance(first, (list, tuple)) and first:
                                prompt_text = first[0]
                            elif isinstance(first, str):
                                prompt_text = first
                        if not prompt_text:
                            prompt_text = kwargs.get('prompt') or kwargs.get('input') or ''
                        fb = self._http_fallback(prompt_text)
                        sanitized_fb = self._sanitize_text(fb)
                        if sanitized_fb:
                            return sanitized_fb
                except Exception as e:
                    print('Error during forced HTTP fallback:', e)

                # Normalize arguments so the wrapped LLM receives a list[str] when appropriate.
                call_args = None
                call_kwargs = dict(kwargs)
                if args:
                    first = args[0]
                    # If a single string was passed, convert to list[str]
                    if isinstance(first, str):
                        call_args = ([first],) + args[1:]
                    elif isinstance(first, (list, tuple)):
                        # If list of message dicts, extract their content
                        if len(first) and isinstance(first[0], dict):
                            text = ' '.join(str(m.get('content', '')) for m in first)
                            call_args = ([text],) + args[1:]
                        else:
                            call_args = (list(first),) + args[1:]
                    else:
                        call_args = ([str(first)],) + args[1:]
                else:
                    # No positional args: look for prompt-like kwargs and normalize
                    prompt_val = None
                    for key in ('prompts', 'prompt', 'input'):
                        if key in call_kwargs:
                            prompt_val = call_kwargs.pop(key)
                            break
                    if isinstance(prompt_val, str):
                        call_args = ([prompt_val],)
                    elif isinstance(prompt_val, (list, tuple)):
                        # if list contains dicts, extract content
                        if prompt_val and isinstance(prompt_val[0], dict):
                            text = ' '.join(str(m.get('content', '')) for m in prompt_val)
                            call_args = ([text],)
                        else:
                            call_args = (list(prompt_val),)
                    else:
                        call_args = ([],)

                out = None
                # Build a single text prompt from normalized args for flexible calling
                try:
                    text = call_args[0][0] if call_args and call_args[0] else ''
                except Exception:
                    text = ''

                if hasattr(self._inner, 'generate'):
                    # Try several call shapes to support different client wrappers
                    call_attempts = [
                        lambda: self._inner.generate(prompts=[text], **call_kwargs),
                        lambda: self._inner.generate([text], **call_kwargs),
                        lambda: self._inner.generate(text, **call_kwargs),
                        lambda: self._inner.generate(prompt=text, **call_kwargs),
                    ]
                    for attempt in call_attempts:
                        try:
                            out = attempt()
                            break
                        except Exception:
                            out = None
                # As a last resort, if the inner is callable, try calling it directly
                if out is None and callable(self._inner):
                    try:
                        out = self._inner(text)
                    except Exception:
                        out = None

                try:
                    if hasattr(out, 'generations'):
                        gens = out.generations
                        texts = []
                        for g in gens:
                            for choice in g:
                                texts.append(getattr(choice, 'text', str(choice)))
                        joined = ' '.join(texts)
                        sanitized = self._sanitize_text(joined)
                        if sanitized:
                            return sanitized
                except Exception:
                    pass

                if out is not None:
                    fallback_candidate = self._sanitize_text(str(out))
                    if fallback_candidate:
                        return fallback_candidate

                try:
                    prompt_text = None
                    if args:
                        first = args[0]
                        if isinstance(first, (list, tuple)) and first:
                            prompt_text = first[0]
                        elif isinstance(first, str):
                            prompt_text = first
                    if not prompt_text:
                        prompt_text = kwargs.get('prompt') or kwargs.get('input') or ''
                    fb = self._http_fallback(prompt_text)
                    return self._sanitize_text(fb)
                except Exception as e:
                    print(f"OLLAMA HTTP fallback failed: {e}")
                    return ""

            def _http_fallback(self, prompt: str) -> str:
                base = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
                model = os.getenv('OLLAMA_MODEL', 'ollama/llama2')
                endpoints = ["/api/generate", "/api/chat", "/v1/generate"]
                last_text = ""
                max_attempts = 3
                for ep in endpoints:
                    url = base.rstrip('/') + ep
                    if ep.endswith('/chat'):
                        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
                    else:
                        payload = {"model": model, "prompt": prompt, "stream": False}

                    attempt = 0
                    while attempt < max_attempts:
                        attempt += 1
                        try:
                            resp = requests.post(url, json=payload, timeout=20)
                        except Exception as e:
                            print(f"OLLAMA HTTP request error for {url}: {e}")
                            break

                        try:
                            body = resp.text
                        except Exception:
                            body = resp.text if hasattr(resp, 'text') else ''

                        try:
                            data = resp.json()

                            # If the server returns a partial response (e.g., done: false), retry briefly
                            done_flag = None
                            response_text = None
                            if isinstance(data, dict):
                                done_flag = data.get('done')
                                response_text = data.get('response')

                            if response_text:
                                return response_text

                            if done_flag is False or (isinstance(response_text, str) and response_text.strip() == ""):
                                if attempt < max_attempts:
                                    time.sleep(0.5)
                                    continue

                            def collect_strings(x):
                                out = []
                                if isinstance(x, str):
                                    out.append(x)
                                elif isinstance(x, dict):
                                    for v in x.values():
                                        out.extend(collect_strings(v))
                                elif isinstance(x, list):
                                    for v in x:
                                        out.extend(collect_strings(v))
                                return out

                            strs = collect_strings(data)
                            if strs:
                                candidate = max(strs, key=len)
                                if candidate:
                                    return candidate

                            import json as _json
                            last_text = _json.dumps(data)
                        except Exception:
                            if body:
                                return body
                        break
                return last_text

            def __call__(self, *args, **kwargs):
                # Support being called with a messages list (crewai-style) or a simple prompt
                if args and isinstance(args[0], (list, tuple)):
                    # Extract content from message list
                    msgs = args[0]
                    try:
                        text = ' '.join(m.get('content', '') if isinstance(m, dict) else str(m) for m in msgs)
                    except Exception:
                        text = str(msgs)
                    # Try to call inner with a single prompt
                    try:
                        # Prefer .call if available
                        if hasattr(self._inner, 'call'):
                            raw = None
                            try:
                                raw = self._inner.call(msgs, **kwargs)
                            except Exception:
                                raw = None
                            sanitized = self._sanitize_text(str(raw)) if raw is not None else ''
                            if sanitized:
                                return sanitized
                            # Try HTTP fallback using the extracted text
                            try:
                                fb = self._http_fallback(text)
                                fb_s = self._sanitize_text(fb)
                                if fb_s:
                                    return fb_s
                            except Exception:
                                pass
                        # Otherwise delegate to generate (or fallback if empty)
                        gen = self.generate([{'role':'user','content': text}], **kwargs)
                        if gen:
                            return gen
                        return self.generate(text, **kwargs)
                    except Exception:
                        return self.generate(text, **kwargs)

                try:
                    out = None
                    try:
                        out = self._inner(*args, **kwargs)
                    except Exception:
                        out = None
                    sanitized = self._sanitize_text(str(out)) if out is not None else ''
                    if sanitized:
                        return sanitized
                    # Try generate fallback which itself will attempt HTTP fallback
                    return self.generate(*args, **kwargs)
                except Exception:
                    return self.generate(*args, **kwargs)

            def call(self, messages, callbacks=None, **kwargs):
                # Accept crewai-style messages list: [{'role': 'user', 'content': '...'}, ...]
                try:
                    if isinstance(messages, (list, tuple)):
                        text = ' '.join(m.get('content', '') if isinstance(m, dict) else str(m) for m in messages)
                    else:
                        text = str(messages)
                except Exception:
                    text = str(messages)

                # Try several invocation patterns against the inner client
                try_patterns = []
                if hasattr(self._inner, 'call'):
                    # Some clients implement call(messages)
                    try_patterns.append(lambda: self._inner.call(messages, callbacks=callbacks, **kwargs))
                if hasattr(self._inner, 'generate'):
                    try_patterns.append(lambda: self._inner.generate([text], **kwargs))
                    try_patterns.append(lambda: self._inner.generate(text, **kwargs))
                    try_patterns.append(lambda: self._inner.generate(prompt=text, **kwargs))
                if callable(self._inner):
                    try_patterns.append(lambda: self._inner(text))

                out = None
                for pat in try_patterns:
                    try:
                        out = pat()
                        break
                    except Exception:
                        out = None

                if out is None:
                    # As a last resort, try the HTTP fallback using the message content so
                    # local Ollama endpoints that don't wire through the client wrapper
                    # still provide a response. This prevents callers (like crewai) from
                    # receiving an empty string and treating the LLM as unhealthy.
                    try:
                        # Derive a prompt string from the messages argument
                        if isinstance(messages, (list, tuple)):
                            text = ' '.join(m.get('content', '') if isinstance(m, dict) else str(m) for m in messages)
                        else:
                            text = str(messages)
                        fb = self._http_fallback(text)
                        if fb:
                            sanitized_fb = self._sanitize_text(fb)
                            print(f"[SanitizingLLM.call] HTTP fallback produced: {repr(sanitized_fb)[:200]}")
                            return sanitized_fb
                    except Exception:
                        pass
                    # If we still have nothing, return a non-empty safe fallback so callers
                    # (like crewai) don't treat the response as invalid/None and raise.
                    try:
                        print(f"[SanitizingLLM.call] empty output from inner LLM; messages sample: {repr(text)[:200]}")
                    except Exception:
                        pass
                    return "No response from LLM (fallback)"

                # Sanitize result
                try:
                    if hasattr(out, 'generations'):
                        gens = out.generations
                        texts = []
                        for g in gens:
                            for choice in g:
                                texts.append(getattr(choice, 'text', str(choice)))
                        return self._sanitize_text(' '.join(texts))
                except Exception:
                    pass

                return self._sanitize_text(str(out))

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __repr__(self):
                return f"SanitizingLLM({repr(self._inner)})"

        # Wrap the adapter with the sanitizer so callers get normalized strings
        self.ollama_llm = SanitizingLLM(OllamaAdapter(raw_ollama))

        # Diagnostic: print which LLMs were successfully initialized (do not print keys)
        try:
            print(
                "LLMManager initialized ->",
                f"OpenAI={'yes' if self.openai_llm else 'no'},",
                f"Gemini={'yes' if self.gemini_llm else 'no'},",
                f"Ollama={'yes' if self.ollama_llm else 'no'},",
            )
            print(
                "Configured models ->",
                f"GEMINI_MODEL={os.getenv('GEMINI_MODEL')},",
                f"OLLAMA_MODEL={ollama_model},",
                f"OLLAMA_BASE_URL={'set' if os.getenv('OLLAMA_BASE_URL') else 'default'},",
            )
        except Exception:
            pass

llm_manager = LLMManager()

# --- AGENT DEFINITIONS ---

# Provide a small factory that uses the real crewai.Agent when available,
# otherwise returns a lightweight placeholder object usable in tests.
def _make_agent(**kwargs):
    if Agent is not None:
        return Agent(**kwargs)
    # lightweight placeholder
    class _Placeholder:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def __repr__(self):
            return f"PlaceholderAgent(role={getattr(self, 'role', None)})"
    return _Placeholder(**kwargs)


# The Supervisor agent acts as the router
supervisor_agent = _make_agent(
    role="Orchestrator Agent",
    goal=(
        "Analyze incoming user requests and delegate them to the appropriate specialist agent. "
        "If the request involves analyzing the screen or visuals, delegate to the VisionAgent. "
        "For all other general queries, coding help, or web surfing questions, delegate to the GeneralAgent."
    ),
    backstory=(
        "You are the master controller of a multi-agent AI system. Your job is to ensure that "
        "every user request is handled by the most qualified agent, ensuring efficiency and accuracy. "
        "When delegating or using tools, provide only concrete string values for tool arguments. "
        "Do NOT output JSON schemas, tool definitions, or nested dictionaries — return only the exact "
        "text values required by the tool inputs."
    ),
    # Preferred order for routing: Gemini -> OpenAI -> Ollama
    llm=llm_manager.gemini_llm or llm_manager.openai_llm or llm_manager.ollama_llm,
    allow_delegation=True,
    verbose=True,
)

# Specialist agent for visual tasks
vision_agent = _make_agent(
    role="Visual Analysis Specialist",
    goal="Analyze the content of images and the user's screen to answer questions and assist with visual tasks.",
    backstory=(
        "You are an expert in computer vision and multimodal analysis. You can interpret what's on a screen, "
        "describe images, and help users with what they are seeing, whether it's a game, a website, or code."
    ),
    llm=llm_manager.openai_llm, # GPT-4o is excellent for vision
    tools=[ScreenCaptureTool()],
    allow_delegation=False,
    verbose=True,
)

# Specialist agent for general tasks
general_agent = _make_agent(
    role="General Purpose Assistant",
    goal="Answer general questions, provide coding assistance, and help with web-related queries without needing visual input.",
    backstory=(
        "You are a knowledgeable and versatile AI assistant. You can write code, explain concepts, and provide "
        "information on a wide range of topics. You prioritize user privacy and efficiency by using local models when possible."
    ),
    # Preferred order for general tasks: OpenAI -> Gemini -> Ollama
    llm=llm_manager.openai_llm or llm_manager.gemini_llm or llm_manager.ollama_llm,
    allow_delegation=False,
    verbose=True,
)