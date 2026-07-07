import os
import shutil
import subprocess
from abc import ABC, abstractmethod

CLAUDE_PATH = shutil.which("claude") or "/home/oem/.local/bin/claude"
AGY_PATH = shutil.which("agy") or "/home/oem/.local/bin/agy"


class LLMAdapter(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class AnthropicAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        from langchain_anthropic import ChatAnthropic

        self._llm = ChatAnthropic(model=model, anthropic_api_key=api_key)  # type: ignore[call-arg]

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(response.content)


class GoogleAdapter(LLMAdapter):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        return str(response.content)


class CLIAdapter(LLMAdapter):
    """Uses the local Claude Code CLI session — no API key required."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._model = model

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "CLAUDECODE",
                "CLAUDE_CODE_SESSION_ID",
                "CLAUDE_CODE_SSE_PORT",
                "CLAUDE_CODE_ENTRYPOINT",
                "AI_AGENT",
            }
        }
        result = subprocess.run(
            [CLAUDE_PATH, "-p", prompt, "--model", self._model, "--settings", "{}"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd="/tmp",
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()


class AgyAdapter(LLMAdapter):
    """Uses the local Antigravity (agy) CLI session."""

    def __init__(self, model: str = "gemini-3.5-flash"):
        self._model = model

    def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        env = os.environ.copy()
        result = subprocess.run(
            [AGY_PATH, "--print", prompt, "--model", self._model, "--dangerously-skip-permissions"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd="/tmp",
        )
        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"
        return result.stdout.strip()


def get_adapter() -> LLMAdapter:
    provider = os.getenv("LLM_PROVIDER", "cli").lower()
    api_key = os.getenv("MODEL_API_KEY", "")

    if provider == "anthropic":
        return AnthropicAdapter(api_key=api_key)
    if provider == "google":
        return GoogleAdapter(api_key=api_key)
    if provider == "agy":
        return AgyAdapter()
    return CLIAdapter()
