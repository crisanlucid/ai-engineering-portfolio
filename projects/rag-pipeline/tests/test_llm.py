"""Unit tests for llm.py — adapter routing and subprocess behaviour."""

import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proc(returncode: int = 0, stdout: str = "ok", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# get_adapter
# ---------------------------------------------------------------------------


class TestGetAdapter:
    def test_defaults_to_cli(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        from llm import CLIAdapter, get_adapter

        assert isinstance(get_adapter(), CLIAdapter)

    def test_cli_explicit(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "cli")
        from llm import CLIAdapter, get_adapter

        assert isinstance(get_adapter(), CLIAdapter)

    def test_agy_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "agy")
        from llm import AgyAdapter, get_adapter

        assert isinstance(get_adapter(), AgyAdapter)

    def test_anthropic_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("MODEL_API_KEY", "test-key")
        # AnthropicAdapter imports langchain_anthropic lazily — patch it
        with patch("llm.AnthropicAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from llm import get_adapter

            get_adapter()
            MockAdapter.assert_called_once_with(api_key="test-key")

    def test_google_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "google")
        monkeypatch.setenv("MODEL_API_KEY", "test-key")
        with patch("llm.GoogleAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from llm import get_adapter

            get_adapter()
            MockAdapter.assert_called_once_with(api_key="test-key")

    def test_unknown_provider_falls_back_to_cli(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        from llm import CLIAdapter, get_adapter

        assert isinstance(get_adapter(), CLIAdapter)


# ---------------------------------------------------------------------------
# CLIAdapter
# ---------------------------------------------------------------------------


class TestCLIAdapter:
    def test_complete_returns_stdout_on_success(self):
        from llm import CLIAdapter

        adapter = CLIAdapter()
        mock_proc = _make_proc(returncode=0, stdout="  hello world  ")
        with patch("subprocess.run", return_value=mock_proc):
            result = adapter.complete("sys", "user")
        assert result == "hello world"

    def test_complete_returns_error_on_failure(self):
        from llm import CLIAdapter

        adapter = CLIAdapter()
        mock_proc = _make_proc(returncode=1, stderr="something went wrong")
        with patch("subprocess.run", return_value=mock_proc):
            result = adapter.complete("sys", "user")
        assert result.startswith("Error:")
        assert "something went wrong" in result

    def test_complete_passes_combined_prompt(self):
        from llm import CLIAdapter

        adapter = CLIAdapter()
        mock_proc = _make_proc(returncode=0, stdout="done")
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            adapter.complete("SYSTEM", "USER")
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert "-p" in cmd
            combined = cmd[cmd.index("-p") + 1]
            assert "SYSTEM" in combined
            assert "USER" in combined

    def test_complete_strips_sensitive_env_vars(self):
        from llm import CLIAdapter

        adapter = CLIAdapter()
        mock_proc = _make_proc(returncode=0, stdout="done")
        with (
            patch("subprocess.run", return_value=mock_proc) as mock_run,
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "secret"}),
        ):
            adapter.complete("sys", "user")
            env = mock_run.call_args[1]["env"]
            assert "ANTHROPIC_API_KEY" not in env


# ---------------------------------------------------------------------------
# AgyAdapter
# ---------------------------------------------------------------------------


class TestAgyAdapter:
    def test_complete_returns_stdout_on_success(self):
        from llm import AgyAdapter

        adapter = AgyAdapter()
        mock_proc = _make_proc(returncode=0, stdout="  agy response  ")
        with patch("subprocess.run", return_value=mock_proc):
            result = adapter.complete("sys", "user")
        assert result == "agy response"

    def test_complete_returns_error_on_failure(self):
        from llm import AgyAdapter

        adapter = AgyAdapter()
        mock_proc = _make_proc(returncode=1, stderr="agy crashed")
        with patch("subprocess.run", return_value=mock_proc):
            result = adapter.complete("sys", "user")
        assert "Error:" in result
        assert "agy crashed" in result

    def test_complete_uses_print_flag(self):
        from llm import AgyAdapter

        adapter = AgyAdapter()
        mock_proc = _make_proc(returncode=0, stdout="done")
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            adapter.complete("sys", "user")
            cmd = mock_run.call_args[0][0]
            assert "--print" in cmd

    def test_complete_includes_skip_permissions_flag(self):
        from llm import AgyAdapter

        adapter = AgyAdapter()
        mock_proc = _make_proc(returncode=0, stdout="done")
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            adapter.complete("sys", "user")
            cmd = mock_run.call_args[0][0]
            assert "--dangerously-skip-permissions" in cmd

    def test_custom_model_is_passed(self):
        from llm import AgyAdapter

        adapter = AgyAdapter(model="gemini-pro")
        mock_proc = _make_proc(returncode=0, stdout="done")
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            adapter.complete("sys", "user")
            cmd = mock_run.call_args[0][0]
            assert "--model" in cmd
            assert cmd[cmd.index("--model") + 1] == "gemini-pro"
