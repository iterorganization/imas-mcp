"""
Contract tests for imas_codex/remote/executor.py.

These tests cover the key behavioural contracts for run_python_script(),
async_run_python_script(), and run_script_via_stdin() that MUST survive
any refactor of the remote execution layer.

ITER XDR context: SSH commands using ``echo <b64> | base64 -d | bash``
are flagged by Extended Detection & Response as potential malware
obfuscation.  Anti-base64 regression gates ensure this pattern never
returns.
"""

import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import imas_codex.remote.executor as executor
from imas_codex.remote.executor import (
    async_run_python_script,
    run_python_script,
    run_script_via_stdin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_SCRIPT = "import sys, json\ndata = json.load(sys.stdin)\nprint('ok')\n"


def _make_proc_result(returncode=0, stdout="ok", stderr=""):
    """Build a mock subprocess.CompletedProcess-like object."""
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


def _patch_script_resource(mock_files, content=_FAKE_SCRIPT):
    """Wire mock importlib.resources so read_text() returns *content*."""
    mock_resource = MagicMock()
    mock_resource.read_text.return_value = content
    mock_files.return_value.joinpath.return_value = mock_resource
    return mock_resource


# ---------------------------------------------------------------------------
# TestRunPythonScript — synchronous path
# ---------------------------------------------------------------------------


class TestRunPythonScript:
    """Contract tests for the synchronous run_python_script() function."""

    # ------------------------------------------------------------------ setup
    def setup_method(self):
        # Reset nice-level registry between tests
        executor._host_nice_levels.clear()
        executor._verified_hosts.clear()

    # ------------------------------------------------------------------ stdin JSON delivery

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_script_receives_json_on_stdin(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """input_data must be serialised as JSON and passed as input= to subprocess."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        input_data = {"shot": 12345, "signals": ["ip"]}
        run_python_script("test.py", input_data=input_data, ssh_host="remote-host")

        _, kwargs = mock_run.call_args
        # subprocess.run is called with keyword arg input=
        assert "input" in kwargs
        assert json.loads(kwargs["input"]) == input_data

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_none_input_data_sends_empty_object(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """When input_data=None, script receives '{}' on stdin."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", input_data=None, ssh_host="remote-host")

        _, kwargs = mock_run.call_args
        assert kwargs["input"] == "{}"

    # ------------------------------------------------------------------ stdout-only return

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_returns_stdout_only(self, mock_files, mock_run, mock_local, mock_ssh):
        """Only stdout is returned; stderr must not contaminate the output."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(
            stdout="{'result': 42}", stderr="libvaccess.so loaded\nwarning: foo"
        )

        result = run_python_script("test.py", ssh_host="remote-host")

        assert result == "{'result': 42}"

    # ------------------------------------------------------------------ non-zero exit raises

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_nonzero_exit_raises(self, mock_files, mock_run, mock_local, mock_ssh):
        """Non-zero exit code must raise subprocess.CalledProcessError."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(
            returncode=1, stdout="", stderr="boom"
        )

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            run_python_script("test.py", ssh_host="remote-host")

        assert exc_info.value.returncode == 1

    # ------------------------------------------------------------------ local execution

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_local_execution_skips_ssh(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """When is_local_host()=True, subprocess.run must NOT receive 'ssh'."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="local")

        result = run_python_script("test.py", ssh_host="localhost")

        cmd = mock_run.call_args[0][0]  # first positional arg = command list
        assert "ssh" not in cmd, f"Expected no SSH for local host, got: {cmd}"
        assert result == "local"

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_local_execution_no_ssh_health_check(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """_ensure_ssh_healthy_once must NOT be called for local execution."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", ssh_host="localhost")

        mock_ssh.assert_not_called()

    # ------------------------------------------------------------------ setup commands

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_setup_commands_prepended(self, mock_files, mock_run, mock_local, mock_ssh):
        """setup_commands must appear in the remote command before Python invocation."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        setup = ["module load imas", "source /etc/profile"]
        run_python_script("test.py", ssh_host="remote-host", setup_commands=setup)

        cmd = mock_run.call_args[0][0]
        # The remote command is the last element of the SSH arg list
        remote_cmd = cmd[-1]
        for sc in setup:
            assert sc in remote_cmd, (
                f"setup command {sc!r} missing from: {remote_cmd!r}"
            )
        # Setup commands should appear BEFORE python invocation
        assert remote_cmd.index("module load imas") < remote_cmd.index("python3")

    # ------------------------------------------------------------------ timeout wrapper

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_timeout_wrapper_applied(self, mock_files, mock_run, mock_local, mock_ssh):
        """Remote command must be wrapped with timeout N+5 for zombie prevention."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        timeout = 30
        run_python_script("test.py", ssh_host="remote-host", timeout=timeout)

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        expected_timeout = f"timeout {timeout + 5}"
        assert expected_timeout in remote_cmd, (
            f"Expected '{expected_timeout}' in remote command, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ nice level

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_nice_level_applied(self, mock_files, mock_run, mock_local, mock_ssh):
        """nice -n N must wrap the command when configured via _host_nice_levels."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        executor._host_nice_levels["remote-host"] = 10
        run_python_script("test.py", ssh_host="remote-host")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "nice -n 10" in remote_cmd, (
            f"Expected 'nice -n 10' in remote command, got: {remote_cmd!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_no_nice_when_not_configured(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """nice must NOT appear in the remote command when no nice level is configured."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        # No nice level registered for this host
        run_python_script("test.py", ssh_host="remote-host")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "nice" not in remote_cmd, (
            f"Expected no 'nice' in remote command, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ -S flag + site-packages

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_site_packages_with_dash_S(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """python3 -S flag must be present with site-packages restoration for python3."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", ssh_host="remote-host", python_command="python3")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        # python3 -S must appear in the remote command
        assert "python3 -S" in remote_cmd, (
            f"Expected 'python3 -S' in remote command, got: {remote_cmd!r}"
        )
        # site-packages restoration must be present
        assert "site-packages" in remote_cmd, (
            f"Expected site-packages restoration in remote command, got: {remote_cmd!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_custom_python_command_no_dash_S(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """Custom python_command (non-python3) must NOT use -S flag."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script(
            "test.py",
            ssh_host="remote-host",
            python_command="/opt/micromamba/envs/myenv/bin/python",
        )

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert " -S " not in remote_cmd and not remote_cmd.endswith(" -S"), (
            f"Expected no -S for custom python, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ SSH 255 invalidation

    @patch("imas_codex.remote.executor._invalidate_ssh_host")
    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_ssh_255_invalidates_host(
        self, mock_files, mock_run, mock_local, mock_ssh, mock_invalidate
    ):
        """Exit code 255 (SSH connection failure) must call _invalidate_ssh_host()."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(
            returncode=255, stdout="", stderr="ssh: connect to host failed"
        )

        with pytest.raises(subprocess.CalledProcessError):
            run_python_script("test.py", ssh_host="remote-host")

        mock_invalidate.assert_called_once_with("remote-host")

    @patch("imas_codex.remote.executor._invalidate_ssh_host")
    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_nonzero_non_255_does_not_invalidate(
        self, mock_files, mock_run, mock_local, mock_ssh, mock_invalidate
    ):
        """Non-255 non-zero exit codes must NOT invalidate the SSH host."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(
            returncode=1, stdout="", stderr="script error"
        )

        with pytest.raises(subprocess.CalledProcessError):
            run_python_script("test.py", ssh_host="remote-host")

        mock_invalidate.assert_not_called()

    # ------------------------------------------------------------------ SSH health check

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_ssh_health_check_called_for_remote(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """_ensure_ssh_healthy_once() must be called for remote host execution."""
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", ssh_host="remote-host")

        mock_ssh.assert_called_once_with("remote-host")

    # ------------------------------------------------------------------ anti-base64 regression gates

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_no_base64_in_remote_command(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """REGRESSION GATE: SSH commands must not use base64 encoding.

        ITER XDR (Extended Detection and Response) flags base64-encoded
        SSH commands as potential malware obfuscation.
        """
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", ssh_host="remote-host")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "base64" not in remote_cmd, (
            f"SSH command must not contain 'base64', got: {remote_cmd[:200]!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    @patch("importlib.resources.files")
    def test_no_base64_import_in_runner(
        self, mock_files, mock_run, mock_local, mock_ssh
    ):
        """REGRESSION GATE: The inline Python runner string must not import base64.

        The runner string is embedded in the SSH command. If it imports base64
        the XDR system flags the command as suspicious obfuscation.
        """
        _patch_script_resource(mock_files)
        mock_run.return_value = _make_proc_result(stdout="")

        run_python_script("test.py", ssh_host="remote-host")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "import base64" not in remote_cmd, (
            f"Runner must not contain 'import base64', got: {remote_cmd[:200]!r}"
        )


# ---------------------------------------------------------------------------
# TestAsyncRunPythonScript — asynchronous path
# ---------------------------------------------------------------------------


class TestAsyncRunPythonScript:
    """Contract tests for the async async_run_python_script() function."""

    def setup_method(self):
        executor._host_nice_levels.clear()
        executor._verified_hosts.clear()

    def _make_async_proc(self, returncode=0, stdout=b"ok", stderr=b""):
        """Build a mock asyncio.subprocess.Process-like object."""
        proc = MagicMock()
        proc.returncode = returncode
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        proc.stdout = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    # ------------------------------------------------------------------ stdin JSON delivery

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_script_receives_json_on_stdin(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """input_data must be serialised as JSON and passed to proc.communicate()."""
        _patch_script_resource(mock_files)
        input_data = {"shot": 99, "node": "\\ip"}
        proc = self._make_async_proc(stdout=json.dumps({"ok": True}).encode())
        mock_exec.return_value = proc

        await async_run_python_script(
            "test.py", input_data=input_data, ssh_host="remote-host"
        )

        communicated = proc.communicate.call_args[0][0]
        assert json.loads(communicated.decode()) == input_data

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_none_input_data_sends_empty_object(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """When input_data=None, communicate() receives b'{}'."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script(
            "test.py", input_data=None, ssh_host="remote-host"
        )

        communicated = proc.communicate.call_args[0][0]
        assert communicated == b"{}"

    # ------------------------------------------------------------------ stdout-only return

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_returns_stdout_only(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """Only stdout is returned; stderr must not contaminate the output."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc(
            stdout=b"structured json output",
            stderr=b"libvaccess warning\nsome noise",
        )
        mock_exec.return_value = proc

        result = await async_run_python_script("test.py", ssh_host="remote-host")

        assert result == "structured json output"

    # ------------------------------------------------------------------ non-zero exit raises

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_nonzero_exit_raises(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """Non-zero exit code must raise subprocess.CalledProcessError."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc(returncode=2, stderr=b"runtime error")
        mock_exec.return_value = proc

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            await async_run_python_script("test.py", ssh_host="remote-host")

        assert exc_info.value.returncode == 2

    # ------------------------------------------------------------------ local execution

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_local_execution_skips_ssh(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """Local execution path must NOT pass 'ssh' as command to subprocess_exec."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc(stdout=b"local-result")
        mock_exec.return_value = proc

        result = await async_run_python_script("test.py", ssh_host="localhost")

        cmd_args = mock_exec.call_args[0]
        assert "ssh" not in cmd_args, f"Expected no SSH for local, got: {cmd_args}"
        assert result == "local-result"

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_local_no_ssh_health_check(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """_ensure_ssh_healthy_once must NOT be called for local execution."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script("test.py", ssh_host="localhost")

        mock_ssh.assert_not_called()

    # ------------------------------------------------------------------ setup commands

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_setup_commands_prepended(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """setup_commands must appear in the remote command before Python."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        setup = ["module load imas", "export PYTHONPATH=/opt/imas/lib"]
        await async_run_python_script(
            "test.py", ssh_host="remote-host", setup_commands=setup
        )

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        for sc in setup:
            assert sc in remote_cmd, f"setup command {sc!r} missing from remote command"
        assert remote_cmd.index("module load imas") < remote_cmd.index("python3")

    # ------------------------------------------------------------------ timeout wrapper

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_timeout_wrapper_applied(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """Remote command must be wrapped with timeout N+5."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        timeout = 45
        await async_run_python_script(
            "test.py", ssh_host="remote-host", timeout=timeout
        )

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert f"timeout {timeout + 5}" in remote_cmd, (
            f"Expected 'timeout {timeout + 5}' in command, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ nice level

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_nice_level_applied(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """nice -n N must appear in the remote command when configured."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        executor._host_nice_levels["remote-host"] = 15
        await async_run_python_script("test.py", ssh_host="remote-host")

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert "nice -n 15" in remote_cmd, (
            f"Expected 'nice -n 15' in remote command, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ -S flag + site-packages

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_site_packages_with_dash_S(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """python3 -S must be present and site-packages must be restored."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script(
            "test.py", ssh_host="remote-host", python_command="python3"
        )

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert "python3 -S" in remote_cmd, (
            f"Expected 'python3 -S' in remote command, got: {remote_cmd!r}"
        )
        assert "site-packages" in remote_cmd, (
            "Expected site-packages restore in remote command"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_custom_python_command_no_dash_S(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """Custom python_command must NOT use -S flag."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script(
            "test.py",
            ssh_host="remote-host",
            python_command="/opt/conda/envs/myenv/bin/python",
        )

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert " -S " not in remote_cmd and not remote_cmd.endswith(" -S"), (
            f"Expected no -S for custom python, got: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ SSH 255 invalidation

    @patch("imas_codex.remote.executor._invalidate_ssh_host")
    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_ssh_255_invalidates_host(
        self, mock_files, mock_exec, mock_local, mock_ssh, mock_invalidate
    ):
        """Exit code 255 must trigger _invalidate_ssh_host() in async path."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc(returncode=255, stderr=b"connection failed")
        mock_exec.return_value = proc

        with pytest.raises(subprocess.CalledProcessError):
            await async_run_python_script("test.py", ssh_host="remote-host")

        mock_invalidate.assert_called_once_with("remote-host")

    @patch("imas_codex.remote.executor._invalidate_ssh_host")
    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_nonzero_non_255_does_not_invalidate(
        self, mock_files, mock_exec, mock_local, mock_ssh, mock_invalidate
    ):
        """Non-255 exit codes must NOT invalidate the SSH host in async path."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc(returncode=1, stderr=b"script error")
        mock_exec.return_value = proc

        with pytest.raises(subprocess.CalledProcessError):
            await async_run_python_script("test.py", ssh_host="remote-host")

        mock_invalidate.assert_not_called()

    # ------------------------------------------------------------------ SSH health check

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_ssh_health_check_called_for_remote(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """_ensure_ssh_healthy_once() must be called for remote async execution."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script("test.py", ssh_host="remote-host")

        mock_ssh.assert_called_once_with("remote-host")

    # ------------------------------------------------------------------ anti-base64 regression gates

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_no_base64_in_remote_command(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """REGRESSION GATE: Async SSH commands must not use base64 encoding.

        ITER XDR flags base64-encoded SSH commands as malware obfuscation.
        """
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script("test.py", ssh_host="remote-host")

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert "base64" not in remote_cmd, (
            f"Async SSH command must not contain 'base64', got: {remote_cmd[:200]!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("asyncio.create_subprocess_exec")
    @patch("importlib.resources.files")
    async def test_no_base64_import_in_runner(
        self, mock_files, mock_exec, mock_local, mock_ssh
    ):
        """REGRESSION GATE: The async inline runner must not import base64."""
        _patch_script_resource(mock_files)
        proc = self._make_async_proc()
        mock_exec.return_value = proc

        await async_run_python_script("test.py", ssh_host="remote-host")

        cmd_args = mock_exec.call_args[0]
        remote_cmd = cmd_args[-1]
        assert "import base64" not in remote_cmd, (
            "Async runner must not contain 'import base64'"
        )


# ---------------------------------------------------------------------------
# TestRunScriptViaStdin — existing non-base64 approach (all should PASS now)
# ---------------------------------------------------------------------------


class TestRunScriptViaStdin:
    """Contract tests for run_script_via_stdin() — the existing non-base64 approach.

    These tests should all PASS now because run_script_via_stdin() already
    uses stdin delivery without base64. They serve as the reference model
    for how run_python_script() should work after the refactor.
    """

    def setup_method(self):
        executor._host_nice_levels.clear()
        executor._verified_hosts.clear()

    # ------------------------------------------------------------------ local execution

    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("subprocess.run")
    def test_local_execution(self, mock_run, mock_local):
        """Local execution passes script as stdin to interpreter directly."""
        mock_run.return_value = _make_proc_result(stdout="hello")

        result = run_script_via_stdin("print('hello')", interpreter="python3")

        cmd = mock_run.call_args[0][0]
        assert "ssh" not in cmd, f"Expected no SSH, got: {cmd}"
        assert mock_run.call_args[1].get("input") == "print('hello')"
        assert result == "hello"

    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("subprocess.run")
    def test_local_bash_uses_bash_s(self, mock_run, mock_local):
        """Local bash execution uses ['bash', '-s']."""
        mock_run.return_value = _make_proc_result(stdout="")

        run_script_via_stdin("echo hi", interpreter="bash")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["bash", "-s"], f"Expected ['bash', '-s'], got {cmd}"

    @patch("imas_codex.remote.executor.is_local_host", return_value=True)
    @patch("subprocess.run")
    def test_local_python_uses_python3_dash(self, mock_run, mock_local):
        """Local python3 execution uses ['python3', '-']."""
        mock_run.return_value = _make_proc_result(stdout="")

        run_script_via_stdin("import sys", interpreter="python3")

        cmd = mock_run.call_args[0][0]
        assert cmd == ["python3", "-"], f"Expected ['python3', '-'], got {cmd}"

    # ------------------------------------------------------------------ remote execution

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_remote_execution_uses_ssh(self, mock_run, mock_local, mock_ssh):
        """Remote execution must use SSH with script piped via stdin."""
        mock_run.return_value = _make_proc_result(stdout="remote-result")

        result = run_script_via_stdin(
            "echo remote", ssh_host="remote-host", interpreter="bash"
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ssh", f"Expected ssh command, got: {cmd}"
        assert cmd[1] == "-T"
        assert cmd[2] == "remote-host"
        # Script delivered via stdin, NOT embedded in the command
        assert "echo remote" not in cmd[-1], (
            "Script should be on stdin, not in the SSH command string"
        )
        assert mock_run.call_args[1].get("input") is not None
        assert result == "remote-result"

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_remote_stdin_contains_script(self, mock_run, mock_local, mock_ssh):
        """The full script content must be passed as subprocess stdin."""
        mock_run.return_value = _make_proc_result(stdout="")
        script = "import sys\nprint(sys.version)\n"

        run_script_via_stdin(script, ssh_host="remote-host", interpreter="python3")

        assert mock_run.call_args[1]["input"] == script

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_remote_timeout_wrapper(self, mock_run, mock_local, mock_ssh):
        """Remote execution must wrap interpreter with timeout N+5."""
        mock_run.return_value = _make_proc_result(stdout="")

        timeout = 20
        run_script_via_stdin(
            "echo hi", ssh_host="remote-host", timeout=timeout, interpreter="bash"
        )

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert f"timeout {timeout + 5}" in remote_cmd, (
            f"Expected 'timeout {timeout + 5}' in remote command: {remote_cmd!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_remote_nice_level(self, mock_run, mock_local, mock_ssh):
        """Remote execution must apply nice level when configured."""
        mock_run.return_value = _make_proc_result(stdout="")

        executor._host_nice_levels["remote-host"] = 5
        run_script_via_stdin("echo hi", ssh_host="remote-host", interpreter="bash")

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "nice -n 5" in remote_cmd, (
            f"Expected 'nice -n 5' in remote command: {remote_cmd!r}"
        )

    # ------------------------------------------------------------------ no base64 (passes now)

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_no_base64_in_command(self, mock_run, mock_local, mock_ssh):
        """run_script_via_stdin must NEVER use base64 in the SSH command.

        This test PASSES now and must remain passing.  It validates the
        reference implementation that run_python_script() will adopt.
        """
        mock_run.return_value = _make_proc_result(stdout="")

        run_script_via_stdin(
            "print('test')", ssh_host="remote-host", interpreter="python3"
        )

        cmd = mock_run.call_args[0][0]
        remote_cmd = cmd[-1]
        assert "base64" not in remote_cmd, (
            f"run_script_via_stdin must not use base64, got: {remote_cmd!r}"
        )

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_ssh_health_checked(self, mock_run, mock_local, mock_ssh):
        """_ensure_ssh_healthy_once() must be called for remote stdin execution."""
        mock_run.return_value = _make_proc_result(stdout="")

        run_script_via_stdin("echo hi", ssh_host="remote-host")

        mock_ssh.assert_called_once_with("remote-host")

    @patch("imas_codex.remote.executor._ensure_ssh_healthy_once")
    @patch("imas_codex.remote.executor.is_local_host", return_value=False)
    @patch("subprocess.run")
    def test_nonzero_exit_raises_when_check(self, mock_run, mock_local, mock_ssh):
        """check=True must raise CalledProcessError on non-zero exit."""
        mock_run.return_value = _make_proc_result(returncode=1, stderr="error")

        with pytest.raises(subprocess.CalledProcessError):
            run_script_via_stdin("bad script", ssh_host="remote-host", check=True)
