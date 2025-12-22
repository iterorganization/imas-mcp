"""Tests for the CommandSandbox read-only command validation."""

import pytest

from imas_codex.discovery.sandbox import CommandSandbox


@pytest.fixture
def sandbox() -> CommandSandbox:
    """Create a sandbox instance for testing."""
    return CommandSandbox()


class TestBasicCommands:
    """Test basic command allowlist."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls",
            "ls -la",
            "ls /etc",
            "find /home -name '*.py'",
            "cat /etc/os-release",
            "head -n 10 /etc/passwd",
            "tail -f /var/log/syslog",
            "grep -r 'pattern' /path",
            "which python3",
            "tree /opt",
            "env",
            "hostname",
            "whoami",
            "uname -a",
            "python3 --version",
            "pip list",
        ],
    )
    def test_allowed_basic_commands(self, sandbox: CommandSandbox, command: str):
        """Basic read-only commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "rm file.txt",
            "rm -rf /",
            "mv source dest",
            "cp file1 file2",
            "chmod 755 script.sh",
            "chown user:group file",
            "sudo ls",
            "su -",
            "dd if=/dev/zero of=/dev/sda",
            "mkdir new_dir",
            "rmdir old_dir",
            "touch newfile",
            "ln -s target link",
            "kill 1234",
            "pkill python",
            "reboot",
            "shutdown -h now",
        ],
    )
    def test_forbidden_commands(self, sandbox: CommandSandbox, command: str):
        """Destructive commands should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"


class TestCommandChaining:
    """Test command chaining with ;, &&, ||."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls /etc; cat /etc/os-release",
            "which python3; python3 --version",
            "ls; pwd; whoami",
            "cat /etc/passwd; cat /etc/group; cat /etc/hosts",
        ],
    )
    def test_semicolon_chaining_allowed(self, sandbox: CommandSandbox, command: str):
        """Command chaining with semicolon should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "which python3 && python3 --version",
            "test -f /etc/passwd && cat /etc/passwd",
            "ls /nonexistent && echo 'found'",
        ],
    )
    def test_and_chaining_allowed(self, sandbox: CommandSandbox, command: str):
        """Command chaining with && should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "cat /file || echo 'not found'",
            "which rg || which grep",
            "test -d /path || echo 'missing'",
        ],
    )
    def test_or_chaining_allowed(self, sandbox: CommandSandbox, command: str):
        """Command chaining with || should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "ls; rm file",
            "cat /etc/passwd; sudo cat /etc/shadow",
            "echo test && rm -rf /",
            "ls || kill 1234",
        ],
    )
    def test_chained_forbidden_commands_blocked(
        self, sandbox: CommandSandbox, command: str
    ):
        """Forbidden commands in chains should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"


class TestPipes:
    """Test pipe operations."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls | head",
            "cat file | grep pattern",
            "ps aux | grep python",
            "find /path -type f | wc -l",
            "rpm -qa | grep python",
            "env | sort",
            "cat /etc/passwd | awk -F: '{print $1}'",
        ],
    )
    def test_pipes_allowed(self, sandbox: CommandSandbox, command: str):
        """Pipe operations should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "cat file | rm",
            "ls | sudo tee /etc/file",
        ],
    )
    def test_pipes_with_forbidden_blocked(self, sandbox: CommandSandbox, command: str):
        """Pipes with forbidden commands should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"


class TestPackageManagers:
    """Test package manager query commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "rpm -qa",
            "rpm -qi python3",
            "rpm -ql python3",
            "rpm -qa | grep python",
        ],
    )
    def test_rpm_queries_allowed(self, sandbox: CommandSandbox, command: str):
        """RPM query commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "dnf list installed",
            "dnf info python3",
            "dnf search numpy",
            "dnf repolist",
        ],
    )
    def test_dnf_queries_allowed(self, sandbox: CommandSandbox, command: str):
        """DNF query commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "dnf install package",
            "dnf remove package",
            "dnf update",
        ],
    )
    def test_dnf_modifications_blocked(self, sandbox: CommandSandbox, command: str):
        """DNF modification commands should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"


class TestEnvironmentModules:
    """Test environment module commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "module avail",
            "module list",
            "module show python/3.9",
            "module whatis mpi",
            "module spider numpy",
            "module help gcc",
        ],
    )
    def test_module_queries_allowed(self, sandbox: CommandSandbox, command: str):
        """Module query commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "module load python/3.9",
            "module unload mpi",
            "module purge",
            "module swap gcc/9 gcc/11",
            "module save myenv",
            "module restore myenv",
        ],
    )
    def test_module_environment_allowed(self, sandbox: CommandSandbox, command: str):
        """Module environment commands are session-scoped and should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "ml avail",
            "ml list",
            "ml show python",
            "ml spider",
        ],
    )
    def test_ml_shortcut_queries_allowed(self, sandbox: CommandSandbox, command: str):
        """Lmod ml shortcut query commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"


class TestAlternatives:
    """Test alternatives system commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "alternatives --display python",
            "alternatives --display java",
            "update-alternatives --display python",
        ],
    )
    def test_alternatives_display_allowed(self, sandbox: CommandSandbox, command: str):
        """Alternatives display commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"


class TestVersionControl:
    """Test version control read-only commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git log --oneline -10",
            "git diff",
            "git branch -a",
            "git tag",
            "git ls-files",
            "git remote -v",
        ],
    )
    def test_git_read_commands_allowed(self, sandbox: CommandSandbox, command: str):
        """Git read-only commands should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"

    @pytest.mark.parametrize(
        "command",
        [
            "git add file",
            "git commit -m 'msg'",
            "git push",
            "git pull",
            "git checkout branch",
            "git merge branch",
            "git reset --hard",
        ],
    )
    def test_git_write_commands_blocked(self, sandbox: CommandSandbox, command: str):
        """Git write commands should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"


class TestPythonEnvironment:
    """Test Python environment commands."""

    @pytest.mark.parametrize(
        "command",
        [
            "python3 --version",
            "python3 -c 'import sys; print(sys.version)'",
            "pip list",
            "pip3 show numpy",
            "pip freeze",
            "conda info",
            "conda list",
            "conda env list",
        ],
    )
    def test_python_queries_allowed(self, sandbox: CommandSandbox, command: str):
        """Python environment queries should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"


class TestRedirections:
    """Test stderr/stdout redirections."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls /nonexistent 2>/dev/null",
            "cat file 2>&1",
            "find /path -type f 2>/dev/null | head",
            "which rg 2>/dev/null || which grep",
        ],
    )
    def test_stderr_redirections_allowed(self, sandbox: CommandSandbox, command: str):
        """Stderr redirections should be allowed."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"


class TestBackgroundProcesses:
    """Test background process prevention."""

    @pytest.mark.parametrize(
        "command",
        [
            "ls &",
            "sleep 100 &",
            "find / -name '*' &",
        ],
    )
    def test_background_blocked(self, sandbox: CommandSandbox, command: str):
        """Background processes should be blocked."""
        is_valid, error = sandbox.validate(command)
        assert not is_valid, f"Command '{command}' should be blocked"

    def test_double_ampersand_allowed(self, sandbox: CommandSandbox):
        """Double ampersand (&&) should be allowed."""
        is_valid, error = sandbox.validate("ls && pwd")
        assert is_valid, f"Command should be allowed: {error}"


class TestCommandSubstitution:
    """Test command substitution handling."""

    def test_backticks_blocked(self, sandbox: CommandSandbox):
        """Backtick command substitution should be blocked."""
        is_valid, _ = sandbox.validate("echo `whoami`")
        assert not is_valid

    def test_dollar_paren_allowed(self, sandbox: CommandSandbox):
        """$() command substitution should be allowed."""
        # This is allowed because we removed the $( pattern from forbidden
        is_valid, error = sandbox.validate("echo $(whoami)")
        assert is_valid, f"$() should be allowed: {error}"


class TestComplexCommands:
    """Test complex real-world command patterns."""

    @pytest.mark.parametrize(
        "command",
        [
            # Environment discovery
            "which python3; python3 --version; pip list 2>/dev/null | head -20",
            # Module exploration
            "module avail 2>&1 | head -50",
            # Package queries
            "rpm -qa | grep -E 'python|numpy|scipy' | sort",
            # System info
            "cat /etc/os-release; uname -a; hostname",
            # Path exploration
            "ls -la /opt; ls -la /usr/local/bin 2>/dev/null",
            # MDSplus style
            "env | grep -i mds; which mdstcl 2>/dev/null",
        ],
    )
    def test_complex_exploration_allowed(self, sandbox: CommandSandbox, command: str):
        """Complex exploration command patterns should work."""
        is_valid, error = sandbox.validate(command)
        assert is_valid, f"Command '{command}' should be allowed: {error}"


class TestScriptValidation:
    """Test script validation for multi-line scripts."""

    def test_valid_script(self, sandbox: CommandSandbox):
        """Valid read-only scripts should pass."""
        script = """#!/bin/bash
echo "=== Environment ==="
python3 --version
which pip3

echo "=== Packages ==="
pip list 2>/dev/null | head -20

echo "=== System ==="
uname -a
"""
        # Should not raise
        sandbox.validate_script(script)

    def test_script_with_rm_blocked(self, sandbox: CommandSandbox):
        """Scripts with rm should be blocked."""
        script = """#!/bin/bash
ls /tmp
rm -rf /tmp/*
"""
        with pytest.raises(ValueError, match="rm command"):
            sandbox.validate_script(script)

    def test_script_with_sudo_blocked(self, sandbox: CommandSandbox):
        """Scripts with sudo should be blocked."""
        script = """#!/bin/bash
sudo apt update
"""
        with pytest.raises(ValueError, match="sudo command"):
            sandbox.validate_script(script)

    def test_script_too_long_blocked(self, sandbox: CommandSandbox):
        """Scripts exceeding max lines should be blocked."""
        script = "\n".join(["echo line"] * 300)
        with pytest.raises(ValueError, match="too long"):
            sandbox.validate_script(script, max_lines=200)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_command(self, sandbox: CommandSandbox):
        """Empty command should be rejected."""
        is_valid, _ = sandbox.validate("")
        assert not is_valid

    def test_path_prefixed_command(self, sandbox: CommandSandbox):
        """Commands with full paths should work."""
        is_valid, error = sandbox.validate("/usr/bin/ls /etc")
        assert is_valid, f"Path-prefixed command should work: {error}"

    def test_quoted_arguments(self, sandbox: CommandSandbox):
        """Commands with quoted arguments should work."""
        is_valid, error = sandbox.validate("grep 'pattern with spaces' file")
        assert is_valid, f"Quoted args should work: {error}"

    def test_command_with_env_vars(self, sandbox: CommandSandbox):
        """Commands with environment variables should work."""
        is_valid, error = sandbox.validate("echo $HOME")
        assert is_valid, f"Env vars should work: {error}"
