"""Ghost operator CLI.

Provides a numbered control plane for Ghost runtime operations, task queue
management, reliability checks, and log inspection.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import webbrowser
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from ghost.config import get_config
from ghost.task_queue import TaskQueueStore

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RUN_DIR: Final[Path] = ROOT / "run"
LOGS_DIR: Final[Path] = ROOT / "logs"

PID_FILES: Final[dict[str, Path]] = {
    "mcp": RUN_DIR / "mcp.pid",
    "agent": RUN_DIR / "agent.pid",
    "ui": RUN_DIR / "ui.pid",
}

LOG_FILES: Final[dict[str, Path]] = {
    "mcp_out": LOGS_DIR / "mcp.out",
    "mcp_err": LOGS_DIR / "mcp.err",
    "agent_out": LOGS_DIR / "agent.out",
    "agent_err": LOGS_DIR / "agent.err",
    "ui_out": LOGS_DIR / "ui.out",
    "ui_err": LOGS_DIR / "ui.err",
}


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = src_path
    return env


def read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None

    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    try:
        return int(raw)
    except ValueError:
        return None


def remove_pid_file(pid_file: Path) -> None:
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_by_pid_file(label: str, pid_file: Path) -> None:
    pid = read_pid(pid_file)
    if pid is None:
        print(f"{label}: not running")
        remove_pid_file(pid_file)
        return

    if not is_process_running(pid):
        print(f"{label}: stale pid ({pid}) removed")
        remove_pid_file(pid_file)
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill.exe", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        killpg = getattr(os, "killpg", None)
        try:
            if killpg is not None:
                killpg(pid, 15)
            else:
                os.kill(pid, 15)
        except OSError:
            os.kill(pid, 15)

    remove_pid_file(pid_file)
    print(f"{label}: stopped ({pid})")


def spawn_managed_process(
    label: str,
    pid_file: Path,
    command: list[str],
    out_log: Path,
    err_log: Path,
) -> None:
    existing_pid = read_pid(pid_file)
    if existing_pid is not None and is_process_running(existing_pid):
        print(f"{label}: already running ({existing_pid})")
        return

    remove_pid_file(pid_file)
    ensure_dirs()

    env = _runtime_env()
    with out_log.open("ab") as out_handle, err_log.open("ab") as err_handle:
        if os.name == "nt":
            detached = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
            child = subprocess.Popen(  # noqa: S603
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                creationflags=detached,
            )
        else:
            child = subprocess.Popen(  # noqa: S603
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                start_new_session=True,
            )

    pid_file.write_text(str(child.pid), encoding="utf-8")
    print(f"{label}: started ({child.pid})")


def test_http_health(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return False

    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    connection_cls = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    target_path = parsed.path or "/"
    if parsed.query:
        target_path = f"{target_path}?{parsed.query}"

    connection = connection_cls(host, port, timeout=2)
    try:
        connection.request("GET", target_path)
        response = connection.getresponse()
        return 200 <= response.status < 500
    except OSError:
        return False
    finally:
        connection.close()


def print_header(title: str) -> None:
    print("====================================================")
    print(f"  {title}")
    print("====================================================")


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{text}{suffix}: ").strip()
    except EOFError:
        return "0"
    return value or default


def pause() -> None:
    try:
        input("\nPress Enter to continue...")
    except EOFError:
        return


def tail_file(file_path: Path, line_count: int = 80) -> None:
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-line_count:] if len(lines) > line_count else lines
    print("\n".join(tail))


def run_command(command: list[str], label: str) -> int:
    print(f"\nRunning {label}: {' '.join(command)}\n")
    result = subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        env=_runtime_env(),
        check=False,
    )
    print(f"\n{label} exit code: {result.returncode}")
    return int(result.returncode)


def start_mcp() -> None:
    spawn_managed_process(
        "mcp",
        PID_FILES["mcp"],
        [sys.executable, "-m", "ghost.mcp_server"],
        LOG_FILES["mcp_out"],
        LOG_FILES["mcp_err"],
    )


def start_agent() -> None:
    spawn_managed_process(
        "agent",
        PID_FILES["agent"],
        [sys.executable, "-m", "agents.training_agent"],
        LOG_FILES["agent_out"],
        LOG_FILES["agent_err"],
    )


def start_ui(ui_port: str) -> None:
    spawn_managed_process(
        "ui",
        PID_FILES["ui"],
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ghost.serving:create_serving_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            ui_port,
        ],
        LOG_FILES["ui_out"],
        LOG_FILES["ui_err"],
    )


def stop_mcp() -> None:
    stop_by_pid_file("mcp", PID_FILES["mcp"])


def stop_agent() -> None:
    stop_by_pid_file("agent", PID_FILES["agent"])


def stop_ui() -> None:
    stop_by_pid_file("ui", PID_FILES["ui"])


def start_stack(ui_port: str) -> None:
    start_mcp()
    start_agent()
    start_ui(ui_port)


def stop_stack() -> None:
    stop_ui()
    stop_agent()
    stop_mcp()


def restart_stack(ui_port: str) -> None:
    stop_stack()
    start_stack(ui_port)


def show_status(ui_port: str, ollama_host: str) -> None:
    mcp_pid = read_pid(PID_FILES["mcp"])
    agent_pid = read_pid(PID_FILES["agent"])
    ui_pid = read_pid(PID_FILES["ui"])

    mcp_running = mcp_pid is not None and is_process_running(mcp_pid)
    agent_running = agent_pid is not None and is_process_running(agent_pid)
    ui_running = ui_pid is not None and is_process_running(ui_pid)

    ui_healthy = test_http_health(f"http://127.0.0.1:{ui_port}/api/overview")
    ollama_healthy = test_http_health(f"{ollama_host.rstrip('/')}/api/tags")

    print_header("Ghost Stack Status")
    print(f"Root:           {ROOT}")
    print(f"Python:         {sys.executable}")
    print(f"Platform:       {platform.platform()}")
    print(f"UI port:        {ui_port}")
    print(f"Ollama host:    {ollama_host}")
    print(
        f"MCP server:     {'running (' + str(mcp_pid) + ')' if mcp_running else 'not running'}"
    )
    print(
        f"Agent:          {'running (' + str(agent_pid) + ')' if agent_running else 'not running'}"
    )
    print(
        f"Web UI:         {'running (' + str(ui_pid) + ')' if ui_running else 'not running'}"
    )
    print(f"UI health:      {'OK' if ui_healthy else 'DOWN'}")
    print(f"Ollama health:  {'OK' if ollama_healthy else 'DOWN'}")


def open_ui_browser(ui_port: str) -> None:
    url = f"http://127.0.0.1:{ui_port}/playground"
    webbrowser.open(url)
    print(f"Opened {url} in default browser.")


def task_queue_menu(task_queue: TaskQueueStore) -> None:
    while True:
        print_header("Task Queue Operations")
        print(f"Queue file: {task_queue.active_path()}")
        print("1) List pending tasks")
        print("2) List all tasks")
        print("3) Add task")
        print("4) Mark task complete")
        print("5) Delete task")
        print("0) Back")

        choice = prompt("\nChoice")
        if choice == "0":
            return

        if choice == "1":
            tasks = task_queue.list_tasks(include_completed=False)
            if not tasks:
                print("No pending tasks.")
            for index, task in enumerate(tasks, start=1):
                suffix = f" [{task.task_id}]" if task.task_id else ""
                print(f"{index}. {task.text}{suffix}")
        elif choice == "2":
            tasks = task_queue.list_tasks(include_completed=True)
            if not tasks:
                print("No tasks found.")
            for index, task in enumerate(tasks, start=1):
                suffix = f" [{task.task_id}]" if task.task_id else ""
                status = "done" if task.completed else "pending"
                print(f"{index}. ({status}) {task.text}{suffix}")
        elif choice == "3":
            text = prompt("Task text")
            if text and text != "0":
                task = task_queue.add_task(text)
                print(f"Created task: {task.task_id or task.text}")
        elif choice == "4":
            task_id = prompt("Task id")
            if task_id and task_id != "0":
                maybe_task = task_queue.update_task(task_id=task_id, completed=True)
                if maybe_task is None:
                    print("Task not found.")
                else:
                    print(f"Marked complete: {maybe_task.task_id or maybe_task.text}")
        elif choice == "5":
            task_id = prompt("Task id")
            if task_id and task_id != "0":
                maybe_task = task_queue.delete_task(task_id=task_id)
                if maybe_task is None:
                    print("Task not found.")
                else:
                    print(f"Deleted: {maybe_task.task_id or maybe_task.text}")
        else:
            print(f"Unknown option: {choice}")

        pause()


def reliability_menu() -> None:
    while True:
        print_header("Reliability and Validation")
        print("1) Run tests")
        print("2) Run Ruff lint")
        print("3) Run Mypy")
        print("4) Run package build")
        print("5) Check Python/FastAPI/uvicorn imports")
        print("0) Back")

        choice = prompt("\nChoice")
        if choice == "0":
            return

        if choice == "1":
            run_command([sys.executable, "-m", "pytest"], "tests")
        elif choice == "2":
            run_command([sys.executable, "-m", "ruff", "check", "."], "ruff")
        elif choice == "3":
            run_command([sys.executable, "-m", "mypy", "src/ghost"], "mypy")
        elif choice == "4":
            run_command([sys.executable, "-m", "build"], "build")
        elif choice == "5":
            run_command(
                [
                    sys.executable,
                    "-c",
                    "import fastapi,uvicorn;from ghost.serving import create_serving_app;create_serving_app();print('serve-surface-ok')",
                ],
                "serve-import-smoke",
            )
        else:
            print(f"Unknown option: {choice}")

        pause()


def logs_menu() -> None:
    while True:
        print_header("Log Inspection")
        print("1) Tail mcp.out")
        print("2) Tail mcp.err")
        print("3) Tail agent.out")
        print("4) Tail agent.err")
        print("5) Tail ui.out")
        print("6) Tail ui.err")
        print("0) Back")

        choice = prompt("\nChoice")
        if choice == "0":
            return

        if choice == "1":
            tail_file(LOG_FILES["mcp_out"])
        elif choice == "2":
            tail_file(LOG_FILES["mcp_err"])
        elif choice == "3":
            tail_file(LOG_FILES["agent_out"])
        elif choice == "4":
            tail_file(LOG_FILES["agent_err"])
        elif choice == "5":
            tail_file(LOG_FILES["ui_out"])
        elif choice == "6":
            tail_file(LOG_FILES["ui_err"])
        else:
            print(f"Unknown option: {choice}")

        pause()


def main() -> None:
    config = get_config()
    ensure_dirs()

    ui_port = os.environ.get("UI_PORT", "8000").strip() or "8000"
    task_queue = TaskQueueStore(config.task_queue_file)

    while True:
        print_header("Ghost Operator CLI")
        print(f"Root: {ROOT}")
        print(f"UI:   http://127.0.0.1:{ui_port}")
        print("")
        print("1) Start full stack")
        print("2) Stop full stack")
        print("3) Restart full stack")
        print("4) Show stack status")
        print("5) Open UI in browser")
        print("6) Start MCP server")
        print("7) Start training agent")
        print("8) Start web UI")
        print("9) Task queue operations")
        print("10) Reliability and validation")
        print("11) Log inspection")
        print("0) Exit")

        choice = prompt("\nChoice")

        if choice == "1":
            start_stack(ui_port)
            pause()
        elif choice == "2":
            stop_stack()
            pause()
        elif choice == "3":
            restart_stack(ui_port)
            pause()
        elif choice == "4":
            show_status(ui_port, config.ollama_host)
            pause()
        elif choice == "5":
            open_ui_browser(ui_port)
            pause()
        elif choice == "6":
            start_mcp()
            pause()
        elif choice == "7":
            start_agent()
            pause()
        elif choice == "8":
            start_ui(ui_port)
            pause()
        elif choice == "9":
            task_queue_menu(task_queue)
        elif choice == "10":
            reliability_menu()
        elif choice == "11":
            logs_menu()
        elif choice == "0":
            return
        else:
            print(f"Unknown option: {choice}")
            pause()


if __name__ == "__main__":
    main()
