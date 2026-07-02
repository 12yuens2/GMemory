import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tasks.run as run_module


class FakeRecorder:
    def __init__(self):
        self.started = 0
        self.ended = 0

    def dataset_begin(self):
        self.started += 1

    def task_begin(self, task_id, task_config):
        self.task_id = task_id

    def task_end(self, reward, done, trials):
        self.ended += 1

    def average_results(self):
        return 1.0, 1.0, 1

    def log(self, message):
        return None

    def dataset_end(self):
        return None


class FakeMAS:
    def __init__(self):
        self.agents_team = {"agent": type("Agent", (), {"name": "agent", "add_task_instruction": lambda self, instruction: instruction})()}
        self.env = type("Env", (), {"set_env": lambda self, task_config: ("main", "desc")})()

    def add_observer(self, recorder):
        self.observer = recorder

    def build_system(self, reasoning_module, mas_memory_module, env, mas_config):
        self.reasoning_module = reasoning_module
        self.mas_memory_module = mas_memory_module
        self.env = env
        self.mas_config = mas_config

    def schedule(self, task_config):
        return 1.0, True, 1


class FakeTaskManager:
    def __init__(self, tasks):
        self.task_name = "alfworld"
        self.mas_type = "autogen"
        self.memory_type = "none"
        self.tasks = tasks
        self.recorder = FakeRecorder()
        self.mas = FakeMAS()
        self.mas_config = {}
        self.mem_config = {}


def test_append_local_result_is_serialized_across_threads(tmp_path, monkeypatch):
    result_path = tmp_path / "results.csv"
    active_writes = []

    class StrictFile:
        def __init__(self, path, mode, encoding=None):
            self.path = path
            self.mode = mode
            self.encoding = encoding

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, text):
            thread_id = threading.get_ident()
            if thread_id in active_writes:
                raise RuntimeError("concurrent write detected")
            active_writes.append(thread_id)
            time.sleep(0.01)
            active_writes.remove(thread_id)
            with open(self.path, "a", encoding=self.encoding) as handle:
                handle.write(text)

    monkeypatch.setattr(run_module, "open", lambda path, mode="r", encoding=None: StrictFile(path, mode, encoding))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_module.append_local_result, str(result_path), f"line-{idx}\n") for idx in range(8)]
        for future in futures:
            future.result()

    lines = result_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8
    assert lines[0].startswith("line-")


def test_run_task_can_use_multiple_workers(tmp_path, monkeypatch):
    created_managers = []

    def fake_build_task(task, mas_type, memory_type, max_steps):
        manager = FakeTaskManager(tasks=[{"task_id": 1}, {"task_id": 2}])
        created_managers.append(manager)
        return manager

    def fake_build_mas(task_manager, reasoning=None, mas_memory=None, llm_type=None):
        return None

    monkeypatch.setattr(run_module, "WORKING_DIR", str(tmp_path))
    monkeypatch.setattr(run_module, "build_task", fake_build_task)
    monkeypatch.setattr(run_module, "build_mas", fake_build_mas)
    monkeypatch.setattr(run_module, "get_task_few_shots", lambda **kwargs: [])
    monkeypatch.setattr(run_module, "get_dataset_system_prompt", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(run_module, "get_price", lambda: (1, 2, 0.0))
    monkeypatch.setattr(run_module, "get_intrinsic_price", lambda: (3, 4))

    base_manager = FakeTaskManager(tasks=[{"task_id": 1}, {"task_id": 2}])
    run_module.run_task(base_manager, seed=7, num_workers=2, reasoning="io", mas_memory="none", llm_type="test-model", max_steps=3)

    assert len(created_managers) == 2
    assert base_manager.recorder.started == 1
