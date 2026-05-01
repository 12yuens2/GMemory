import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from .base_env import BaseEnv, BaseRecorder

# ---------------------------------------------------------------------------
# Subprocess runner for code generation / self-repair test execution.
# Passed as the -c argument to a fresh Python process so that student code
# memory allocations are isolated from the main process.
# ---------------------------------------------------------------------------
_CODE_RUNNER = r'''
import sys, json, io
from types import ModuleType
from unittest.mock import patch

sys.set_int_max_str_digits(50000)

_PREAMBLE = (
    "from string import *\nfrom re import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom math import *\n"
    "from itertools import *\nfrom functools import *\nfrom operator import *\n"
    "from typing import *\nimport sys, json\nsys.setrecursionlimit(50000)\n"
)

def _run(code, inputs, outputs, fn_name):
    full_code = _PREAMBLE + "\n" + code

    if fn_name:
        # Compile once to extract function / class
        module = ModuleType("solution")
        try:
            exec(full_code, module.__dict__)
        except Exception as e:
            return False, {"error_code": -1, "error": str(e)[:300]}

        sol_cls = module.__dict__.get("Solution")
        if isinstance(sol_cls, type):
            try:
                instance = sol_cls()
                method = getattr(instance, fn_name, None)
            except Exception as e:
                return False, {"error_code": -1, "error": str(e)[:300]}
        else:
            method = module.__dict__.get(fn_name)
        if method is None:
            return False, {"error_code": -1, "error": f"Function {fn_name!r} not found"}

        for inp, exp in zip(inputs, outputs):
            try:
                args = [json.loads(line) for line in inp.strip().split("\n") if line.strip()]
                result = method(*args)
                if isinstance(result, tuple):
                    result = list(result)
                expected = json.loads(exp)
                if result != expected:
                    return False, {
                        "error_code": -2,
                        "inputs": str(inp)[:200],
                        "expected": str(expected)[:200],
                        "output": str(result)[:200],
                    }
            except Exception as e:
                return False, {
                    "error_code": -4,
                    "inputs": str(inp)[:200],
                    "error": str(e)[:200],
                }
        return True, {}
    else:
        # Run code per test case with mocked stdin/stdout.
        # Do NOT pre-exec here — top-level code calls input() which would
        # hit the already-exhausted stdin pipe used to deliver the JSON payload.
        for inp, exp in zip(inputs, outputs):
            try:
                captured = io.StringIO()
                mock_stdin = io.StringIO(inp)
                with patch("sys.stdin", mock_stdin), patch("sys.stdout", captured):
                    exec(full_code, {"__name__": "__main__"})
                actual = captured.getvalue().strip()
                expected = str(exp).strip()
                if actual != expected:
                    return False, {
                        "error_code": -2,
                        "inputs": str(inp)[:200],
                        "expected": expected[:200],
                        "output": actual[:200],
                    }
            except Exception as e:
                return False, {
                    "error_code": -4,
                    "inputs": str(inp)[:200],
                    "error": str(e)[:200],
                }
        return True, {}

data = json.loads(sys.stdin.read())
passed, metadata = _run(data["code"], data["inputs"], data["outputs"], data.get("fn_name"))
print(json.dumps({"passed": passed, "metadata": metadata}))
'''


def _run_code_tests(code: str, input_output_json: str, timeout: int = 30) -> tuple[bool, str]:
    """Execute code against test cases in an isolated subprocess."""
    io_data = json.loads(input_output_json)
    payload = json.dumps({
        "code": code,
        "inputs": io_data.get("inputs", []),
        "outputs": io_data.get("outputs", []),
        "fn_name": io_data.get("fn_name"),
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CODE_RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            return data["passed"], json.dumps(data["metadata"])
        err = (result.stderr or "subprocess failed")[:300]
        return False, json.dumps({"error_code": -5, "error": err})
    except subprocess.TimeoutExpired:
        return False, json.dumps({"error_code": -3, "error": "execution timeout"})
    except Exception as e:
        return False, json.dumps({"error_code": -5, "error": str(e)[:200]})


def _extract_code(text: str) -> str:
    """Extract Python code from the first markdown code fence."""
    lines = text.split("\n")
    fences = [i for i, ln in enumerate(lines) if "```" in ln]
    if len(fences) >= 2:
        return "\n".join(lines[fences[0] + 1 : fences[1]])
    return text.strip()


def _format_error(metadata: dict) -> str:
    code = metadata.get("error_code", -5)
    if code == -1:
        return f"Compilation error: {metadata.get('error', '')}"
    if code == -2:
        return (
            f"Wrong answer.\nInput: {metadata.get('inputs', '')}\n"
            f"Expected: {metadata.get('expected', '')}\n"
            f"Got: {metadata.get('output', '')}"
        )
    if code == -3:
        return "Time limit exceeded."
    if code == -4:
        return f"Runtime error: {metadata.get('error', '')}"
    return f"Execution error: {metadata.get('error', '')}"


# ---------------------------------------------------------------------------
# LCBCodegenEnv
# ---------------------------------------------------------------------------

class LCBCodegenEnv(BaseEnv):

    def __init__(self, env_config: dict[str, Any], max_trials: int = 1) -> None:
        self.env_config = env_config
        self.max_trials = max_trials
        self.config: dict = {}
        self.input_output: str = None
        self.reward: float = 0.0
        self.generated_code: str = ""
        self.error_metadata: str = "{}"
        self.results: list[dict] = []

    def set_env(self, task_config: dict) -> tuple[str, str]:
        self.config = task_config
        self.input_output = task_config["input_output"]
        task_main = task_config["question_title"]
        task_description = self._format_problem(
            task_config["question_content"],
            task_config.get("starter_code", ""),
        )
        return task_main, task_description

    def reset(self) -> None:
        self.reward = 0.0
        self.generated_code = ""
        self.error_metadata = "{}"

    def step(self, action: str) -> tuple[str, float, bool]:
        code = self.process_action(action)
        self.generated_code = code
        passed, metadata_json = _run_code_tests(code, self.input_output)
        self.error_metadata = metadata_json
        if passed:
            self.reward = 1.0
            obs = "All tests passed."
        else:
            obs = _format_error(json.loads(metadata_json))
        self.results.append({
            "question_id": self.config.get("question_id", ""),
            "question_title": self.config.get("question_title", ""),
            "question_content": self.config.get("question_content", ""),
            "starter_code": self.config.get("starter_code", ""),
            "input_output": self.input_output,
            "generated_code": code,
            "passed": passed,
            "error_metadata": metadata_json,
        })
        return obs, self.reward, True

    @staticmethod
    def process_action(action: str) -> str:
        return _extract_code(action)

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        feedback = "Tests passed." if done else f"Tests failed. {self.error_metadata}"
        return self.reward, done, feedback

    @staticmethod
    def _format_problem(question_content: str, starter_code: str) -> str:
        prompt = f"### Question:\n{question_content}\n\n"
        if starter_code:
            prompt += (
                "### Format: Use the following starter code and enclose your solution "
                "within triple-backtick Python code fences.\n"
                f"```python\n{starter_code}\n```\n\n"
            )
        else:
            prompt += (
                "### Format: Read inputs from stdin, solve the problem and write the answer "
                "to stdout. Enclose your solution within triple-backtick Python code fences.\n"
                "```python\n# YOUR CODE HERE\n```\n\n"
            )
        prompt += "### Answer: (use the provided format with backticks)\n\n"
        return prompt


@dataclass
class LCBCodegenRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.counts: int = 0
        self.rewards: float = 0.0
        self.env: LCBCodegenEnv = None

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.counts += 1
        self.log(
            f"reward: {reward}, pass@1: {self.rewards / self.counts:.3f} "
            f"({int(self.rewards)}/{self.counts})"
        )
        if self.env is not None:
            results_path = os.path.join(self.working_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(self.env.results, f, indent=2)


# ---------------------------------------------------------------------------
# LCBCodeExecEnv
# ---------------------------------------------------------------------------

class LCBCodeExecEnv(BaseEnv):

    def __init__(self, env_config: dict[str, Any], max_trials: int = 1) -> None:
        self.env_config = env_config
        self.max_trials = max_trials
        self.config: dict = {}
        self.reward: float = 0.0

    def set_env(self, task_config: dict) -> tuple[str, str]:
        self.config = task_config
        task_main = task_config.get("function_name", task_config.get("id", ""))
        code = task_config["code"]
        inp = task_config["input"]
        task_description = (
            "Given the following Python function and input, predict the exact output "
            "when the function is executed with that input.\n\n"
            f"```python\n{code}\nassert {inp} == ??\n```\n\n"
            "Provide the full assertion with the correct output inside [ANSWER] and [/ANSWER] tags.\n"
            "Example: [ANSWER]assert foo(x=1) == 42[/ANSWER]"
        )
        return task_main, task_description

    def reset(self) -> None:
        self.reward = 0.0

    def step(self, action: str) -> tuple[str, float, bool]:
        predicted = self.process_action(action)
        expected = str(self.config["expected_output"]).strip()
        correct = self._compare_outputs(predicted, expected)
        self.reward = 1.0 if correct else 0.0
        obs = "Correct." if correct else f"Incorrect. Expected: {expected}"
        return obs, self.reward, True

    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip()
        if "[ANSWER]" in action and "[/ANSWER]" in action:
            start = action.index("[ANSWER]") + len("[ANSWER]")
            end = action.index("[/ANSWER]")
            action = action[start:end].strip()
        if "==" in action:
            action = action.split("==", 1)[1].strip()
        return action.strip()

    @staticmethod
    def _compare_outputs(predicted: str, expected: str) -> bool:
        try:
            import ast
            return ast.literal_eval(predicted.strip()) == ast.literal_eval(expected.strip())
        except Exception:
            return predicted.strip() == expected.strip()

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        feedback = "Correct." if done else "Incorrect."
        return self.reward, done, feedback


@dataclass
class LCBCodeExecRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.counts: int = 0
        self.rewards: float = 0.0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.counts += 1
        self.log(
            f"reward: {reward}, accuracy: {self.rewards / self.counts:.3f} "
            f"({int(self.rewards)}/{self.counts})"
        )


# ---------------------------------------------------------------------------
# LCBTestPredEnv
# ---------------------------------------------------------------------------

class LCBTestPredEnv(BaseEnv):

    def __init__(self, env_config: dict[str, Any], max_trials: int = 1) -> None:
        self.env_config = env_config
        self.max_trials = max_trials
        self.config: dict = {}
        self.reward: float = 0.0

    def set_env(self, task_config: dict) -> tuple[str, str]:
        self.config = task_config
        task_main = task_config.get("question_title", task_config.get("question_id", ""))
        test_input = task_config["test_input"]
        starter_code = task_config.get("starter_code", "")
        task_description = (
            "Given the following problem description and starter function, predict the output "
            "for the provided test case.\n\n"
            f"### Problem:\n{task_config['question_content']}\n\n"
            f"### Starter Code:\n```python\n{starter_code}\n```\n\n"
            f"### Test Input:\n{test_input}\n\n"
            "Provide the full assertion with the correct output inside [ANSWER] and [/ANSWER] tags.\n"
            "Example: [ANSWER]assert solution(x=1) == 42[/ANSWER]"
        )
        return task_main, task_description

    def reset(self) -> None:
        self.reward = 0.0

    def step(self, action: str) -> tuple[str, float, bool]:
        predicted = self.process_action(action)
        expected = str(self.config["expected_output"]).strip()
        correct = self._compare_outputs(predicted, expected)
        self.reward = 1.0 if correct else 0.0
        obs = "Correct." if correct else f"Incorrect. Expected: {expected}"
        return obs, self.reward, True

    @staticmethod
    def process_action(action: str) -> str:
        action = action.strip()
        if "[ANSWER]" in action and "[/ANSWER]" in action:
            start = action.index("[ANSWER]") + len("[ANSWER]")
            end = action.index("[/ANSWER]")
            action = action[start:end].strip()
        if "==" in action:
            action = action.split("==", 1)[1].strip()
        return action.strip()

    @staticmethod
    def _compare_outputs(predicted: str, expected: str) -> bool:
        try:
            import ast
            return ast.literal_eval(predicted.strip()) == ast.literal_eval(expected.strip())
        except Exception:
            return predicted.strip() == expected.strip()

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        feedback = "Correct." if done else "Incorrect."
        return self.reward, done, feedback


@dataclass
class LCBTestPredRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.counts: int = 0
        self.rewards: float = 0.0

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.counts += 1
        self.log(
            f"reward: {reward}, accuracy: {self.rewards / self.counts:.3f} "
            f"({int(self.rewards)}/{self.counts})"
        )


# ---------------------------------------------------------------------------
# LCBSelfRepairEnv
# ---------------------------------------------------------------------------

class LCBSelfRepairEnv(BaseEnv):

    def __init__(self, env_config: dict[str, Any], max_trials: int = 1) -> None:
        self.env_config = env_config
        self.max_trials = max_trials
        self.config: dict = {}
        self.input_output: str = None
        self.reward: float = 0.0
        self.generated_code: str = ""
        self.error_metadata: str = "{}"
        self.results: list[dict] = []

    def set_env(self, task_config: dict) -> tuple[str, str]:
        self.config = task_config
        self.input_output = task_config["input_output"]
        task_main = task_config.get("question_title", task_config.get("question_id", ""))
        failed_code = task_config.get("generated_code", "")
        error_meta = task_config.get("error_metadata", "{}")
        try:
            error_info = json.loads(error_meta)
        except Exception:
            error_info = {}
        error_msg = _format_error(error_info) if error_info else "Unknown error."
        task_description = (
            "### Problem:\n"
            f"{task_config['question_content']}\n\n"
        )
        starter = task_config.get("starter_code", "")
        if starter:
            task_description += f"### Starter Code:\n```python\n{starter}\n```\n\n"
        task_description += (
            f"### Failed Solution:\n```python\n{failed_code}\n```\n\n"
            f"### Error:\n{error_msg}\n\n"
            "### Format: Provide a corrected solution enclosed within triple-backtick Python "
            "code fences.\n```python\n# YOUR FIXED CODE HERE\n```\n\n"
            "### Answer: (use the provided format with backticks)\n\n"
        )
        return task_main, task_description

    def reset(self) -> None:
        self.reward = 0.0
        self.generated_code = ""
        self.error_metadata = "{}"

    def step(self, action: str) -> tuple[str, float, bool]:
        code = self.process_action(action)
        self.generated_code = code
        passed, metadata_json = _run_code_tests(code, self.input_output)
        self.error_metadata = metadata_json
        if passed:
            self.reward = 1.0
            obs = "All tests passed."
        else:
            obs = _format_error(json.loads(metadata_json))
        self.results.append({
            "question_id": self.config.get("question_id", ""),
            "question_title": self.config.get("question_title", ""),
            "original_code": self.config.get("generated_code", ""),
            "repaired_code": code,
            "passed": passed,
            "error_metadata": metadata_json,
        })
        return obs, self.reward, True

    @staticmethod
    def process_action(action: str) -> str:
        return _extract_code(action)

    def feedback(self) -> tuple[float, bool, str]:
        done = self.reward == 1.0
        feedback = "Tests passed." if done else f"Tests failed. {self.error_metadata}"
        return self.reward, done, feedback


@dataclass
class LCBSelfRepairRecorder(BaseRecorder):

    def __post_init__(self):
        super().__post_init__()
        self.counts: int = 0
        self.rewards: float = 0.0
        self.env: LCBSelfRepairEnv = None

    def task_begin(self, task_id: int, task_config: dict):
        super().task_begin(task_id, task_config)
        self.log(f"---------- Task: {task_id} ----------")

    def task_end(self, reward: float, done: bool):
        self.rewards += reward
        self.counts += 1
        self.log(
            f"reward: {reward}, pass@1: {self.rewards / self.counts:.3f} "
            f"({int(self.rewards)}/{self.counts})"
        )
        if self.env is not None:
            results_path = os.path.join(self.working_dir, "results.json")
            with open(results_path, "w") as f:
                json.dump(self.env.results, f, indent=2)
