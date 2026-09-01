"""What a run puts on stdout, per LLM call.

One Slurm job writes a single .out file, from up to 90 processes at once, over
the order of 100,000 requests. Every response, every memory-update prompt and
every line of every experiment log was going into it, while the per-experiment
log files already hold all of it, one per experiment and named for the config.

So the console keeps the one line per completed task that run_task prints, and
the rest is available at DEBUG when someone asks for it.
"""

import logging
import tempfile

from mas.llm import Message, TokenTracker
from mas.module_map import MAS_MEMORY_MODULES
from mas.settings import LLMSettings

from tasks.envs.base_env import BaseRecorder
from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM, chat_over_fake_completions

PROMPT = [Message("user", "what next?")]
SETTINGS = LLMSettings(api_base="http://localhost:9999/v1", api_key="none")


def test_an_llm_response_does_not_reach_the_console(capsys):
    chat, _ = build_chat_answering("the model said something distinctive")

    chat(PROMPT)

    captured = capsys.readouterr()
    assert "distinctive" not in captured.out + captured.err


def test_an_llm_response_is_there_at_debug(caplog):
    chat, _ = build_chat_answering("the model said something distinctive")

    with caplog.at_level(logging.DEBUG, logger="mas.llm"):
        chat(PROMPT)

    assert "distinctive" in caplog.text


def test_a_memory_update_prompt_does_not_reach_the_console(capsys):
    memory = MAS_MEMORY_MODULES["intrinsicmemory-notemplate"](
        namespace="intrinsicmemory-notemplate",
        global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=FakeLLM(tracker=TokenTracker()),
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context("a task", task_description="a description")
    memory.move_memory_state("look at desk 1", "you see a distinctive mug")

    memory.summarize(solver_message="the solver said this")

    captured = capsys.readouterr()
    assert "distinctive" not in captured.out + captured.err


def test_the_experiment_log_goes_to_its_file_and_not_the_console(capsys, tmp_path):
    recorder = BaseRecorder(working_dir=str(tmp_path), namespace="an-experiment")

    recorder.log("a distinctive line of the experiment log")

    captured = capsys.readouterr()
    assert "distinctive" not in captured.out + captured.err
    assert "distinctive" in (tmp_path / "an-experiment.log").read_text()


def build_chat_answering(answer: str):
    return chat_over_fake_completions([answer], settings=SETTINGS)
