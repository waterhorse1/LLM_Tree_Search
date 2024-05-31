from importlib import import_module
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Optional, Callable, Dict
from .utils import build_critic_data_component, build_sft_data_component


def get_env_datasets(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")
    return task_module.get_train_test_dataset(**kwargs)


def get_default_sft_data_builder(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")
    return partial(
        build_sft_data_component,
        build_query_str_fn=task_module.Env.build_query_str,
        build_response_str_fn=task_module.Env.build_response_str,
        sep=task_module.SEP,
        cot_task_desc_str=task_module.COT_TASK_DESC,
        cot_example_str=task_module.COT_EXAMPLES,
        problem_format_str=task_module.PROBLEM_FORMAT_STR,
    )


def get_default_critic_data_builder(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")
    return partial(
        build_critic_data_component,
        build_query_str_fn=task_module.Env.build_query_str,
        sep=task_module.SEP,
        cot_task_desc_str=task_module.COT_TASK_DESC,
        cot_example_str=task_module.COT_EXAMPLES,
        problem_format_str=task_module.PROBLEM_FORMAT_STR,
    )


def get_default_query_str_builder(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")

    def fn(problem_input: str, is_few_shot: bool):
        return task_module.Env.build_query_str(
            cot_task_desc=task_module.COT_TASK_DESC,
            cot_examples=task_module.COT_EXAMPLES,
            problem_format_str=task_module.PROBLEM_FORMAT_STR,
            problem_input=problem_input,
            sep=task_module.SEP,
            is_few_shot=is_few_shot,
        )

    return fn


def get_default_response_str_builder(env_name: str, **kwargs):
    task_module = import_module(f"tsllm.envs.{env_name}")

    def fn(problem_input: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool):
        return task_module.Env.build_response_str(
            problem_input,
            tokenizer,
            add_eos_token,
        )
    return fn


def get_env_answer_checker(env_name):
    task_module = import_module(f"tsllm.envs.{env_name}")

    def judge_answer(problem_str, groundtruth_str, answer_completion: str):
        # should feed the unprocessed `groundtruth_str` and `answer_completion_str`
        return task_module.judge_correct(
            problem_str,
            task_module.extract_groundtruth(groundtruth_str),
            task_module.extract_answer(answer_completion),
        )

    return judge_answer
