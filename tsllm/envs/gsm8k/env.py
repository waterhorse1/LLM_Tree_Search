import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import CoTEnv, TokenEnv, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
STOP_STR = "The answer is "


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
    else:
        return INVALID_ANS
    return match_str


def extract_groundtruth(groundtruth_str: str):
    x = groundtruth_str.split("#### ")[1].strip().replace(",", "")
    try:
        float(x)
    except:
        raise ValueError(
            "Warning: Error should raise since the extracted groundtruth string {}\
             cannot be converted to float".format(
                x
            )
        )
    return x


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str):
    float_groundtruth = float(extracted_groundtruth)
    try:
        return abs(float(answer) - float_groundtruth) < 1e-5
    except Exception:
        return False


class Gsm8kEnv(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    @property
    def stop_str(self):
        return STOP_STR

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            f"Question: {self.math_problem['question']}\nAnswer: Let's think step by step"
        ]

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0


class Gsm8kTokenEnv(TokenEnv):
    sep = SEP
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    @property
    def stop_str(self):
        return self.tokenizer.eos_token

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.problem["question"], self.problem["answer"], extracted_answer
        )

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            f"Question: {self.problem['question']}\nAnswer: Let's think step by step"
        ]

    def get_reward(self, *args, **kwargs):
        """To implement based on learned reward model"""
        return 0
    
    # def step(self, action, update_legal_action=True, ):
    #     self.action_history.append(action)
    #     state = self.get_state()
    #     reward = self.get_reward(state)
    #     terminated, truncated, info = self.get_done_and_info()
        
    #     # update legal actions
    #     if not (terminated or truncated) and update_legal_action:
    #         self._legal_actions = self.update_legal_actions()
    #     else:
    #         self._legal_actions = None
    #     return state, reward, terminated, truncated, info
