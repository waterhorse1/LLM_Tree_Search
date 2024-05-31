from tsllm.envs.gsm8k.env import (
    Gsm8kEnv,
    COT_EXAMPLES,
    COT_TASK_DESC,
    PROBLEM_FORMAT_STR,
    SEP,
)
import pytest

if __name__ == "__main__":
    problem_input = "1 3 3 4"
    env = Gsm8kEnv(
        config={},
        math_problems=[{"question": "1 3 3 4", "answer": "3"}],
        tokenizer=None,
        llm_gen_fn=None,
        reset=False,
    )

    env.reset(False)
    print(env.get_state())

    print(env._is_correct("The answer is 3"))
    print(env._is_correct("\n\nThe answer is 3."))
    print(env._is_correct("The answer is 4"))
    print(env._is_correct("The answer is x"))

    build_query_str = Gsm8kEnv.build_query_str
    print("\n\n====== ZERO SHOT COT ============")
    print(
        build_query_str(
            COT_TASK_DESC, COT_EXAMPLES, PROBLEM_FORMAT_STR, problem_input, SEP, False
        )
    )
    # print("\n\n====== FEW SHOT COT ============")
    # print(
    #     build_query_str(
    #         COT_TASK_DESC, COT_EXAMPLES, PROBLEM_FORMAT_STR, problem_input, SEP, True
    #     )
    # )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print("\n\n====== default sft dataset ============")
    from tsllm.envs import get_default_sft_data_builder, get_env_datasets

    train_ds, _ = get_env_datasets("gsm8k")
    q2idx_dict = {}
    for idx, problem_inst in enumerate(train_ds):
        question = problem_inst["question"]
        q2idx_dict[question] = idx
    sft_data = get_default_sft_data_builder("gsm8k")(
        "tsllm/envs/gsm8k/train_data/sft_init.jsonl",
        q2idx_dict,
        tokenizer=tokenizer,
        add_eos_token=True,
        is_few_shot=False,
    )

    print("Len train_ds: {}\ntrian_ds[0]:\n{}".format(len(train_ds), train_ds[0]))
    print("Len sft_data: {}\nsft_data[0]:\n{}".format(len(sft_data), sft_data[0]))

    print("\n\n====== default critic dataset ============")
    from tsllm.envs import get_default_critic_data_builder

    critic_data = get_default_critic_data_builder("gsm8k")(
        "tsllm/envs/gsm8k/train_data/sft_init.jsonl",
        q2idx_dict,
        tokenizer=tokenizer,
        is_few_shot=False,
    )
    print(
        "Len critic_data: {}\ncritic_data[0]:\n{}".format(
            len(critic_data), critic_data[0]
        )
    )
    print(len(tokenizer.encode(critic_data[0]["query_str"] + critic_data[0]["answer"])))
    print(len(tokenizer.encode(critic_data[0]["query_str"])))
