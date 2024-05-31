# TS_LLM: AlphaZero-like tree-search learning framework for LLMs 
The official implementation of paper: [Alphazero-like Tree-Search can guide large language model decoding and training](https://arxiv.org/pdf/2309.17179.pdf)

# Enviroment Installation
please use correct version of `transformers` and `ctranlate2`
```
conda create -n tsllm python==3.10
conda activate tsllm

pip install -r requirement.txt

pip install -e .
```


# Runnable Scripts
We show examples of one task, other tasks are similar and we provide the corresponding scripts.

## Start
We use [Ctranslate2(3.17.1)](https://github.com/OpenNMT/CTranslate2) to speedup LLM inference, which is implemented in C++ and much faster than python huggingface. To use Ctranslate2, you need first transform your LLM model, here is an example:
```bash
ct2-transformers-converter --model {your huggingface model path} --quantization bfloat16 --output_dir {your ct2_cache target path}
# We use bfloat 16 for LLaMA model and float32 for GPT2 model
```

Note that we use Ctranslate2 for all policy inference, so for any policy in our codebase, do not forget to convert to ct2 model first.

## Training of Value and Policy
Examples are shown in `tran_mcts_scripts`, use GSM8k as example
```bash
cd train_mcts_scripts/gsm8k
# SFT for GSM8K, Game24 and ProntoQA
# Note that For RLHF we do not conduct SFT training, we directly utilize vicgalle/gpt2-open-instruct-v1.
accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_sft.py 

# Critic training for all four tasks, data is collected by data collection section.
accelerate launch --config_file mcts_gsm8k_llama_deepspeed.yaml train_gsm8k_critic.py
```
You can customize `config` in each py files, e.g. `config["train"]["checkpoint_dir"]` and `config["train"]["project_name"]`. (we use accelerate so we also provide the accelerate config in `accelerate_config.yaml`)

## Data Collection for Value Training
Examples are shown in `tsllm/offline_rl`, use GSM8k as example 

```bash
cd tsllm/offline_rl

# please check the scripts, for gsm8k and game24, we use 3 checkpoints to rollout data
# which is named as ${CT2_CACHE}/llama2_sft_ep${i}_ct2
sh gsm8k_data/gen_3.sh {your ct2 transformed path} {your model tokenizer path} # This is for dataset generation

sh gsm8k_data/process.sh # This is for dataset processing
```

## Testing over CoT, CoT-SC and TS-LLM
For GSM8K, Game24, ProtoQA, you should use `tsllm/offline_rl/test_sft_and_v.py` to test the policy model and value function.
To run the tests, you should know 2 key concepts used in the code, the first one is 4 test settings, which is controlled by setting environment variables; the other is search arguments, which is set in `tsllm/offline_rl/test_sft_and_v.py` as elements in `arg_list`

There are four types of test setting:
- `TEST_NO_TERMINAL` is MCTS/other tree search methods in GSM8K/ProntoQA/Game24 (we assume we do not know the final reward in these 3 tasks)
- `TEST_WITH_TERMINAL` is MCTS/other tree search methods in RLHF (we assume we know the final reward in RLHF)
- `TEST_COT_GREEDY` is CoT greedy decoding
- `TEST_COT_SC` is CoT-SC

There are several args to control CoT-SC and Tree Search methods, see `tsllm/offline_rl/test_sft_and_v.py::SearchArgs` for more information.

To run the test, using the following scripts (examples in `train_mcts_scripts/gsm8k/test_policy_and_value.sh`)
```
cd train_mcts_scripts/gsm8k/
sh test_policy_and_value_sh {your save dir}
```

**Please make sure the SearchArguments you are using are correct, e.g. check `"max_action"`, `"max_length"`, etc.**


For RLHF environment, it is basically similar except that you should use `tsllm/offline_rl/test_sft_and_v_rlhf.py`. We assume we have a reward function in RLHF setting, so you shoud set `TEST_WITH_TERMINAL=1` for rlhf experiment. 

There are several args to control CoT-SC and Tree Search methods, see `tsllm/offline_rl/test_sft_and_v_rlhf.py::SearchArgs` for more information.

To run the test, we provide an example in `train_mcts_scripts/rlhf/test_policy_and_value.sh`.

## Iterative Update
For iterative update, please refer to `train_mcts_scripts/gsm8k` and `train_mcts_scripts/rlhf` for more instructions.

## Citation
If you find our repo useful, please cite it in your publications.

```bibtex
@article{feng2023alphazero,
  title={Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training},
  author={Feng, Xidong and Wan, Ziyu and Wen, Muning and Wen, Ying and Zhang, Weinan and Wang, Jun},
  journal={arXiv preprint arXiv:2309.17179},
  year={2023}
}
```

## Acknowledgement
Our code implementation refers to code from [lightzero](https://github.com/opendilab/LightZero).

