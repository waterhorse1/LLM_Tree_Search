## iterative update of GSM8k
Here we describe how to rollout and training iteratively, taking GSM8k for example,

### Rollout
use `test_sft_and_v.py` to sample examples on training dataset. Rollout hyperparameters for GSM8k:
```python
{
    "temperature": 1.0, 
    "max_length": 8, 
    "max_action": 10, 
    "pb_c_init": 3, 
    "num_simulations": 5, 
    "num_mcts_aggregation": 12, 
    "rollout_method": "mcts.get_next_action", 
    "mcts_sample": True,
    "clear_tree": True,
    "reset_total_tree": False,

    # # useless hyperparameters in SearchArgs
    # "prune_ratio": 0.7,
    # "prune_value": None,
    # "select_by_prior": False,
    # "max_simulation": None,
    # "max_token": 51200,
    # "k_maj": 100,
}
```
Run `test_sft_and_v.py` with `--test False`, which means testing on training dataset.
After rollout, first merge all data, using `tsllm/merge_jsonl.py`.
Then check `it1_gsm8k.ipynb` to merge data for SFT and critic training.

### Training
check `train_mcts_scripts/gsm8k/` for `train_gsm8k_{sft/critic}.py`, modify args in `config`, then train it.