## Iterative update of RLHF
Here we describe how to rollout and training iteratively for RLHF.

### Rollout
use `tsllm/offline_rl/test_sft_and_v_rlhf.py` to sample examples on training dataset. Rollout hyperparameters for RLHF:
```python
{
    "temperature": 1.0, 
    "max_length": 64, 
    "pb_c_init": 3, 
    "num_simulations": 5, 
    "num_mcts_aggregation": 10, 
    "rollout_method": "mcts.get_next_action", # this is mcts-alpha
    "mcts_sample": True,
    "clear_tree": True,
    "reset_total_tree": False,
    "select_by_prior": False, 
}
```

In addition to these hyperparameters,, you need also to modify the hyperparameters in `test_policy_and_value.sh` with `--train`, which means we conduct rollouts on training dataset. After rollouts, you can refer to `ts_llm/offline_rlhf/process.sh` (convert rollout data to a corresponding format), `filter_top_data_policy_training.py` (filter top k data from rollouts to construct policy training data) and `mix_value_data.py` (mix original value dataset and new sampled value dataset, to construct the value training data) to get the final processed training data.


### Training
After getting the data in rollout, check `train_mcts_scripts/rlhf` for `train_rlhf_{sft/critic}.py`, modify args in `config`, then train it. (we use accelerate so we also provide the accelerate config in `train_mcts_scripts/rlhf/accelerate_config.yaml`)
