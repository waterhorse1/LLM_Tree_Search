from tsllm.rl.trainer.mcts_trainer_traj_ct2_sft import AccelerateMCTSTrainer
from tsllm.rl.config import RLConfig
from tsllm.model.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

config = {
    "model": {
        "model_path": "meta-llama/Llama-2-7b-hf",
    },
    "tokenizer": {
        "tokenizer_path": "meta-llama/Llama-2-7b-hf",
        "padding_side": "right",
    },
    "optimizer": {
        "name": "adamw",
        "kwargs": dict(lr=2.0e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0),
    },
    "scheduler": {"name": "cosine_warmup", "kwargs": dict(warmup_ratio=0.03)},
    "train": {
        "pre_sft_datapath": "../../tsllm/envs/prontoqa/train_data/train.jsonl",
        "env_name": "prontoqa",
        "epochs": 1,
        "train_epoch": 1,
        "sft_micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "seq_length": 1024,
        "eval_interval": 1,
        "sft_loss_coef": 1.0,
        "checkpoint_interval": 1,
        "checkpoint_dir": tmp_for_check,
        "save_optimizer": False,
        "project_name": "tmp_for_check",
        "tracker": "tensorboard",
        "logging_dir": "logs/",
        "sft_per_problem_max_size": 1000,
    },
    "mcts": {},
    "env": {},
}

config = RLConfig.from_dict(config)
trainer = AccelerateMCTSTrainer(config)

trainer.learn()
