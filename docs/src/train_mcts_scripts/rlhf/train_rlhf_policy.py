import os

import math
from datetime import timedelta
from typing import Dict
from dataclasses import dataclass
import torch
from traitlets import Any
from tsllm.rl.trainer.mcts_trainer_traj_ct2_sft import AccelerateMCTSTrainer, loop_iter
from tsllm.rl.config import RLConfig
from peft import LoraConfig, PeftType

config = {
    "model": {
        "model_path": "vicgalle/gpt2-open-instruct-v1",
    },
    "tokenizer": {
        "tokenizer_path": "vicgalle/gpt2-open-instruct-v1",
        "padding_side": "right"
    },
    "mcts": {},
    "env": {},
    "optimizer": {
        "name":
            "adamw",
        "kwargs":
            dict(lr=2.0e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
    },
    "scheduler": {
        "name": "cosine_warmup",
        "kwargs": dict(warmup_ratio=0.03)
    },
    "train": {
        "gamma": 1.0,
        "gae_lambda": 0.9,
        "seq_length": 1024,
        "epochs": 3, # this is the epoch for the whole sampling/training process
        "sft_micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "train_epoch": 1, # this is the epoch for training process after each sampling
        "sft_loss_coef": 1.0,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "checkpoint_dir": # Your checkpoint dir,
        "save_optimizer": False,
        "tracker": "tensorboard",
        "logging_dir": "logs/",
        "project_name": # Your project name,
        "pre_sft_datapath": # Your SFT jsonl datapath,
        "pre_onpolicy_datapath": None,      
        "onpolicy_per_problem_max_size": 1000,
        "sft_per_problem_max_size": 1000,
        "env_name": 'rlhf',
        "task_dataset_kwargs":{
            "path": 'Dahoas/synthetic-instruct-gptj-pairwise', # or call "dataset_path": Your dataset path,
            "num_train_data": 30000,
        }
    },
}

config = RLConfig.from_dict(config)
trainer = AccelerateMCTSTrainer(config)

trainer.learn()