from tsllm.rl.trainer.mcts_trainer_traj_ct2_value import AccelerateMCTSTrainer
from tsllm.rl.config import RLConfig
from peft import LoraConfig, PeftType

config = {
    "model": {
        "model_path": "vicgalle/gpt2-open-instruct-v1",
        "value_model_type_name": "AutoModelForCausalLMWithValueHead"
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
            dict(lr=2e-5, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
    },
    "scheduler": {
        "name": "cosine_warmup",
        "kwargs": dict(warmup_ratio=0.03)
    },
    "train": {
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "seq_length": 1024,
        "epochs": 2, # this is the epoch for the whole sampling/training process
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "train_epoch": 1, #  this is the epoch for training process after each sampling
        "value_loss_coef": 1.0,
        "eval_interval": 1,
        "checkpoint_interval": 1,
        "checkpoint_dir": # Your checkpoint path,
        "save_optimizer": False,
        "tracker": "tensorboard",
        "logging_dir": "logs/",
        "project_name": # Your project name,
        "pre_onpolicy_datapath": # Your onpolicy value data,
        "pre_onpolicy_datapath_train_test": None,
        "pre_onpolicy_datapath_test": None,
        "onpolicy_per_problem_max_size": 1000,
        "sft_per_problem_max_size": 1000,
        "env_name": 'rlhf',
        "task_dataset_kwargs":{
            "path": 'Dahoas/synthetic-instruct-gptj-pairwise',
            "num_train_data": 30000,
        }
    },
}

config = RLConfig.from_dict(config)
trainer = AccelerateMCTSTrainer(config)

trainer.learn()
