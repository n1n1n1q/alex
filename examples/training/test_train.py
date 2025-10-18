from minestudio.online.rollout.start_manager import start_rolloutmanager
from minestudio.online.trainer.start_trainer import start_trainer
from omegaconf import OmegaConf
from train_utils import env_generator, policy_generator


online_dict = {
 "trainer_name": "PPOTrainer",
    "detach_rollout_manager": True,
    "rollout_config": {
    "num_rollout_workers": 1,
    "num_gpus_per_worker": 0.1,
    "num_cpus_per_worker": 1,
    "fragment_length": 256,
    "to_send_queue_size": 6,
    "worker_config": {
        "num_envs": 2,
        "batch_size": 2,
        "restart_interval": 60,  # 1h
        "video_fps": 20,
        "video_output_dir": "output/videos",
    },
    "replay_buffer_config": {
        "max_chunks": 64,
        "max_reuse": 2,
        "max_staleness": 2,
        "fragments_per_report": 40,
        "fragments_per_chunk": 1,
        "database_config": {
            "path": "output/replay_buffer_cache",
            "num_shards": 8,
        },
    },
    "episode_statistics_config": {},
    },
    "train_config": {
        "num_workers": 1,
        "num_gpus_per_worker": 0.5,
        "num_cpus_per_worker": 1,
        "num_iterations": 1,
        "vf_warmup": 0,
        "learning_rate": 0.00002,
        "anneal_lr_linearly": False,
        "weight_decay": 0.04,
        "adam_eps": 1e-8,
        "batch_size_per_gpu": 1,
        "batches_per_iteration": 1,
        "gradient_accumulation": 1, 
        "epochs_per_iteration": 1, 
        "context_length": 64,
        "discount": 0.999,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "clip_vloss": False, 
        "max_grad_norm": 5, 
        "zero_initial_vf": True,
        "ppo_policy_coef": 1.0,
        "ppo_vf_coef": 0.5, 
        "kl_divergence_coef_rho": 0.1,
        "entropy_bonus_coef": 0.0,
        "coef_rho_decay": 0.9995,
        "log_ratio_range": 50,  
        "normalize_advantage_full_batch": True, 
        "use_normalized_vf": True,
        "num_readers": 4,
        "num_cpus_per_reader": 0.1,
        "prefetch_batches": 2,
        "save_interval": 1,
        "keep_interval": 1,
        "record_video_interval": 2,
        "fix_decoder": False,
        "resume": None, 
        "resume_optimizer": True,
        "save_path": "output",
        "use_amp": True,
    },

    "logger_config": {
        "project": "minestudio_online",
        "name": "bow_cow"
    },
}


if __name__ == "__main__":

    with open("/alex/examples/training/test_train.py", 'r', encoding="utf8") as f:
        whole_config = f.read()

    online_config = OmegaConf.create(online_dict)
    start_rolloutmanager(policy_generator, env_generator, online_config)
    start_trainer(policy_generator, env_generator, online_config, whole_config)
