"""Train a small PPO actor/critic on September--January; select on February.

The network and optimizer come from Stable-Baselines3's tested PPO implementation
(https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). This file
owns experiment configuration, seeded training, logs, and checkpoint selection.
It never loads the testing split. Training uses the configured primary parent
size; evaluating other sizes later measures generalization, not extra training.

Example: python train.py --config configs/experiment.json --output runs/ppo
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
from statistics import fmean, pstdev
import time

from data import DATA_DIR, SPLIT_DAYS, load_split
from features import FEATURE_NAMES


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PPO = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 1.0,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    """Predeclare a deterministic subset without inspecting episode outcomes.

    Both endpoints are included when selecting at least two episodes. A single
    requested episode is the middle episode; oversized requests use all rows.
    """
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (total, count)):
        raise ValueError("total and count must be positive integers")
    count = min(total, count)
    if count == 1:
        return (total // 2,)
    return tuple(round(index * (total - 1) / (count - 1)) for index in range(count))


def training_settings(config: dict) -> dict:
    """Resolve defaults, with nested training settings overriding top-level ones."""
    training = config.get("training", {})
    unknown = set(training.get("ppo", {})) - set(DEFAULT_PPO)
    if unknown:
        raise ValueError(f"unknown PPO settings: {sorted(unknown)}")
    settings = {
        "total_timesteps": training.get("total_timesteps", config.get("total_timesteps", 200_000)),
        "validation_interval": training.get("validation_interval", 50_000),
        "validation_episodes": training.get("validation_episodes", 32),
        "torch_threads": training.get("torch_threads", 1),
        "ppo": {**DEFAULT_PPO, **training.get("ppo", {})},
    }
    ppo = settings["ppo"]
    for name, value in list(settings.items())[:-1] + [
        (name, ppo[name]) for name in ("n_steps", "batch_size", "n_epochs")
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if ppo["n_steps"] < 2 or ppo["batch_size"] < 2:
        raise ValueError("n_steps and batch_size must be at least two for normalized advantages")
    if ppo["n_steps"] % ppo["batch_size"]:
        raise ValueError("batch_size must divide n_steps for full minibatches")
    for name in ("learning_rate", "gamma", "gae_lambda", "ent_coef", "clip_range", "vf_coef", "max_grad_norm"):
        value = ppo[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 < ppo["gamma"] <= 1 or not 0 <= ppo["gae_lambda"] <= 1:
        raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
    if any(ppo[name] <= 0 for name in ("learning_rate", "clip_range", "max_grad_norm")):
        raise ValueError("learning_rate, clip_range, and max_grad_norm must be positive")
    if ppo["ent_coef"] < 0 or ppo["vf_coef"] < 0:
        raise ValueError("ent_coef and vf_coef must be nonnegative")
    return settings


def evaluate_checkpoint(model, env, indices: tuple[int, ...]) -> dict:
    """Evaluate deterministic actions on the same ordered validation episodes.

    Completion score is minimized: it includes net execution shortfall and the
    environment's residual penalty. Comparing only filled-order prices would
    otherwise reward a policy for leaving difficult inventory unsold.
    """
    summaries = []
    action_counts = [0, 0, 0, 0]
    for episode_index in indices:
        observation, _ = env.reset(options={"episode_index": episode_index})
        for _ in range(60):
            action, _ = model.predict(observation, deterministic=True)
            action_id = int(action)
            action_counts[action_id] += 1
            observation, _, terminated, truncated, info = env.step(action_id)
            if terminated or truncated:
                if truncated or "summary" not in info:
                    raise RuntimeError("execution validation must terminate with a summary")
                summaries.append(info["summary"])
                break
        else:
            raise RuntimeError("execution validation exceeded its 60-step horizon")
    scores = [summary["completion_score_bps"] for summary in summaries]
    if not scores or not all(math.isfinite(score) for score in scores):
        raise ValueError("validation must produce finite completion scores")
    return {
        "episode_count": len(summaries),
        "mean_completion_score_bps": fmean(scores),
        "std_completion_score_bps": pstdev(scores),
        "completion_rate": fmean(float(summary["completed"]) for summary in summaries),
        "mean_total_reward": fmean(summary["total_reward"] for summary in summaries),
        **{f"action_{action}_count": count for action, count in enumerate(action_counts)},
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _versions() -> dict[str, str]:
    names = ("numpy", "torch", "gymnasium", "stable-baselines3")
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in names}}


def _parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_seed(training_episodes, validation_episodes, config: dict,
               output_dir: str | Path, seed: int) -> dict:
    """Train one independently seeded PPO run in a new directory.

    Episode sampling, NumPy, PyTorch, and the action space receive the seed via
    SB3 and Gymnasium. The caller supplies training/validation episodes only.
    Every eligible checkpoint is ranked on a fixed validation subset; full
    February evaluation and selection across seeds are a subsequent step.
    """
    # Lazy imports keep configuration/data utilities usable without installing
    # the training stack, and make dependency errors point at the right phase.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from gym_env import GymSellExecutionEnv

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    settings = training_settings(config)
    size_name = config.get("primary_size", "medium")
    parent_quantity = float(config["sizes"][size_name])
    fee_bps = float(config.get("fee_bps", 10.0))
    if not math.isfinite(parent_quantity) or parent_quantity <= 0:
        raise ValueError("primary parent quantity must be finite and positive")
    if not math.isfinite(fee_bps) or not 0 <= fee_bps < 10_000:
        raise ValueError("fee_bps must be finite and in [0, 10000)")
    if not training_episodes or not validation_episodes:
        raise ValueError("nonempty training and validation episodes are required")
    indices = evenly_spaced_indices(len(validation_episodes), settings["validation_episodes"])
    output = Path(output_dir)
    # Never mix stale checkpoints or logs into a new experiment accidentally.
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    training_env = Monitor(
        GymSellExecutionEnv(training_episodes, parent_quantity, fee_bps=fee_bps),
        filename=str(output / "episodes.monitor.csv"),
    )
    validation_env = GymSellExecutionEnv(validation_episodes, parent_quantity, fee_bps=fee_bps)

    # The actor maps ten features through 64-Tanh-64-Tanh to four categorical
    # action logits. A separate 64-Tanh-64-Tanh critic estimates expected future
    # reward. They share the observation input, not hidden-layer parameters.
    # SB3 stores (s, a, reward, done, value, log pi_old) in an on-policy rollout
    # buffer. GAE recursively accumulates TD residuals using gamma/lambda; done
    # stops bootstrapping at the real execution deadline or completed order.
    # PPO maximizes mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)), with normalized
    # advantages, plus entropy; it minimizes critic mean-squared return error.
    # Adam and gradient-norm clipping update both networks for n_epochs. Gamma
    # 1 preserves the undiscounted execution-cost objective. Rewards remain in
    # the env's bps units; observations already have fixed feature scales.
    model = PPO(
        "MlpPolicy", training_env, device="cpu", seed=seed, verbose=0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                       "activation_fn": torch.nn.Tanh},
        normalize_advantage=True, clip_range_vf=None,
        **settings["ppo"],
    )
    model.set_logger(configure(str(output), ["csv"]))
    initial_digest = _parameter_digest(model)
    validation_records = []
    episode_fields = ("num_timesteps", "episode_index", "completion_score_bps",
                      "completion_fraction", "completed", "total_reward")

    class ValidationCallback(BaseCallback):
        """Checkpoint after completed PPO updates, never during a partial update."""

        def __init__(self):
            super().__init__()
            self.best_score = math.inf
            self.best_timesteps = None
            self.next_validation = settings["validation_interval"]

        def validate(self, stage: str):
            metrics = evaluate_checkpoint(self.model, validation_env, indices)
            eligible = stage != "initial"
            record = {"num_timesteps": self.num_timesteps, "stage": stage,
                      "eligible_for_selection": eligible, **metrics}
            validation_records.append(record)
            validation_writer.writerow(record)
            validation_handle.flush()
            if eligible:
                checkpoint = output / "checkpoints" / f"step_{self.num_timesteps:09d}.zip"
                self.model.save(checkpoint)
                if metrics["mean_completion_score_bps"] < self.best_score:
                    self.best_score = metrics["mean_completion_score_bps"]
                    self.best_timesteps = self.num_timesteps
                    shutil.copyfile(checkpoint, output / "best_model.zip")
                _write_json(output / "selection.json", {
                    "selection_split": "validation",
                    "criterion": "minimum mean_completion_score_bps; earlier checkpoint wins ties",
                    "validation_episode_indices": list(indices),
                    "best_timesteps": self.best_timesteps,
                    "best_mean_completion_score_bps": self.best_score,
                    "best_model": "best_model.zip",
                    "evaluations": validation_records,
                })
            self.logger.record("validation/mean_completion_score_bps", metrics["mean_completion_score_bps"])
            print(f"seed={seed} steps={self.num_timesteps} {stage} "
                  f"validation_score_bps={metrics['mean_completion_score_bps']:.6f}", flush=True)

        def _on_training_start(self):
            # Log the untrained reference, but only trained checkpoints can win.
            self.validate("initial")

        def _on_rollout_start(self):
            # The previous rollout's optimizer update has completed here.
            if self.num_timesteps >= self.next_validation:
                self.validate("checkpoint")
                interval = settings["validation_interval"]
                self.next_validation = (self.num_timesteps // interval + 1) * interval

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", ()):
                summary = info.get("summary")
                if summary is not None:
                    episode_writer.writerow({
                        "num_timesteps": self.num_timesteps,
                        "episode_index": info["episode_index"],
                        **{name: summary[name] for name in episode_fields[2:]},
                    })
            return True

        def _on_training_end(self):
            # SB3 may round the budget up to a complete rollout. This evaluation
            # includes the final optimizer update, unlike an on_step callback.
            self.validate("final")

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    validation_fields = ("num_timesteps", "stage", "eligible_for_selection", "episode_count",
                         "mean_completion_score_bps", "std_completion_score_bps", "completion_rate",
                         "mean_total_reward", *(f"action_{action}_count" for action in range(4)))
    callback = ValidationCallback()
    try:
        with (output / "validation.csv").open("w", newline="") as validation_handle, \
                (output / "training_episodes.csv").open("w", newline="") as episode_handle:
            validation_writer = csv.DictWriter(validation_handle, fieldnames=validation_fields)
            validation_writer.writeheader()
            episode_writer = csv.DictWriter(episode_handle, fieldnames=episode_fields)
            episode_writer.writeheader()
            model.learn(total_timesteps=settings["total_timesteps"], callback=callback, log_interval=1)
        model.save(output / "last_model.zip")
        shutil.copyfile(output / "best_model.zip", output / "model.zip")
        # SB3 normally logs training losses before the next rollout. Explicitly
        # flush once so the last update's losses are present in progress.csv too.
        model.logger.record("time/total_timesteps", model.num_timesteps)
        model.logger.dump(model.num_timesteps)
        manifest = {
            "seed": seed, "started_at_utc": started_at,
            "elapsed_seconds": time.monotonic() - started,
            "config": config, "resolved_training": settings, "versions": _versions(),
            "device": "cpu", "primary_size": size_name, "parent_quantity_btc": parent_quantity,
            "fee_bps": fee_bps, "requested_timesteps": settings["total_timesteps"],
            "actual_timesteps": model.num_timesteps,
            "training_episode_count": len(training_episodes),
            "validation_episode_count": len(validation_episodes),
            "validation_episode_indices": list(indices),
            "validation_episode_starts": [str(validation_episodes[index][0]["decision_time"]) for index in indices],
            "feature_names": list(FEATURE_NAMES),
            "observation_scaling": "fixed features.py scales; no fitted normalization",
            "reward_scaling": 1.0,
            "actor_hidden_sizes": [64, 64], "critic_hidden_sizes": [64, 64], "activation": "Tanh",
            "best_validation_score_bps": callback.best_score,
            "best_checkpoint_timesteps": callback.best_timesteps,
            "initial_parameter_sha256": initial_digest,
            "final_parameter_sha256": _parameter_digest(model),
            "selected_model": str((output / "model.zip").resolve()),
            "selection_scope": "fixed February subset within this seed; full validation chooses final seed later",
        }
        _write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        training_env.close()
        validation_env.close()
        model.logger.close()


def load_trained_policy(model_path: str | Path):
    """Load trusted local SB3 model artifacts for deterministic ``predict`` calls."""
    from stable_baselines3 import PPO
    return PPO.load(model_path, device="cpu")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "experiment.json")
    parser.add_argument("--output", type=Path, default=Path("runs/ppo"))
    parser.add_argument("--timesteps", type=int, help="override the budget for each seed")
    parser.add_argument("--seeds", nargs="+", type=int, help="override configured seeds")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.timesteps is not None:
        config.setdefault("training", {})["total_timesteps"] = args.timesteps
    seeds = args.seeds if args.seeds is not None else config.get("seeds", [7, 19, 42])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide at least one seed with no duplicates")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.output}")
    training_settings(config)
    data_dir = Path(config.get("data_dir", DATA_DIR))
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    print("Loading training and February validation episodes...", flush=True)
    training_episodes = list(load_split("training", data_dir))
    validation_episodes = list(load_split("validation", data_dir))
    args.output.mkdir(parents=True, exist_ok=True)
    _write_json(args.output / "configuration.json", {**config, "seeds": seeds})
    manifests = []
    for seed in seeds:
        print(f"Training seed {seed} on {len(training_episodes)} episodes", flush=True)
        manifests.append(train_seed(training_episodes, validation_episodes, config,
                                    args.output / f"seed_{seed}", seed))
    _write_json(args.output / "training_manifest.json", {
        "training_days": list(SPLIT_DAYS["training"]),
        "validation_days": list(SPLIT_DAYS["validation"]),
        "seeds": seeds,
        "runs": [{name: manifest[name] for name in
                  ("seed", "actual_timesteps", "best_checkpoint_timesteps", "best_validation_score_bps", "selected_model")}
                 for manifest in manifests],
        "next_step": "Rank per-seed model.zip on full February validation; freeze selection before testing.",
    })


if __name__ == "__main__":
    main()
