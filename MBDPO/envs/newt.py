"""Single-task Newt/MMBench environment entry point for MBDPO-JAX.

Phase 2 intentionally supports only state observations and single child
environments. It does not instantiate Newt's vectorized multitask soup runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np

from MBDPO.common.mmbench import MMBENCH_TASK_METADATA

if importlib.util.find_spec("torch") is not None:
    import torch
else:  # pragma: no cover - torch is optional for JAX env I/O.
    torch = None

if importlib.util.find_spec("tensordict") is not None:
    from tensordict import TensorDictBase
else:  # pragma: no cover - tensordict is optional.
    TensorDictBase = ()


_SUPPORTED_OBS_MODE = "state"
_NEWT_FACTORY_MODULES = (
    "envs.dmcontrol",
    "envs.maniskill",
    "envs.metaworld",
    "envs.mujoco",
    "envs.box2d",
    "envs.robodesk",
    "envs.ogbench",
    "envs.pygame",
    "envs.atari",
)
_NEWT_MODULE_PREFIXES = ("envs", "common")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_newt_root(newt_root: str | Path | None) -> Path:
    root = Path(newt_root or "third_party/newt")
    if not root.is_absolute():
        root = _repo_root() / root
    return root


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _require_state_obs_mode(cfg: Any) -> str:
    obs_mode = str(_cfg_get(cfg, "newt_obs_mode", _cfg_get(cfg, "obs", "state"))).lower()
    obs = str(_cfg_get(cfg, "obs", obs_mode)).lower()
    unsupported = {obs_mode, obs} - {_SUPPORTED_OBS_MODE}
    if unsupported:
        raise NotImplementedError(
            "Newt Phase 2 supports state-only observations. "
            f"Requested newt_obs_mode={obs_mode!r}, obs={obs!r}; RGB/state+RGB support is not implemented yet."
        )
    return _SUPPORTED_OBS_MODE


def _metadata_for_task(task: str) -> dict[str, Any]:
    if task not in MMBENCH_TASK_METADATA:
        raise ValueError(f"Unknown Newt/MMBench task {task!r}.")
    return MMBENCH_TASK_METADATA[task]


class _NewtChildConfig(SimpleNamespace):
    """Small config object matching fields used by Newt child factories."""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _make_child_cfg(cfg: Any) -> _NewtChildConfig:
    task = str(_cfg_get(cfg, "task"))
    metadata = _metadata_for_task(task)
    return _NewtChildConfig(
        task=task,
        child_env=True,
        obs=_SUPPORTED_OBS_MODE,
        num_envs=1,
        seed=int(_cfg_get(cfg, "seed", 1)),
        save_video=False,
        num_demos=0,
        rank=int(_cfg_get(cfg, "rank", 0)),
        render_size=int(_cfg_get(cfg, "render_size", 224)),
        episode_length=int(metadata["episode_length"]),
        action_dim=int(metadata["action_dim"]),
        domain=metadata["domain"],
        instruction=metadata.get("instruction"),
        discount=float(metadata["discount"]),
    )


@contextmanager
def _newt_import_context(newt_root: Path):
    """Temporarily expose Newt's tdmpc2 package for its absolute imports.

    Newt factory modules import siblings as ``envs.*``. MBDPO also has a top-level
    ``envs`` package on ``sys.path``, so the context temporarily removes existing
    top-level Newt/MBDPO aliases and prepends ``third_party/newt/tdmpc2`` while
    importing the official child factories. Loaded MBDPO modules remain available
    through their fully-qualified ``MBDPO.*`` names.
    """

    tdmpc2_root = newt_root / "tdmpc2"
    if not tdmpc2_root.is_dir():
        raise FileNotFoundError(f"Cannot find Newt tdmpc2 package at {tdmpc2_root}.")

    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "envs" or name.startswith("envs.") or name == "common" or name.startswith("common.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    path = str(tdmpc2_root)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "envs" or name.startswith("envs.") or name == "common" or name.startswith("common."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


def _import_newt_factory_modules(newt_root: Path):
    modules = []
    with _newt_import_context(newt_root):
        for module_name in _NEWT_FACTORY_MODULES:
            try:
                modules.append(importlib.import_module(module_name))
            except ImportError:
                # Some Newt domains have optional dependencies. Keep imports lazy
                # enough that a missing domain does not prevent other domains from
                # being constructed.
                continue
    return modules


def make_child_env(cfg):
    """Create one official Newt child env for ``cfg.task`` without vectorization."""

    _require_state_obs_mode(cfg)
    newt_root = _resolve_newt_root(_cfg_get(cfg, "newt_root", None))
    child_cfg = _make_child_cfg(cfg)
    errors = []
    for module in _import_newt_factory_modules(newt_root):
        try:
            return module.make_env(child_cfg)
        except ValueError as exc:
            if "Unknown task" in str(exc):
                continue
            raise
        except ImportError as exc:
            errors.append(f"{module.__name__}: {exc}")
            continue
    details = f" Optional dependency errors: {'; '.join(errors)}" if errors else ""
    raise ValueError(f"Failed to create Newt child environment for task {child_cfg.task!r}.{details}")


class NewtEnv(gym.Env):
    """State-only, single-task Newt environment normalized for MBDPO."""

    metadata = {}

    def __init__(self, env, cfg):
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.task = str(_cfg_get(cfg, "task"))
        self.task_metadata = _metadata_for_task(self.task)
        self.domain = str(self.task_metadata["domain"])
        self.observation_space = self._state_observation_space(env.observation_space)
        self.action_space = env.action_space
        self.max_episode_steps = int(
            getattr(env, "max_episode_steps", self.task_metadata["episode_length"])
        )

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def rand_act(self):
        return np.asarray(self.action_space.sample(), dtype=np.float32)

    def reset(self, *args, **kwargs):
        reset_out = self.env.reset(*args, **kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return self._obs_to_state(obs)

    def step(self, action):
        action = self._action_to_numpy(action)
        step_out = self.env.step(action)
        if not isinstance(step_out, tuple):
            raise TypeError(f"Expected Newt step() to return a tuple, got {type(step_out)!r}.")
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated = bool(done)
            truncated = False
        else:
            raise ValueError(f"Expected Newt step() 4- or 5-tuple, got {len(step_out)} values.")
        info = self._normalize_info(info, reward, terminated, truncated)
        return self._obs_to_state(obs), np.float32(reward), done, info

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        raise NotImplementedError("Underlying Newt child environment does not expose render().")

    def _state_observation_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.Dict):
            if "state" not in observation_space.spaces:
                raise ValueError("State-only Newt observations require an observation_space with a 'state' key.")
            return observation_space.spaces["state"]
        return observation_space

    def _obs_to_state(self, obs):
        if torch is not None and torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        elif TensorDictBase and isinstance(obs, TensorDictBase):
            obs = obs.get("state") if "state" in obs.keys() else obs
        if isinstance(obs, dict):
            if "state" not in obs:
                raise ValueError("Newt observation dict does not contain required 'state' key.")
            obs = obs["state"]
        if torch is not None and torch.is_tensor(obs):
            obs = obs.detach().cpu().numpy()
        if TensorDictBase and isinstance(obs, TensorDictBase):
            if "state" not in obs.keys():
                raise ValueError("Newt TensorDict observation does not contain required 'state' key.")
            obs = obs.get("state")
            if torch is not None and torch.is_tensor(obs):
                obs = obs.detach().cpu().numpy()
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _action_to_numpy(self, action):
        if torch is not None and torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_dim = int(np.prod(self.action_space.shape))
        if action.size < action_dim:
            raise ValueError(
                f"Action for Newt task {self.task!r} has width {action.size}, "
                f"but the child environment requires {action_dim}."
            )
        if action.size > action_dim:
            action = action[:action_dim]
        return action.astype(np.float32, copy=False).reshape(self.action_space.shape)

    def _normalize_info(self, info, reward, terminated, truncated):
        info = dict(info or {})
        info["terminated"] = bool(info.get("terminated", terminated))
        info["truncated"] = bool(info.get("truncated", truncated))
        info["success"] = float(info.get("success", 0.0))
        # Use the immediate reward as a safe scalar score when the child env does
        # not expose Newt's normalized score metric for this domain/task.
        info["score"] = float(info.get("score", reward if reward is not None else 0.0))
        info["domain"] = info.get("domain", self.domain)
        info["task"] = info.get("task", self.task)
        return info


def make_env(cfg):
    """Create a state-only single-task Newt/MMBench environment for MBDPO-JAX."""

    _require_state_obs_mode(cfg)
    env = make_child_env(cfg)
    return NewtEnv(env, cfg)
