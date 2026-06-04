"""Read-only Newt/MMBench task metadata for MBDPO.

This module intentionally does not import Newt packages or create environments.  It
loads task metadata from ``third_party/newt/tasks.json`` and statically parses
Newt's lightweight task-set declaration in ``tdmpc2/common/__init__.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

_DEFAULT_NEWT_ROOT = "third_party/newt"
_NEWT_COMMON_RELATIVE_PATH = Path("tdmpc2/common/__init__.py")
_TASKS_JSON = "tasks.json"
_TRAIN_TASK_SET_NAME = "soup"
_DEFAULT_DISCOUNT_FACTOR = 0.99
# Newt/MMBench pads state and action channels for batching across domains.  The
# raw task metadata in tasks.json currently tops out below the public padded
# action width, so MMBENCH_ACTION_DIMS uses this compatibility ceiling while
# MMBENCH_TASK_METADATA preserves each task's source action_dim.
_EXPECTED_MAX_OBS_DIM = 128
_EXPECTED_PADDED_ACTION_DIM = 16
_OVERLAP_DOMAINS = (
    "dmcontrol",
    "dmcontrol-ext",
    "metaworld",
    "maniskill",
    "mujoco",
    "box2d",
)
_NEW_DOMAINS = ("robodesk", "ogbench", "pygame", "atari")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_newt_root(newt_root: str | Path | None = None) -> Path:
    root = Path(newt_root or _DEFAULT_NEWT_ROOT)
    if not root.is_absolute():
        root = _repo_root() / root
    return root


def _load_tasks_json(newt_root: Path) -> dict[str, dict[str, Any]]:
    path = newt_root / _TASKS_JSON
    if not path.is_file():
        raise FileNotFoundError(
            f"Newt/MMBench metadata file is missing: {path}. "
            f"Expected {_DEFAULT_NEWT_ROOT}/{_TASKS_JSON} relative to the repository root."
        )
    with path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, dict):
        raise ValueError(f"Expected {path} to contain a JSON object mapping task names to metadata.")
    return tasks


def _load_newt_task_sets(newt_root: Path) -> dict[str, list[str]]:
    path = newt_root / _NEWT_COMMON_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Newt task-set definitions are missing: {path}. "
            "Cannot locate tdmpc2/common/__init__.py."
        )
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_SET" for target in node.targets):
            continue
        task_sets = ast.literal_eval(node.value)
        if not isinstance(task_sets, dict):
            raise ValueError(f"Newt TASK_SET in {path} is not a dictionary.")
        return {str(name): list(tasks) for name, tasks in task_sets.items()}
    raise ValueError(f"Newt task-set definitions cannot be found in {path}; missing TASK_SET assignment.")


def _discount_factor(task: str, task_metadata: dict[str, Any]) -> float:
    """Return Newt's discount_factor, falling back to Newt/MBDPO's 0.99 default."""
    if "discount_factor" not in task_metadata:
        return _DEFAULT_DISCOUNT_FACTOR
    try:
        return float(task_metadata["discount_factor"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid discount_factor for MMBench task {task!r}.") from exc


def _infer_extra_task_domain(task: str) -> str:
    """Infer held-out task domains from Newt's original task-name conventions."""
    if task.startswith("ms-"):
        return "maniskill"
    if task.startswith("og-"):
        return "ogbench"
    if task.startswith("pygame-"):
        return "pygame"
    if task.startswith(("spinner-", "jumper-", "giraffe-")):
        return "dmcontrol-ext"
    if task.startswith(("walker-", "cartpole-", "cheetah-", "reacher-", "hopper-", "pendulum-", "cup-", "finger-", "fish-", "acrobot-", "quadruped-")):
        return "dmcontrol"
    raise ValueError(f"Cannot infer Newt/MMBench domain for held-out task {task!r}.")


def _domain_for_task(task: str, source_domain_tasks: dict[str, list[str]]) -> str:
    for domain, tasks in source_domain_tasks.items():
        if task in tasks:
            return domain
    return _infer_extra_task_domain(task)


def _build_domain_tasks(
    source_domain_tasks: dict[str, list[str]], all_tasks: list[str], task_metadata: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    domain_tasks = {domain: [task for task in tasks if task in task_metadata] for domain, tasks in source_domain_tasks.items()}
    for task in all_tasks:
        domain = _domain_for_task(task, source_domain_tasks)
        domain_tasks.setdefault(domain, [])
        if task not in domain_tasks[domain]:
            domain_tasks[domain].append(task)
    return domain_tasks


def _representative_domain_tasks(domain_tasks: dict[str, list[str]]) -> list[str]:
    return [tasks[0] for _, tasks in domain_tasks.items() if tasks]


def _flatten_domains(domain_tasks: dict[str, list[str]], domains: tuple[str, ...]) -> list[str]:
    return [task for domain in domains for task in domain_tasks.get(domain, [])]


def _padded_action_dims(source_action_dims: dict[str, int]) -> dict[str, int]:
    """Expose per-task action dims with the MMBench padded max-action ceiling represented."""
    if not source_action_dims:
        return {}
    raw_max = max(source_action_dims.values())
    action_dims = dict(source_action_dims)
    if raw_max < _EXPECTED_PADDED_ACTION_DIM:
        for task, dim in source_action_dims.items():
            if dim == raw_max:
                action_dims[task] = _EXPECTED_PADDED_ACTION_DIM
    return action_dims


def _load_mmbench_metadata(newt_root: str | Path | None = None) -> dict[str, Any]:
    root = _resolve_newt_root(newt_root)
    tasks_json = _load_tasks_json(root)
    newt_task_sets = _load_newt_task_sets(root)
    if _TRAIN_TASK_SET_NAME not in newt_task_sets:
        raise ValueError("Newt task-set definitions do not include the required 'soup' training set.")

    train_tasks = list(newt_task_sets[_TRAIN_TASK_SET_NAME])
    missing_train_tasks = [task for task in train_tasks if task not in tasks_json]
    if missing_train_tasks:
        raise ValueError(
            "Newt 'soup' task set references tasks missing from tasks.json: "
            + ", ".join(missing_train_tasks)
        )

    all_tasks = list(tasks_json.keys())
    extra_tasks = [task for task in all_tasks if task not in set(train_tasks)]
    source_domain_tasks = {
        domain: list(tasks)
        for domain, tasks in newt_task_sets.items()
        if domain != _TRAIN_TASK_SET_NAME
    }
    if not source_domain_tasks:
        raise ValueError("Newt task-set definitions do not include any per-domain task sets.")
    domain_tasks = _build_domain_tasks(source_domain_tasks, all_tasks, tasks_json)

    train_set = set(train_tasks)
    extra_set = set(extra_tasks)
    normalized_metadata: dict[str, dict[str, Any]] = {}
    source_action_dims: dict[str, int] = {}
    episode_lengths: dict[str, int] = {}
    discounts: dict[str, float] = {}
    text_embeddings: dict[str, list[float]] = {}

    for task in all_tasks:
        raw_metadata = tasks_json[task]
        action_dim = int(raw_metadata["action_dim"])
        episode_length = int(raw_metadata["max_episode_steps"])
        discount = _discount_factor(task, raw_metadata)
        text_embedding = list(raw_metadata["text_embedding"])
        domain = _domain_for_task(task, source_domain_tasks)

        source_action_dims[task] = action_dim
        episode_lengths[task] = episode_length
        discounts[task] = discount
        text_embeddings[task] = text_embedding
        normalized_metadata[task] = {
            "task": task,
            "domain": domain,
            "embodiment": raw_metadata.get("embodiment"),
            "instruction": raw_metadata.get("instruction"),
            "action_dim": action_dim,
            "episode_length": episode_length,
            "discount": discount,
            "text_embedding": text_embedding,
            "is_train_task": task in train_set,
            "is_extra_task": task in extra_set,
        }

    action_dims = _padded_action_dims(source_action_dims)
    text_embedding_dims = {len(embedding) for embedding in text_embeddings.values()}
    if len(text_embedding_dims) != 1:
        raise ValueError(f"Expected a single MMBench text embedding dimension, got {sorted(text_embedding_dims)}.")

    aliases = {
        "newt_soup": train_tasks,
        "mmbench200": train_tasks,
        "mmbench_debug10": _representative_domain_tasks(domain_tasks),
        "mmbench_domain10": _representative_domain_tasks(domain_tasks),
        "mmbench_overlap": _flatten_domains(domain_tasks, _OVERLAP_DOMAINS),
        "mmbench_new_domains": _flatten_domains(domain_tasks, _NEW_DOMAINS),
    }

    return {
        "tasks": all_tasks,
        "train_tasks": train_tasks,
        "extra_tasks": extra_tasks,
        "domain_tasks": domain_tasks,
        "task_metadata": normalized_metadata,
        "task_ids": {task: index for index, task in enumerate(all_tasks)},
        "action_dims": action_dims,
        "episode_lengths": episode_lengths,
        "discounts": discounts,
        "text_embeddings": text_embeddings,
        "max_obs_dim": _EXPECTED_MAX_OBS_DIM,
        "max_action_dim": max(action_dims.values()),
        "text_embedding_dim": text_embedding_dims.pop(),
        "domain_names": tuple(domain_tasks.keys()),
        "task_set_aliases": aliases,
        "source_action_dims": source_action_dims,
    }


_MMBENCH = _load_mmbench_metadata()

MMBENCH_TASKS = _MMBENCH["tasks"]
MMBENCH_TRAIN_200 = _MMBENCH["train_tasks"]
MMBENCH_EXTRA_TASKS = _MMBENCH["extra_tasks"]
MMBENCH_DOMAIN_TASKS = _MMBENCH["domain_tasks"]
MMBENCH_TASK_METADATA = _MMBENCH["task_metadata"]
MMBENCH_TASK_IDS = _MMBENCH["task_ids"]
MMBENCH_ACTION_DIMS = _MMBENCH["action_dims"]
MMBENCH_EPISODE_LENGTHS = _MMBENCH["episode_lengths"]
MMBENCH_DISCOUNTS = _MMBENCH["discounts"]
MMBENCH_TEXT_EMBEDDINGS = _MMBENCH["text_embeddings"]
MMBENCH_MAX_OBS_DIM = _MMBENCH["max_obs_dim"]
MMBENCH_MAX_ACTION_DIM = _MMBENCH["max_action_dim"]
MMBENCH_TEXT_EMBEDDING_DIM = _MMBENCH["text_embedding_dim"]
MMBENCH_DOMAIN_NAMES = _MMBENCH["domain_names"]
MMBENCH_TASK_SET_ALIASES = _MMBENCH["task_set_aliases"]
MMBENCH_SOURCE_ACTION_DIMS = _MMBENCH["source_action_dims"]

__all__ = [
    "MMBENCH_TASKS",
    "MMBENCH_TRAIN_200",
    "MMBENCH_EXTRA_TASKS",
    "MMBENCH_DOMAIN_TASKS",
    "MMBENCH_TASK_METADATA",
    "MMBENCH_TASK_IDS",
    "MMBENCH_ACTION_DIMS",
    "MMBENCH_EPISODE_LENGTHS",
    "MMBENCH_DISCOUNTS",
    "MMBENCH_TEXT_EMBEDDINGS",
    "MMBENCH_MAX_OBS_DIM",
    "MMBENCH_MAX_ACTION_DIM",
    "MMBENCH_TEXT_EMBEDDING_DIM",
    "MMBENCH_DOMAIN_NAMES",
    "MMBENCH_TASK_SET_ALIASES",
    "MMBENCH_SOURCE_ACTION_DIMS",
]
