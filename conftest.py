# MIT License
#
# Copyright (c) 2025 nimiCurtis
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# conftest.py
from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
import pytest


# --------------------------------------------------------------------------------------
# Early: stop pytest from autoloading unrelated system plugins (e.g., ROS launch_testing)
# --------------------------------------------------------------------------------------
def pytest_load_initial_conftests(early_config, parser, args) -> None:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


# --------------------------------------------------------------------------------------
# Keep imports working even when running `pytest` from the repo root
# --------------------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------------------
# Global test configuration: logging + numpy print options
# --------------------------------------------------------------------------------------
def pytest_configure(config) -> None:
    # Logging
    level_name = os.getenv("TEST_LOGLEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname).1s %(asctime)s %(name)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Numpy pretty print to make failures easier to read
    np.set_printoptions(precision=4, suppress=True, linewidth=120)


# =====================================
# Re-usable DUMMY “ROS-like” messages
# =====================================


@dataclass
class DummyImage:
    width: int
    height: int
    encoding: str = "rgb8"

    def __post_init__(self) -> None:
        channels = 3 if self.encoding in {"rgb8", "bgr8"} else 1
        value = np.arange(self.width * self.height * channels, dtype=np.uint8)
        self.data = value.tobytes()


@dataclass
class DummyJointState:
    name: list[str]
    position: list[float]


@dataclass
class DummyArray:
    data: list[float]


# =====================================
# Fixtures
# =====================================


@pytest.fixture(scope="session")
def random_seed() -> int:
    """
    A fixed seed for reproducibility across tests.
    """
    seed = int(os.getenv("TEST_SEED", "1337"))
    random.seed(seed)
    np.random.seed(seed)
    # Torch is optional — seed it if available
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


@pytest.fixture
def tmp_out_dir(tmp_path: Path) -> Path:
    """
    Create a standard output folder layout for tests:
        <tmp>/output/.logs
    """
    out = tmp_path / "output"
    (out / ".logs").mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def dummy_image_factory() -> Callable[[int, int, str], DummyImage]:
    """
    Factory to create DummyImage instances.
    Usage: img = dummy_image_factory(320, 240, "mono8")
    """

    def _make(width: int, height: int, encoding: str = "rgb8") -> DummyImage:
        return DummyImage(width=width, height=height, encoding=encoding)

    return _make


@pytest.fixture
def dummy_jointstate_factory() -> Callable[[Iterable[str], Iterable[float]], DummyJointState]:
    """
    Factory to create DummyJointState instances.
    Usage: js = dummy_jointstate_factory(names, positions)
    """

    def _make(names: Iterable[str], positions: Iterable[float]) -> DummyJointState:
        return DummyJointState(name=list(names), position=list(positions))

    return _make


@pytest.fixture
def dummy_array_factory() -> Callable[[Iterable[float]], DummyArray]:
    """
    Factory to create DummyArray instances.
    Usage: arr = dummy_array_factory([0, 1, 2, 3, 4, 5])
    """

    def _make(data: Iterable[float]) -> DummyArray:
        return DummyArray(data=list(data))

    return _make


@pytest.fixture
def make_config_yaml(tmp_path: Path) -> Callable[[Dict[str, Any]], Path]:
    """
    Factory: write a minimal JSON-as-YAML config file and return its path.
    JSON is a valid subset of YAML, so Config.from_yaml can read it safely.
    Pass a dict of overrides; sensible defaults are provided.

    Usage:
        path = make_config_yaml({"root": "/tmp/out", "robot_type": "so101"})
    """

    def _make(overrides: Dict[str, Any]) -> Path:
        defaults: Dict[str, Any] = {
            "root": str(tmp_path),
            "repo_id": "org/dataset",
            "robot_type": "so101",
            "episode_per_bag": True,
            "downsample_by": 1,
            "use_lerobot_ranges_norms": False,
            "force": False,
            "sync_reference": "state",
            "sync_tolerance_s": 0.1,
            "use_videos": False,
            "fps": 30,
            "video": {},
            "images": {"wrist": {"key": "wrist_rgb", "shape": [3, 2, 2], "topic": "/camera"}},
            "state": {"key": "state", "size": 6},
            "action": {"key": "action", "size": 6},
            "joint_order": [],
            "topics": {"state": "/joint_state", "action": "/target"},
            "bags_root": "/bags",
            "task_text": "pick",
        }
        cfg = {**defaults, **overrides}
        path = tmp_path / "config.json"  # valid YAML
        with path.open("w") as f:
            json.dump(cfg, f, indent=2)
        return path

    return _make


@pytest.fixture
def topic_buffer_factory():
    """
    Factory to build and prefill a TopicBuffer for tests.

    Usage:
        buf = topic_buffer_factory([(0.0, "first"), (0.1, "second")])
    """
    from so101_rosbag2lerobot_dataset.sync import TopicBuffer

    def _make(stamps_data: Iterable[Tuple[float, Any]]) -> TopicBuffer:
        buf = TopicBuffer()
        for t, d in stamps_data:
            buf.add(float(t), d)
        buf.finalize()
        return buf

    return _make
