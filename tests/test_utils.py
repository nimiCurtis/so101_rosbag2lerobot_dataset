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

import math
from pathlib import Path

import numpy as np
import pytest

from so101_rosbag2lerobot_dataset import utils
from so101_rosbag2lerobot_dataset.config import Config
from so101_rosbag2lerobot_dataset.sync import SyncStats

# ---------------------------
# Image utils
# ---------------------------


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "mono8"])
def test_ros_image_to_hwc_float01_shape(encoding, dummy_image_factory):
    msg = dummy_image_factory(width=2, height=2, encoding=encoding)
    arr = utils.ros_image_to_hwc_float01(msg)
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.float32
    assert np.all((0.0 <= arr) & (arr <= 1.0))


# ---------------------------
# Vector conversion utils
# ---------------------------


def test_ros_jointstate_to_vec6_with_order_and_norm(dummy_jointstate_factory):
    names_order = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",  # ensure gripper is recognized
    ]
    msg = dummy_jointstate_factory(names_order, [math.pi / 2] * 6)
    vec = utils.ros_jointstate_to_vec6(msg, joint_order=msg.name, use_lerobot_ranges_norms=True)

    assert vec.shape == (6,)
    assert np.allclose(vec[:-1], 50.0)
    assert np.isclose(vec[-1], 60.0)


def test_ros_float64multiarray_to_vec6_normalized(dummy_array_factory):
    msg = dummy_array_factory([math.pi / 2] * 6)
    vec = utils.ros_float64multiarray_to_vec6(msg, use_lerobot_ranges_norms=True)
    assert vec.shape == (6,)
    assert np.isclose(vec[-1], 60.0)


# ---------------------------
# Versioned output paths
# ---------------------------


def test_get_versioned_paths_creates_unique_names(tmp_out_dir: Path):
    """
    Given existing logs/data directories ending with 'dataset',
    the next call should suffix them with _2.
    """
    # conftest created: <tmp_out_dir>/logs and <tmp_out_dir>/data
    logs_dir = tmp_out_dir / "logs" / "dataset"
    data_dir = tmp_out_dir / "data" / "dataset"
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    logs_path, data_path = utils.get_versioned_paths(str(tmp_out_dir), "dataset")
    assert logs_path.endswith("dataset_2")
    assert data_path.endswith("dataset_2")


# ---------------------------
# TopicBuffer nearest / dt
# ---------------------------


def test_topic_buffer_nearest_with_dt(topic_buffer_factory):
    buf = topic_buffer_factory([(0.0, "first"), (0.1, "second")])

    stamped, dt = buf.nearest_with_dt(0.08, 0.05)
    assert stamped is not None
    assert stamped.data == "second"
    assert dt == pytest.approx(0.02, rel=1e-3)


# ---------------------------
# SyncStats
# ---------------------------


def test_sync_stats_summary():
    stats = SyncStats()
    stats.add(True, 0.01)
    stats.add(True, 0.02)
    stats.add(False, None)
    summary = stats.summary()

    assert pytest.approx(summary["match_rate"], rel=1e-6) == 2 / 3
    assert summary["max_abs_dt_s"] == 0.02


# ---------------------------
# Config from YAML
# ---------------------------


def test_config_from_yaml(make_config_yaml):
    config_path = make_config_yaml({})
    cfg = Config.from_yaml(str(config_path))

    assert cfg.images["wrist"].topic == "/camera"
    assert cfg.root.endswith("dataset")
    assert cfg.topics.state == "/joint_state"
