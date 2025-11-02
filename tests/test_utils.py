import math
from pathlib import Path

import numpy as np
import pytest

from so101_rosbag2lerobot_dataset import utils
from so101_rosbag2lerobot_dataset.config import Config
from so101_rosbag2lerobot_dataset.sync import SyncStats, TopicBuffer


class DummyImage:
    def __init__(self, width: int, height: int, encoding: str = "rgb8"):
        self.width = width
        self.height = height
        self.encoding = encoding
        channels = 3 if encoding in {"rgb8", "bgr8"} else 1
        value = np.arange(width * height * channels, dtype=np.uint8)
        self.data = value.tobytes()


class DummyJointState:
    def __init__(self, name, position):
        self.name = name
        self.position = position


class DummyArray:
    def __init__(self, data):
        self.data = data


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "mono8"])
def test_ros_image_to_hwc_float01_shape(encoding):
    msg = DummyImage(width=2, height=2, encoding=encoding)
    arr = utils.ros_image_to_hwc_float01(msg)
    assert arr.shape == (2, 2, 3)
    assert arr.dtype == np.float32
    assert np.all((0.0 <= arr) & (arr <= 1.0))


def test_ros_jointstate_to_vec6_with_order_and_norm():
    msg = DummyJointState(name=[f"joint_{i}" for i in range(6)], position=[math.pi / 2] * 6)
    vec = utils.ros_jointstate_to_vec6(msg, joint_order=msg.name, use_lerobot_ranges_norms=True)
    assert vec.shape == (6,)
    # Normalization scales pi/2 to 50 for non-gripper joints
    assert np.allclose(vec[:-1], 50.0)
    # The last joint (gripper) adds offset 10
    assert np.isclose(vec[-1], 60.0)


def test_ros_float64multiarray_to_vec6_normalized():
    msg = DummyArray(data=[math.pi / 2] * 6)
    vec = utils.ros_float64multiarray_to_vec6(msg, use_lerobot_ranges_norms=True)
    assert vec.shape == (6,)
    assert np.isclose(vec[-1], 60.0)


def test_get_versioned_pathes_creates_unique_names(tmp_path: Path):
    out_dir = tmp_path / "output"
    logs_dir = out_dir / "logs" / "dataset"
    data_dir = out_dir / "data" / "dataset"
    logs_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    logs_path, data_path = utils.get_versioned_pathes(str(out_dir), "dataset")
    assert logs_path.endswith("dataset_2")
    assert data_path.endswith("dataset_2")


def test_topic_buffer_nearest_with_dt():
    buf = TopicBuffer()
    buf.add(0.0, "first")
    buf.add(0.1, "second")
    buf.finalize()

    stamped, dt = buf.nearest_with_dt(0.08, 0.05)
    assert stamped is not None
    assert stamped.data == "second"
    assert dt == pytest.approx(0.02, rel=1e-3)


def test_sync_stats_summary():
    stats = SyncStats()
    stats.add(True, 0.01)
    stats.add(True, 0.02)
    stats.add(False, None)
    summary = stats.summary()
    assert pytest.approx(summary["match_rate"], rel=1e-6) == 2 / 3
    assert summary["max_abs_dt_s"] == 0.02


def test_config_from_yaml(tmp_path: Path):
    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(
        """
        out_dir: {out_dir}
        data_dir_name: dataset
        repo_id: org/dataset
        robot_type: so101
        episode_per_bag: true
        downsample_by: 1
        use_lerobot_ranges_norms: false
        force: false
        sync_reference: state
        sync_tolerance_s: 0.1
        use_videos: false
        fps: 30
        video: {{}}
        images:
          wrist:
            key: wrist_rgb
            shape: [3, 2, 2]
            topic: /camera
        state:
          key: state
          size: 6
        action:
          key: action
          size: 6
        joint_order: []
        topics:
          state: /joint_state
          action: /target
        bags_root: /bags
        task_text: pick
        """.format(
            out_dir=tmp_path
        )
    )

    cfg = Config.from_yaml(str(config_yaml))
    assert cfg.images["wrist"].topic == "/camera"
    assert cfg.root.endswith("dataset")
    assert cfg.topics.state == "/joint_state"
