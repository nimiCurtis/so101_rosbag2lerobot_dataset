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


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

from .utils import get_versioned_paths


@dataclass
class VideoInfo:
    """Configuration describing how image streams are stored on disk."""

    codec: str = "av1"
    pix_fmt: str = "yuv420p"
    is_depth_map: bool = False
    has_audio: bool = False
    writer_processes: int = 4
    writer_threads: int = 4


@dataclass
class VectorSpec:
    """Metadata describing a state or action vector stored in the dataset."""

    key: str
    size: int
    msg_type: Optional[str] = None


@dataclass
class ImageStream:
    """Description of a ROS topic that carries image data."""

    key: str
    shape: Tuple[int, int, int]  # (H, W, C)
    topic: str


@dataclass
class Topics:
    """Container listing ROS topics of interest for the conversion."""

    state: str
    action: str  # was 'target'


@dataclass
class Config:
    """Configuration for converting ROS 2 bag files into LeRobot datasets."""

    data_dir_name: str
    root: str
    repo_id: str
    robot_type: str
    episode_per_bag: bool
    downsample_by: int
    use_lerobot_ranges_norms: bool
    force: bool
    upload_to_hub: bool
    sync_reference: str  # "image:<name>" | "state" | "action"
    sync_tolerance_s: float
    use_videos: bool
    fps: int
    video: VideoInfo
    images: Dict[str, ImageStream]
    state: VectorSpec
    action: VectorSpec
    joint_order: Optional[List[str]]  # 6 names or None
    topics: Topics
    bags_root: str
    task_text: str

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Create a :class:`Config` instance from a YAML configuration file.

        Args:
            path: Path to a configuration YAML file.

        Returns:
            A fully populated :class:`Config` object ready for conversion.
        """

        with open(path, "r") as f:
            y = yaml.safe_load(f)

        images = {
            name: ImageStream(
                key=spec["key"],
                shape=tuple(spec["shape"]),
                topic=spec["topic"],
            )
            for name, spec in y["images"].items()
        }

        joint_order = y.get("joint_order", None)
        if joint_order is not None and len(joint_order) == 0:
            joint_order = None

        out_dir = y.get("out_dir", "output")
        data_dir_name = y.get("data_dir_name", "lerobot_dataset")

        _, data_path = get_versioned_paths(out_dir, data_dir_name)
        root = data_path

        def _vector_spec(spec: Dict[str, object]) -> VectorSpec:
            msg_type = spec.get("msg_type") or spec.get("type")
            return VectorSpec(
                key=str(spec["key"]),
                size=int(spec["size"]),
                msg_type=str(msg_type) if msg_type else None,
            )

        return Config(
            data_dir_name=data_dir_name,
            root=root,
            repo_id=y["repo_id"],
            robot_type=y["robot_type"],
            episode_per_bag=y.get("episode_per_bag", True),
            downsample_by=int(y.get("downsample_by", 1)),
            use_lerobot_ranges_norms=bool(y.get("use_lerobot_ranges_norms", False)),
            force=bool(y.get("force", False)),
            upload_to_hub=bool(y.get("upload_to_hub", False)),
            sync_reference=y.get("sync_reference", "image:wrist"),
            sync_tolerance_s=float(y.get("sync_tolerance_s", 0.04)),
            use_videos=bool(y.get("use_videos", True)),
            fps=int(y.get("fps", 30)),
            video=VideoInfo(**y.get("video", {})),
            images=images,
            state=_vector_spec(y["state"]),
            action=_vector_spec(y["action"]),
            joint_order=joint_order,
            topics=Topics(**y["topics"]),
            bags_root=y["bags_root"],
            task_text=y.get("task_text", "pick and place"),
        )
