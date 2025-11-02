from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import yaml

from .utils import get_versioned_pathes


@dataclass
class VideoInfo:
    codec: str = "av1"
    pix_fmt: str = "yuv420p"
    is_depth_map: bool = False
    has_audio: bool = False
    backend: str = "pyavc"  # "pyavc"
    writer_processes: int = 4
    writer_threads: int = 4


@dataclass
class VectorSpec:
    key: str
    size: int


@dataclass
class ImageStream:
    key: str
    shape: Tuple[int, int, int]  # (C, H, W)
    topic: str


@dataclass
class Topics:
    state: str
    action: str  # was 'target'


@dataclass
class Config:
    out_dir: str
    data_dir_name: str
    root: str
    repo_id: str
    robot_type: str
    episode_per_bag: bool
    downsample_by: int
    force: bool
    sync_reference: str          # "image:<name>" | "state" | "action"
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

        out_dir = y["out_dir"]
        data_dir_name = y.get("data_dir_name", "lerobot_dataset")

        _, data_path = get_versioned_pathes(out_dir, data_dir_name)
        root = data_path

        return Config(
            out_dir=out_dir,
            data_dir_name=data_dir_name,
            root=root,
            repo_id=y["repo_id"],
            robot_type=y["robot_type"],
            episode_per_bag=y.get("episode_per_bag", True),
            downsample_by=int(y.get("downsample_by", 1)),
            force=bool(y.get("force", False)),
            sync_reference=y.get("sync_reference", "image:wrist"),
            sync_tolerance_s=float(y.get("sync_tolerance_s", 0.04)),
            use_videos=bool(y.get("use_videos", True)),
            fps=int(y.get("fps", 30)),
            video=VideoInfo(**y.get("video", {})),
            images=images,
            state=VectorSpec(key=y["state"]["key"], size=int(y["state"]["size"])),
            action=VectorSpec(key=y["action"]["key"], size=int(y["action"]["size"])),
            joint_order=joint_order,
            topics=Topics(**y["topics"]),
            bags_root=y["bags_root"],
            task_text=y.get("task_text", "pick and place"),
        )
