from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Literal
from pathlib import Path

import logging
from tqdm import tqdm

from .config import Config
from .io_bag import Rosbag2Reader
from .sync import TopicBuffer
from .utils import (
    ros_image_to_chw_float01,
    jointstate_to_vec6,
    float64multiarray_to_vec6,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class FrameBuffers:
    images: Dict[str, TopicBuffer]
    state: TopicBuffer
    action: TopicBuffer


class RosbagToLeRobotConverter:
    def __init__(self, cfg: Config, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)

    def _create_dataset(self) -> LeRobotDataset:
        features = {
            self.cfg.state.key: {
                "dtype": "float32",
                "shape": [self.cfg.state.size],
                "names": {"motors": [f"j{i}" for i in range(self.cfg.state.size)]},
            },
            self.cfg.action.key: {
                "dtype": "float32",
                "shape": [self.cfg.action.size],
                "names": {"motors": [f"j{i}" for i in range(self.cfg.action.size)]},
            },
        }
        for name, stream in self.cfg.images.items():
            features[stream.key] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": list(stream.shape),
                "names": ["channel", "height", "width"],
                "video_info": {
                    "video.fps": float(self.cfg.fps),
                    "video.codec": self.cfg.video.codec,
                    "video.pix_fmt": self.cfg.video.pix_fmt,
                    "video.is_depth_map": bool(self.cfg.video.is_depth_map),
                    "has_audio": bool(self.cfg.video.has_audio),
                } if self.cfg.use_videos else None,
            }
        ds = LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            features=features,
            root=self.cfg.out_dir,
            fps=self.cfg.fps,
            use_videos=self.cfg.use_videos,
        )
        self.log.info("Created dataset at %s (fps=%s, videos=%s)", self.cfg.out_dir, self.cfg.fps, self.cfg.use_videos)
        return ds

    def _buffers_from_bag(self, bag_path: Path) -> FrameBuffers:
        self.log.info("Reading bag: %s", bag_path.name)
        reader = Rosbag2Reader(bag_path)

        IMG_T = "sensor_msgs/msg/Image"
        JS_T  = "sensor_msgs/msg/JointState"
        F64_T = "std_msgs/msg/Float64MultiArray"   # <<— action type

        bufs = FrameBuffers(
            images={name: TopicBuffer() for name in self.cfg.images},
            state=TopicBuffer(),
            action=TopicBuffer(),
        )

        for msg in reader.iter_messages():
            # images
            matched_image = False
            for name, stream in self.cfg.images.items():
                if msg.topic == stream.topic:
                    des = reader.deserialize(msg.raw, IMG_T)
                    bufs.images[name].add(msg.t_sec, des)
                    matched_image = True
            if matched_image:
                continue

            # state (JointState) and action (Float64MultiArray)
            if msg.topic == self.cfg.topics.state:
                des = reader.deserialize(msg.raw, JS_T)
                bufs.state.add(msg.t_sec, des)
            elif msg.topic == self.cfg.topics.action:
                des = reader.deserialize(msg.raw, F64_T)
                bufs.action.add(msg.t_sec, des)

        for b in bufs.images.values():
            b.finalize()
        bufs.state.finalize()
        bufs.action.finalize()
        return bufs

    def _parse_sync_reference(self) -> Tuple[Literal["image", "state", "action"], Optional[str]]:
        ref = (self.cfg.sync_reference or "").strip()
        if ref.startswith("image:"):
            name = ref.split(":", 1)[1].strip() if ":" in ref else ""
            if not name:
                raise ValueError("sync_reference='image:<name>' must include a non-empty <name>")
            if name not in self.cfg.images:
                raise ValueError(
                    f"sync_reference image '{name}' not found. Available: {list(self.cfg.images.keys())}"
                )
            return "image", name
        if ref in ("state", "action"):
            return ref, None
        # Fallback: default to 'state' to avoid None and log a warning
        self.log.warning("Unknown sync_reference '%s'; defaulting to 'state'", ref)
        return "state", None
    
    def _iter_synced(self, bufs: FrameBuffers):
        ref_kind, ref_name = self._parse_sync_reference()
        tol = self.cfg.sync_tolerance_s

        if ref_kind == "image":
            # ref_name is guaranteed valid/non-empty by _parse_sync_reference
            ref_buf = bufs.images[ref_name]  # type: ignore[index]
            other_image_names = [n for n in bufs.images.keys() if n != ref_name]
        else:
            # avoid getattr; map is type-safe
            ref_map = {"state": bufs.state, "action": bufs.action}
            ref_buf = ref_map[ref_kind]
            other_image_names = list(bufs.images.keys())

        self.log.info("Sync reference: %s (frames=%d)", self.cfg.sync_reference, len(ref_buf.t))
        keep_count = 0
        total_ref = len(ref_buf.t)
        for i in tqdm(range(total_ref), desc="Sync frames", leave=False):
            tref = ref_buf.t[i]
            result = {"t": tref, "ref": ref_buf.d[i]}

            ok = True
            for name in other_image_names:
                cand = bufs.images[name].nearest(tref, tol)
                if cand is None:
                    ok = False
                    break
                result[f"image:{name}"] = cand.data
            if not ok:
                continue

            for k in ("state", "action"):
                buf = getattr(bufs, k)
                cand = buf.nearest(tref, tol)
                if cand is None:
                    ok = False
                    break
                result[k] = cand.data
            if not ok:
                continue

            keep_count += 1
            if self.cfg.downsample_by > 1 and (keep_count % self.cfg.downsample_by) != 0:
                continue
            yield result

    def convert(self) -> int:
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        ds = self._create_dataset()

        all_bags = list(self._discover_bags(Path(self.cfg.bags_root)))
        self.log.info("Discovered %d bag(s) in %s", len(all_bags), self.cfg.bags_root)

        total_frames = 0
        for bag in tqdm(all_bags, desc="Bags", unit="bag"):
            bufs = self._buffers_from_bag(bag)
            episode_frames = 0
            dropped = 0

            ref_kind, ref_name = self._parse_sync_reference()
            if ref_kind == "image":
                ref_len = len(bufs.images[ref_name].t)  # type: ignore[index]
            else:
                ref_map = {"state": bufs.state, "action": bufs.action}
                ref_len = len(ref_map[ref_kind].t)

            for m in self._iter_synced(bufs):
                st6 = jointstate_to_vec6(m["state"], self.cfg.joint_order)
                ac6 = float64multiarray_to_vec6(m["action"], size=self.cfg.action.size)

                frame = {
                    self.cfg.state.key: st6.astype("float32"),
                    self.cfg.action.key: ac6.astype("float32"),
                }

                for name, stream in self.cfg.images.items():
                    img_msg = m.get(f"image:{name}") if f"image:{name}" in m else (m["ref"] if self.cfg.sync_reference == f"image:{name}" else None)
                    if img_msg is None:
                        dropped += 1
                        break
                    frame[stream.key] = ros_image_to_chw_float01(img_msg)
                else:
                    ds.add_frame(frame=frame, task=self.cfg.task_text)
                    episode_frames += 1
                    total_frames += 1

            if episode_frames > 0:
                ds.save_episode()
                ds.clear_episode_buffer()

            self.log.info("Bag %s → episode frames: %d (ref=%d, dropped=%d)", bag.name, episode_frames, ref_len, dropped)

        self.log.info("Completed. Total frames written: %d", total_frames)
        return total_frames

    @staticmethod
    def _discover_bags(root: Path):
        from .io_bag import discover_bags
        yield from discover_bags(root)
