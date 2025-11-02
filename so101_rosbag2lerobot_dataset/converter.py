from __future__ import annotations
from dataclasses import dataclass
from pyexpat import features
from typing import Dict, Optional, Tuple, Literal
from pathlib import Path

import logging
import numpy as np
from tqdm import tqdm

from .config import Config
from .io_bag import Rosbag2Reader
from .sync import TopicBuffer, SyncStats
from .utils import (
    ros_image_to_hwc_float01,
    ros_jointstate_to_vec6,
    ros_float64multiarray_to_vec6,
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

    def _assert_schema(self, features):
        for k, f in features.items():
            if f["dtype"] in ("image","video"):
                assert f["names"] == ["height","width","channel"], f"{k}: wrong names"
                h,w,c = f["shape"]
                assert c in (1,3), f"{k}: unexpected channels {c}"
        
        # TODO: motors ->> check name in config match names from rosparams 
        

    def _create_dataset(self) -> LeRobotDataset:
        """Create a LeRobot dataset from the extracted features.

        Returns:
            LeRobotDataset: The created dataset.
        """
        features = {
            self.cfg.state.key: {
                "dtype": "float32",
                "shape": (self.cfg.state.size,),
                "names": {"motors": [f"j{i}" for i in range(self.cfg.state.size)]},
            },
            self.cfg.action.key: {
                "dtype": "float32",
                "shape": (self.cfg.action.size,),
                "names": {"motors": [f"j{i}" for i in range(self.cfg.action.size)]},
            },
        }

        # Validate schema
        self._assert_schema(features)

        # images
        for name, stream in self.cfg.images.items():
            features[stream.key] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": list(stream.shape),  # HWC
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(self.cfg.fps),
                    "video.codec": self.cfg.video.codec,
                    "video.pix_fmt": self.cfg.video.pix_fmt,
                    "video.is_depth_map": bool(self.cfg.video.is_depth_map),
                    "has_audio": bool(self.cfg.video.has_audio),
                    "video_backend": self.cfg.video.backend,
                    "image_writer_processes": self.cfg.video.writer_processes,
                    "image_writer_threads": self.cfg.video.writer_threads,
                } if self.cfg.use_videos else None,
            }

        ds = LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            features=features,
            root=self.cfg.root,
            robot_type=self.cfg.robot_type,
            fps=self.cfg.fps,
            use_videos=self.cfg.use_videos,
            video_backend=self.cfg.video.backend,
            image_writer_processes=self.cfg.video.writer_processes,
            image_writer_threads=self.cfg.video.writer_threads,
        )
        self.log.info("Created dataset at %s (fps=%s, videos=%s)", self.cfg.root, self.cfg.fps, self.cfg.use_videos)
        return ds

    def _buffers_from_bag(self, bag_path: Path) -> FrameBuffers:
        self.log.info("Reading bag: %s", bag_path.name)
        reader = Rosbag2Reader(bag_path, self.cfg.force, logger=self.log)

        if reader.processed:
            self.log.warning(f"Skipping already processed bag: {bag_path}")
            return FrameBuffers(images={}, state=TopicBuffer(), action=TopicBuffer())
        
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
        
        reader.close()
        
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
    
    def _log_sync_report(self, stats: Dict[str, SyncStats]):
        # Pretty print a one-line summary per topic
        for name, st in stats.items():
            s = st.summary()
            self.log.info(
                "[sync] %-22s tried=%4d matched=%4d rate=%.1f%% |median|dt|=%.4fs "
                "mean=%.4fs p95=%.4fs max=%.4fs",
                name,
                st.tried,
                st.matched,
                100.0 * s["match_rate"],
                s["median_abs_dt_s"],
                s["mean_abs_dt_s"],
                s["p95_abs_dt_s"],
                s["max_abs_dt_s"],
            )
            
    def _iter_synced(self, bufs: FrameBuffers):
        ref_kind, ref_name = self._parse_sync_reference()
        tol = self.cfg.sync_tolerance_s

        if ref_kind == "image":
            ref_buf = bufs.images[ref_name]  # type: ignore[index]
            other_image_names = [n for n in bufs.images.keys() if n != ref_name]
        else:
            ref_map = {"state": bufs.state, "action": bufs.action}
            ref_buf = ref_map[ref_kind]
            other_image_names = list(bufs.images.keys())

        # --- NEW: stats collectors per topic (and for state/action) ---
        stats: Dict[str, SyncStats] = {}
        for nm in other_image_names:
            stats[f"image:{nm}"] = SyncStats()
        stats["state"] = SyncStats()
        stats["action"] = SyncStats()

        self.log.info("Sync reference: %s (frames=%d)", self.cfg.sync_reference, len(ref_buf.t))
        keep_count = 0
        total_ref = len(ref_buf.t)

        for i in tqdm(range(total_ref), desc="Sync frames", leave=False):
            tref = ref_buf.t[i]
            result = {"t": tref, "ref": ref_buf.d[i]}

            ok = True

            # images first
            for name in other_image_names:
                buf = bufs.images[name]
                cand, dt = buf.nearest_with_dt(tref, tol)
                stats_key = f"image:{name}"
                stats[stats_key].add(cand is not None, dt)
                if cand is None:
                    ok = False
                    # don't break here; still record stats for state/action for a fair "tried" count?
                    # We'll break to keep previous logic strict:
                    break
                result[f"image:{name}"] = cand.data
            if not ok:
                continue

            # state & action
            for k in ("state", "action"):
                buf = getattr(bufs, k)
                cand, dt = buf.nearest_with_dt(tref, tol)
                stats[k].add(cand is not None, dt)
                if cand is None:
                    ok = False
                    break
                result[k] = cand.data
            if not ok:
                continue

            keep_count += 1
            if self.cfg.downsample_by > 1 and (keep_count % self.cfg.downsample_by) != 0:
                # Even if we skip yielding, we still counted "tried" and "matched" above, which is fine.
                continue

            yield result

        self._log_sync_report(stats)

    def convert(self) -> int:
        """Convert all rosbag files under cfg.bags_root into a LeRobotDataset."""
        # Note: Directory creation is now handled in cli.py via get_versioned_dataset_dir
        ds = self._create_dataset()

        all_bags = list(self._discover_bags(Path(self.cfg.bags_root)))
        self.log.info("Discovered %d bag(s) in %s", len(all_bags), self.cfg.bags_root)

        total_frames = 0
        for bag in tqdm(all_bags, desc="Processed Bags", unit="bag"):
            bufs = self._buffers_from_bag(bag)
            if not bufs.state.t:
                self.log.warning("Skipping bag %s: no state messages found.", bag.name)
                continue
            
            episode_frames = 0
            dropped = 0

            ref_kind, ref_name = self._parse_sync_reference()
            if ref_kind == "image":
                ref_len = len(bufs.images[ref_name].t)  # type: ignore[index]
            else:
                ref_map = {"state": bufs.state, "action": bufs.action}
                ref_len = len(ref_map[ref_kind].t)

            self.log.info(f"Syncing bag {bag.name} with {ref_len} reference frames...")

            for m in self._iter_synced(bufs):
                try:
                    # Convert ROS messages → numpy arrays
                    # inside the frame loop
                    st6 = np.asarray(ros_jointstate_to_vec6(m["state"], self.cfg.joint_order),
                                    dtype=np.float32).ravel()
                    ac6 = np.asarray(ros_float64multiarray_to_vec6(m["action"], size=self.cfg.action.size),
                                    dtype=np.float32).ravel()

                    # Build the frame dict expected by LeRobotDataset
                    frame = {
                        self.cfg.state.key: st6.astype("float32"),
                        self.cfg.action.key: ac6.astype("float32"),
                        "task": self.cfg.task_text,      # required
                    }

                    # Add all image streams
                    missing_image = False
                    for name, stream in self.cfg.images.items():
                        img_msg = (
                            m.get(f"image:{name}")
                            if f"image:{name}" in m
                            else (m["ref"] if self.cfg.sync_reference == f"image:{name}" else None)
                        )

                        if img_msg is None:
                            missing_image = True
                            dropped += 1
                            self.log.debug(f"Dropped frame {episode_frames} (missing image:{name})")
                            break

                        frame[stream.key] = ros_image_to_hwc_float01(img_msg)

                    if missing_image:
                        continue  # skip this frame entirely

                    # Add frame to dataset
                    ds.add_frame(frame=frame)
                    episode_frames += 1
                    total_frames += 1

                except Exception as e:
                    dropped += 1
                    self.log.error(f"Error processing frame {episode_frames} in bag {bag.name}: {e}")
                    continue

            # After finishing one bag (episode)
            if episode_frames > 0:
                ds.save_episode()
                ds.clear_episode_buffer()

            self.log.info(
                "Bag %s → frames saved: %d / ref=%d | dropped=%d",
                bag.name, episode_frames, ref_len, dropped
            )

        self.log.info("✅ Completed conversion. Total frames written: %d", total_frames)
        return total_frames

    @staticmethod
    def _discover_bags(root: Path):
        from .io_bag import discover_bags
        yield from discover_bags(root)
