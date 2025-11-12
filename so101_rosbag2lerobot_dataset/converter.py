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


from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import sanity_check_dataset_name
from tqdm import tqdm

from .config import Config, VectorSpec
from .constants import F64_T, IMG_T, JS_T
from .io_bag import Rosbag2Reader
from .sync import SyncStats, TopicBuffer
from .utils import (
    ros_float64multiarray_to_vec6,
    ros_image_to_hwc_float01,
    ros_jointstate_to_vec6,
)


@dataclass
class FrameBuffers:
    """Collection of synchronized buffers for a single rosbag."""

    images: Dict[str, TopicBuffer]
    state: TopicBuffer
    action: TopicBuffer
    state_type: Optional[str] = None
    action_type: Optional[str] = None


class RosbagToLeRobotConverter:
    """Convert ROS 2 bag files into LeRobot datasets based on a configuration."""

    def __init__(self, cfg: Config, logger: Optional[logging.Logger] = None):
        """Initialize the converter with configuration and optional logger."""

        self.cfg = cfg
        self.log = logger or logging.getLogger(__name__)

    def _assert_schema(self, features):
        """Validate the generated dataset schema matches image expectations."""

        for k, f in features.items():
            if f["dtype"] in ("image", "video"):
                assert f["names"] == ["height", "width", "channels"], f"{k}: wrong names"
                h, w, c = f["shape"]
                assert c in (1, 3), f"{k}: unexpected channels {c}"

    def _create_dataset(self) -> LeRobotDataset:
        """Create and configure the target :class:`LeRobotDataset` instance."""

        ## Assert joint_order length is equal to state/action size if provided
        if self.cfg.joint_order is not None:
            if len(self.cfg.joint_order) != self.cfg.state.size:
                raise ValueError(
                    f"Length of joint_order ({len(self.cfg.joint_order)}) does not match "
                    f"state size ({self.cfg.state.size})"
                )
            if len(self.cfg.joint_order) != self.cfg.action.size:
                raise ValueError(
                    f"Length of joint_order ({len(self.cfg.joint_order)}) does not match "
                    f"action size ({self.cfg.action.size})"
                )

        # Use a ternary operator to build the names list.
        # If joint_order is provided, use it to create names like "joint_name.pos".
        # Otherwise, fall back to default names like "j0", "j1", etc.
        state_names = (
            [f"{joint}.pos" for joint in self.cfg.joint_order]
            if self.cfg.joint_order
            else [f"j{i}" for i in range(self.cfg.state.size)]
        )

        action_names = (
            [f"{joint}.pos" for joint in self.cfg.joint_order]
            if self.cfg.joint_order
            else [f"j{i}" for i in range(self.cfg.action.size)]
        )

        features = {
            self.cfg.state.key: {
                "dtype": "float32",
                "shape": (self.cfg.state.size,),
                # Use the flat list of names
                "names": state_names,
            },
            self.cfg.action.key: {
                "dtype": "float32",
                "shape": (self.cfg.action.size,),
                # Use the flat list of names
                "names": action_names,
            },
        }

        # Validate schema
        self._assert_schema(features)

        # images
        for name, stream in self.cfg.images.items():
            features[stream.key] = {
                "dtype": "video" if self.cfg.use_videos else "image",
                "shape": list(stream.shape),  # HWC
                "names": ["height", "width", "channels"],
                "info": (
                    {
                        "video.fps": float(self.cfg.fps),
                        "video.codec": self.cfg.video.codec,
                        "video.pix_fmt": self.cfg.video.pix_fmt,
                        "video.is_depth_map": bool(self.cfg.video.is_depth_map),
                        "has_audio": bool(self.cfg.video.has_audio),
                        "image_writer_processes": self.cfg.video.writer_processes,
                        "image_writer_threads": self.cfg.video.writer_threads,
                    }
                    if self.cfg.use_videos
                    else None
                ),
            }

        ds = LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            features=features,
            root=self.cfg.root,
            robot_type=self.cfg.robot_type,
            fps=self.cfg.fps,
            use_videos=self.cfg.use_videos,
            image_writer_processes=self.cfg.video.writer_processes,
            image_writer_threads=self.cfg.video.writer_threads,
        )
        self.log.info(
            "Created dataset at %s (fps=%s, videos=%s)",
            self.cfg.root,
            self.cfg.fps,
            self.cfg.use_videos,
        )
        return ds

    def _resolve_topic_type(
        self,
        reader: Rosbag2Reader,
        topic_name: str,
        configured_type: Optional[str],
    ) -> str:
        """Determine the ROS message type for a topic from config or metadata."""

        if configured_type:
            return configured_type

        metadata = reader.metadata
        rosbag2_bagfile_information = metadata.get("rosbag2_bagfile_information", {}) or {}
        topics = rosbag2_bagfile_information.get("topics_with_message_count", []) or []
        for entry in topics:
            topic_meta = entry.get("topic_metadata", {}) or {}
            if topic_meta.get("name") == topic_name:
                msg_type = topic_meta.get("type")
                if msg_type:
                    return msg_type

        raise ValueError(
            f"Unable to determine message type for topic '{topic_name}'. "
            "Set 'type' (or 'msg_type') in the config for this topic."
        )

    def _vector_from_msg(self, msg, msg_type: str, spec: VectorSpec):
        """Convert a ROS message into a numeric vector based on its type."""

        if msg_type == JS_T:
            return ros_jointstate_to_vec6(
                msg,
                joint_order=self.cfg.joint_order,
                use_lerobot_ranges_norms=self.cfg.use_lerobot_ranges_norms,
            )
        if msg_type == F64_T:
            return ros_float64multiarray_to_vec6(
                msg,
                size=spec.size,
                use_lerobot_ranges_norms=self.cfg.use_lerobot_ranges_norms,
            )

        raise ValueError(
            f"Unsupported message type '{msg_type}' for vector '{spec.key}'. "
            "Expected JointState or Float64MultiArray."
        )

    def _buffers_from_bag(self, bag_path: Path) -> FrameBuffers:
        """Load all relevant messages from a rosbag into in-memory buffers."""

        self.log.info("Reading bag: %s", bag_path.name)
        reader = Rosbag2Reader(bag_path, self.cfg.force, logger=self.log)

        # Skip already processed bags unless forced
        if reader.processed and not self.cfg.force:
            self.log.warning(f"Skipping already processed bag: {bag_path}")
            return FrameBuffers(images={}, state=TopicBuffer(), action=TopicBuffer())

        state_type = None
        action_type = None

        try:
            state_type = self._resolve_topic_type(
                reader, self.cfg.topics.state, self.cfg.state.msg_type
            )
        except ValueError as exc:
            self.log.error(str(exc))
            raise

        try:
            action_type = self._resolve_topic_type(
                reader, self.cfg.topics.action, self.cfg.action.msg_type
            )
        except ValueError as exc:
            self.log.error(str(exc))
            raise

        bufs = FrameBuffers(
            images={name: TopicBuffer() for name in self.cfg.images},
            state=TopicBuffer(),
            action=TopicBuffer(),
            state_type=state_type,
            action_type=action_type,
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

            # state and action topics
            if msg.topic == self.cfg.topics.state:
                des = reader.deserialize(msg.raw, state_type)
                bufs.state.add(msg.t_sec, des)
            elif msg.topic == self.cfg.topics.action:
                des = reader.deserialize(msg.raw, action_type)
                bufs.action.add(msg.t_sec, des)

        for b in bufs.images.values():
            b.finalize()
        bufs.state.finalize()
        bufs.action.finalize()

        reader.close()

        return bufs

    def _parse_sync_reference(self) -> Tuple[Literal["image", "state", "action"], Optional[str]]:
        """Parse the synchronization reference from the configuration."""

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
        """Emit a human readable summary of synchronization quality per topic."""

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
        """Yield synchronized frames combining images, state, and actions."""

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
        """Convert all rosbag files under ``cfg.bags_root`` into a dataset."""
        # Note: Directory creation is now handled in cli.py via get_versioned_dataset_dir

        try:
            # This checks for valid repo_id format (e.g., "user/repo")
            sanity_check_dataset_name(self.cfg.repo_id, policy_cfg=None)
            self.log.info(f"Dataset repo_id '{self.cfg.repo_id}' passed sanity check.")
        except ValueError as e:
            self.log.error(f"Invalid dataset repo_id: {e}")
            # Stop conversion if the name is invalid
            raise

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

            state_type = bufs.state_type or self.cfg.state.msg_type
            action_type = bufs.action_type or self.cfg.action.msg_type

            if not state_type or not action_type:
                raise ValueError(
                    "State or action message type could not be resolved for bag " f"{bag.name}."
                )

            for m in self._iter_synced(bufs):
                try:
                    # Convert ROS messages → numpy arrays
                    # inside the frame loop
                    st6 = np.asarray(
                        self._vector_from_msg(m["state"], state_type, self.cfg.state),
                        dtype=np.float32,
                    ).ravel()
                    ac6 = np.asarray(
                        self._vector_from_msg(m["action"], action_type, self.cfg.action),
                        dtype=np.float32,
                    ).ravel()

                    # Build the frame dict expected by LeRobotDataset
                    frame = {
                        self.cfg.state.key: st6.astype("float32"),
                        self.cfg.action.key: ac6.astype("float32"),
                        "task": self.cfg.task_text,  # required
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
                    self.log.error(
                        f"Error processing frame {episode_frames} in bag {bag.name}: {e}"
                    )
                    continue

            # After finishing one bag (episode)
            if episode_frames > 0:
                ds.save_episode()

            self.log.info(
                "Bag %s → frames saved: %d / ref=%d | dropped=%d",
                bag.name,
                episode_frames,
                ref_len,
                dropped,
            )

        # Finalize dataset, must be called after all episodes are added for closing parquet writers
        ds.finalize()

        # Optionally upload to Hugging Face Hub
        if self.cfg.upload_to_hub:
            ds.push_to_hub(repo_id=self.cfg.repo_id)

        self.log.info("✅ Completed conversion. Total frames written: %d", total_frames)
        return total_frames

    @staticmethod
    def _discover_bags(root: Path):
        """Yield bag files found recursively under ``root``."""

        from .io_bag import discover_bags

        yield from discover_bags(root)
