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


import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import yaml

_SPLIT_RE = re.compile(r"^(?P<prefix>.+?)_(?P<idx>\d+)\.(?P<ext>db3|bag)$", re.IGNORECASE)


@dataclass
class BagMessage:
    """A deserialized rosbag record accompanied by its timestamp."""

    topic: str
    t_sec: float
    raw: bytes  # serialized ROS message bytes


class Rosbag2Reader:
    """Thin wrapper around ``rosbag2_py`` for iterating over bag messages."""

    def __init__(self, bag_path: Path, force: bool, logger):
        """Create a reader for a single rosbag directory."""

        self.bag_path = bag_path
        self.force = force
        self.log = logger
        self.rosbag2_py = importlib.import_module("rosbag2_py")
        self.deserialize_message = importlib.import_module(
            "rclpy.serialization"
        ).deserialize_message
        self.get_message = importlib.import_module("rosidl_runtime_py.utilities").get_message

        # Check if the bag is already processed
        self._processed = self.metadata.get("processed", False)
        if self._processed and not self.force:
            self.log.warning(f"Rosbag at {self.bag_path} is already processed.")
        elif self._processed and self.force:
            self.log.info(f"Force re-processing rosbag at {self.bag_path}.")
            self._processed = False

    @property
    def metadata(self):
        """Return the raw metadata dictionary for the bag."""

        return self._metadata()

    @property
    def processed(self):
        """Indicate whether the bag was already marked as processed."""

        return self._processed

    def _metadata(self):
        """Load the ``metadata.yaml`` next to the bag file."""

        metadata_path = self.bag_path.parent / "metadata.yaml"
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        return metadata

    def iter_messages(self) -> Iterator[BagMessage]:
        """Yield messages from the rosbag in chronological order."""

        storage_options = self.rosbag2_py.StorageOptions(
            uri=str(self.bag_path), storage_id="sqlite3"
        )
        converter_options = self.rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader = self.rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        while reader.has_next():
            topic, raw, t_ns = reader.read_next()
            yield BagMessage(topic=topic, t_sec=t_ns * 1e-9, raw=raw)

    def deserialize(self, raw: bytes, type_str: str):
        """Deserialize raw ROS bytes into a Python message instance."""

        msg_type = self.get_message(type_str)
        return self.deserialize_message(raw, msg_type)

    def close(self):
        """Mark the rosbag as processed in its metadata file."""

        metadata_path = self.bag_path.parent / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            metadata["processed"] = True
            with open(metadata_path, "w") as f:
                yaml.safe_dump(metadata, f)


def _series_sort_key(p: Path) -> Tuple[str, str, int]:
    """
    Sort key that groups by parent dir and series prefix, then numeric split index.
    For unsuffixed files (no _<num>), idx = -1 so they appear before any numbered parts
    of the same prefix (rare for rosbag2, but safe).
    """
    fname = p.name
    m = _SPR = _SPLIT_RE.match(fname)
    if m:
        prefix = m.group("prefix")
        idx = int(m.group("idx"))
    else:
        # No numeric suffix; treat the whole stem as prefix and idx = -1
        prefix = p.stem
        idx = -1
    # Group by parent path string for stable cross-platform ordering
    return (str(p.parent), prefix, idx)


def discover_bags(root: Path) -> Iterator[Path]:
    """Yield rosbag database files under `root`, sorted so ..._0, _1, _2, ... per series."""
    candidates: List[Path] = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".db3", ".bag")):
                candidates.append(Path(dirpath) / f)

    # Sort by (parent_dir, series_prefix, split_idx)
    candidates.sort(key=_series_sort_key)

    # Yield in the sorted order
    for p in candidates:
        yield p
