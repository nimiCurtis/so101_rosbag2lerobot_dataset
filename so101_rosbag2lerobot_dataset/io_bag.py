from dataclasses import dataclass
from typing import Iterator
import importlib
import os
from pathlib import Path

import yaml
# from rosbag2_py import StorageOptions, ConverterOptions, SequentialReader
# from rclpy.serialization import deserialize_message
# from rosidl_runtime_py.utilities import get_message
@dataclass
class BagMessage:
    topic: str
    t_sec: float
    raw: bytes  # serialized ROS message bytes


class Rosbag2Reader:
    def __init__(self, bag_path: Path, force: bool, logger):
        self.bag_path = bag_path
        self.force = force
        self.log = logger
        self.rosbag2_py = importlib.import_module("rosbag2_py")
        self.deserialize_message = importlib.import_module("rclpy.serialization").deserialize_message
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
        return self._metadata()

    @property
    def processed(self):
        return self._processed
    
    def _metadata(self):
        metadata_path = self.bag_path.parent / "metadata.yaml"
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        return metadata
    
    def iter_messages(self) -> Iterator[BagMessage]:
        storage_options = self.rosbag2_py.StorageOptions(uri=str(self.bag_path), storage_id="sqlite3")
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
        msg_type = self.get_message(type_str)
        return self.deserialize_message(raw, msg_type)\
            
    def close(self):
        # Load the metadata yaml file and set "processed" to True to release resources
        metadata_path = self.bag_path.parent / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            metadata["processed"] = True
            with open(metadata_path, "w") as f:
                yaml.safe_dump(metadata, f)

def discover_bags(root: Path):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".db3") or f.endswith(".bag"):
                yield Path(dirpath) / f
