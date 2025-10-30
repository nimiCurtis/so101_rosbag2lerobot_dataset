from dataclasses import dataclass
from typing import Iterator
import importlib
import os
from pathlib import Path


@dataclass
class BagMessage:
    topic: str
    t_sec: float
    raw: bytes  # serialized ROS message bytes


class Rosbag2Reader:
    def __init__(self, bag_path: Path):
        self.bag_path = bag_path
        self.rosbag2_py = importlib.import_module("rosbag2_py")
        self.deserialize_message = importlib.import_module("rclpy.serialization").deserialize_message
        self.get_message = importlib.import_module("rosidl_runtime_py.utilities").get_message

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
        return self.deserialize_message(raw, msg_type)


def discover_bags(root: Path):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".db3") or f.endswith(".bag"):
                yield Path(dirpath) / f
