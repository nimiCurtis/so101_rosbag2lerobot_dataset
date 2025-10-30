from pathlib import Path
from typing import List, Optional
import numpy as np


def ros_image_to_hwc_float01(msg) -> np.ndarray:
    enc = msg.encoding
    if enc not in ("rgb8", "bgr8", "mono8"):
        raise ValueError(f"Unsupported encoding: {enc}")
    ch = 3 if enc in ("rgb8", "bgr8") else 1
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
    if enc == "bgr8":
        arr = arr[..., ::-1]
    if ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    return (arr.astype(np.float32) / 255.0)


def ros_jointstate_to_vec6(js_msg, joint_order: Optional[List[str]] = None) -> np.ndarray:
    pos = list(getattr(js_msg, "position", []))
    names = list(getattr(js_msg, "name", []))
    out = np.zeros((6,), dtype=np.float32)

    if joint_order:
        if len(joint_order) != 6:
            raise ValueError(f"joint_order must have 6 names, got {len(joint_order)}")
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            vals = [pos[name_to_idx[n]] for n in joint_order]
        except KeyError as e:
            missing = set(joint_order) - set(names)
            raise KeyError(f"Joint(s) {missing} not found in JointState.name") from e
        out[:] = np.array(vals, dtype=np.float32)
    else:
        if len(pos) < 6:
            raise ValueError(f"JointState.position has {len(pos)} values, need >= 6")
        out[:] = np.array(pos[:6], dtype=np.float32)

    return out


def ros_float64multiarray_to_vec6(arr_msg, size: int = 6) -> np.ndarray:
    """
    Convert std_msgs/Float64MultiArray -> (size,) float32 vector.
    Takes the first `size` elements from msg.data.
    """
    data = list(getattr(arr_msg, "data", []))
    if len(data) < size:
        raise ValueError(f"Float64MultiArray.data has {len(data)} values, need >= {size}")
    return np.asarray(data[:size], dtype=np.float32)


def build_dataset_dir(out_dir: str, data_dir_name: str) -> str:
    """
    Build the dataset directory path from out_dir and data_dir_name.
    """
    # Root for LeRobotDataset should be construct from out_dir + "self.cfg.data_dir_name" + "_{version_number}"
    # version number should be incremented if the directory already exists
    
    base_path = Path(out_dir) / data_dir_name
    version = 1
    dataset_path = base_path
    while dataset_path.exists():
        version += 1
        dataset_path = Path(f"{base_path}_{version}")   

    return str(dataset_path)
