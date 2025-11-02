from pathlib import Path
from typing import List, Optional, Tuple
import math
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


def radians_to_normalized(joint_name: str, rad: float) -> float:
    """
    converts a command in radians from MoveIt to the format expected by the SO101 API.
    """
    if joint_name == "gripper":
        # Convert radian command [0, pi] to the robot's expected gripper range [10, ~110]
        normalized = (rad / math.pi) * 100.0 + 10.0
    else:
        # Convert radians to normalized range [-100, 100]
        normalized = (rad / math.pi) * 100.0
    return normalized


def ros_jointstate_to_vec6(js_msg, joint_order: Optional[List[str]] = None, use_lerobot_ranges_norms: bool = False) -> np.ndarray:
    pos = list(getattr(js_msg, "position", []))
    names = list(getattr(js_msg, "name", []))
    out = np.zeros((6,), dtype=np.float32)

    if joint_order:
        if len(joint_order) != 6:
            raise ValueError(f"joint_order must have 6 names, got {len(joint_order)}")
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            vals = [pos[name_to_idx[n]] for n in joint_order]
            if use_lerobot_ranges_norms:
                vals = [radians_to_normalized(joint_name, val) 
                    for joint_name, val in zip(joint_order, vals)]
        except KeyError as e:
            missing = set(joint_order) - set(names)
            raise KeyError(f"Joint(s) {missing} not found in JointState.name") from e
        out[:] = np.array(vals, dtype=np.float32)
    else:
        if len(pos) < 6:
            raise ValueError(f"JointState.position has {len(pos)} values, need >= 6")
        vals = pos[:6]
        if use_lerobot_ranges_norms:
            # When no joint_order is provided, use the joint names from the message
            joint_names = names[:6] if len(names) >= 6 else [f"joint_{i}" for i in range(6)]
            vals = [radians_to_normalized(joint_name, val) 
                for joint_name, val in zip(joint_names, vals)]
        out[:] = np.array(vals, dtype=np.float32)

    return out


def ros_float64multiarray_to_vec6(arr_msg, size: int = 6, use_lerobot_ranges_norms: bool = False) -> np.ndarray:
    """
    Convert std_msgs/Float64MultiArray -> (size,) float32 vector.
    Takes the first `size` elements from msg.data.
    """
    data = list(getattr(arr_msg, "data", []))
    if len(data) < size:
        raise ValueError(f"Float64MultiArray.data has {len(data)} values, need >= {size}")
    
    vals = data[:size]
    if use_lerobot_ranges_norms:
        # Apply radians_to_normalized conversion
        # Last element is treated as gripper, others as regular joints
        normalized_vals = []
        for i, val in enumerate(vals):
            if i == size - 1:  # Last element is gripper
                joint_name = "gripper"
            else:
                joint_name = f"joint_{i}" # It does not matter what name we use here
            normalized_vals.append(radians_to_normalized(joint_name, val))
        vals = normalized_vals
    
    return np.asarray(vals, dtype=np.float32)


def get_versioned_pathes(out_dir: str, data_dir_name: str) -> Tuple[str, str]:
    """
    Get the versioned dataset directory name without creating the data directory.
    The data directory will be created by LeRobotDataset.create().
    Only creates the logs directory.
    
    Returns: (versioned_dataset_name, data_path)
    
    Example:
        out_dir = "/tmp/output"
        data_dir_name = "my_dataset"
        
        Returns: ("my_dataset_1", "/tmp/output/data/my_dataset_1")
        
        Creates:
        - /tmp/output/logs/my_dataset_1/  (for logs)
        
        But does NOT create:
        - /tmp/output/data/my_dataset_1/  (LeRobotDataset.create() will do this)
    """
    version = 1
    
    # Find next available version by checking both data and logs directories
    while True:
        if version == 1:
            versioned_name = data_dir_name
        else:
            versioned_name = f"{data_dir_name}_{version}"
        
        data_path = Path(out_dir) / "data" / versioned_name
        logs_path = Path(out_dir) / "logs" / versioned_name
        
        # If neither exists, we found our version
        if not data_path.exists() and not logs_path.exists():
            break
            
        version += 1

    return str(logs_path), str(data_path)