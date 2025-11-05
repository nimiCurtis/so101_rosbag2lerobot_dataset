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


import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def ros_image_to_hwc_float01(msg) -> np.ndarray:
    """Convert a ROS image message to an HWC float array scaled to ``[0, 1]``.

    Args:
        msg: A ROS ``sensor_msgs/Image`` like object with ``encoding``, ``height``,
            ``width`` and ``data`` attributes.

    Returns:
        ``numpy.ndarray`` with shape ``(height, width, 3)`` containing float32 values in
        the range ``[0, 1]``.

    Raises:
        ValueError: If the encoding is not one of ``rgb8``, ``bgr8`` or ``mono8``.
    """

    enc = msg.encoding
    if enc not in ("rgb8", "bgr8", "mono8"):
        raise ValueError(f"Unsupported encoding: {enc}")
    ch = 3 if enc in ("rgb8", "bgr8") else 1
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)
    if enc == "bgr8":
        arr = arr[..., ::-1]
    if ch == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.float32) / 255.0


def radians_to_normalized(joint_name: str, rad: float) -> float:
    """Convert a radian command into the normalized SO101 joint range.

    Args:
        joint_name: Name of the joint for which the command is expressed. The gripper has
            a special conversion.
        rad: Command expressed in radians as provided by MoveIt.

    Returns:
        The normalized joint value expected by the SO101 API.
    """
    if joint_name == "gripper":
        # Convert radian command [0, pi] to the robot's expected gripper range [10, ~110]
        normalized = (rad / math.pi) * 100.0 + 10.0
    else:
        # Convert radians to normalized range [-100, 100]
        normalized = (rad / math.pi) * 100.0
    return normalized


def ros_jointstate_to_vec6(
    js_msg,
    joint_order: Optional[List[str]] = None,
    use_lerobot_ranges_norms: bool = False,
) -> np.ndarray:
    """Convert a ``sensor_msgs/JointState`` message to a six element vector.

    Args:
        js_msg: Message providing ``position`` and optionally ``name`` attributes.
        joint_order: Optional explicit joint ordering for the returned vector.
        use_lerobot_ranges_norms: Whether to map the values to LeRobot's normalized
            ranges using :func:`radians_to_normalized`.

    Returns:
        ``numpy.ndarray`` shaped ``(6,)`` containing the joint positions.

    Raises:
        ValueError: If the provided message does not contain enough joint positions or
            the ``joint_order`` does not contain exactly six joints.
        KeyError: If a name specified in ``joint_order`` is missing in the message.
    """

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
                vals = [
                    radians_to_normalized(joint_name, val)
                    for joint_name, val in zip(joint_order, vals)
                ]
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
            vals = [
                radians_to_normalized(joint_name, val) for joint_name, val in zip(joint_names, vals)
            ]
        out[:] = np.array(vals, dtype=np.float32)

    return out


def ros_float64multiarray_to_vec6(
    arr_msg,
    size: int = 6,
    use_lerobot_ranges_norms: bool = False,
) -> np.ndarray:
    """Convert a ROS ``Float64MultiArray`` to a fixed-size float vector.

    Args:
        arr_msg: Message providing the ``data`` attribute.
        size: Number of elements to extract from the message.
        use_lerobot_ranges_norms: Whether to normalize the commands using the
            :func:`radians_to_normalized` helper.

    Returns:
        ``numpy.ndarray`` with ``size`` float32 elements.

    Raises:
        ValueError: If the provided message does not contain enough elements.
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
                joint_name = f"joint_{i}"  # It does not matter what name we use here
            normalized_vals.append(radians_to_normalized(joint_name, val))
        vals = normalized_vals

    return np.asarray(vals, dtype=np.float32)


def get_versioned_paths(out_dir: str, data_dir_name: str) -> Tuple[str, str]:
    """Return versioned log and data paths without creating the dataset directory.

    The helper inspects both the ``data`` and ``logs`` directories to find the next
    available suffix. Only the logs directory is created by the caller when needed – the
    data directory is intentionally left for :class:`LeRobotDataset` to create.

    Args:
        out_dir: Root output directory containing ``data`` and ``logs`` sub-folders.
        data_dir_name: Base name used for versioned directories.

    Returns:
        A tuple ``(logs_path, data_path)`` pointing to the computed directories.
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
