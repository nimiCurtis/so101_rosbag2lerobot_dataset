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


import argparse
import logging
from pathlib import Path

from so101_rosbag2lerobot_dataset.config import Config
from so101_rosbag2lerobot_dataset.converter import RosbagToLeRobotConverter


def _setup_logger(log_path: str) -> logging.Logger:
    """Create a logger that writes both to stdout and ``log_path``."""

    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("rosbag2lerobot")
    log.setLevel(logging.INFO)

    format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # file handler
    fh = logging.FileHandler(log_dir / "convert.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(format)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(format)

    # avoid duplicate handlers if CLI is called twice in same process
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


def main():
    """Entry point for the ``so101-rosbag2lerobot`` command line interface."""

    parser = argparse.ArgumentParser(
        description="Convert ROS 2 bag(s) to LeRobot Dataset v3 (videos/images + state + actions)."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    log = _setup_logger(cfg.logs_dir)

    log.info("=== so101_rosbag2lerobot_dataset ===")
    log.info("Data dir: %s", cfg.root)
    log.info("Logs dir: %s", cfg.logs_dir)
    log.info("Versioned dataset: %s", cfg.repo_id)
    log.info("Bags root: %s", cfg.bags_root)
    log.info("Repo id: %s | FPS: %s | Videos: %s", cfg.repo_id, cfg.fps, cfg.use_videos)
    log.info("Image streams: %s", ", ".join(f"{n}:{s.topic}" for n, s in cfg.images.items()))
    log.info(
        "Sync ref: %s | Tol: %.1f ms | Downsample: %d",
        cfg.sync_reference,
        cfg.sync_tolerance_s * 1000.0,
        cfg.downsample_by,
    )

    conv = RosbagToLeRobotConverter(cfg, logger=log)
    total = conv.convert()
    log.info("Done. Total frames written: %d", total)


if __name__ == "__main__":
    main()
