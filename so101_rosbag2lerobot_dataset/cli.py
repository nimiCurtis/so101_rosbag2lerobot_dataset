import argparse
import logging
from pathlib import Path

from so101_rosbag2lerobot_dataset.utils import build_dataset_dir
from so101_rosbag2lerobot_dataset.config import Config
from so101_rosbag2lerobot_dataset.converter import RosbagToLeRobotConverter


def _setup_logger(out_dir: str, data_dir_name: str) -> logging.Logger:
    
    log_dir = Path(out_dir) / "logs" / data_dir_name
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
    parser = argparse.ArgumentParser(
        description="Convert ROS 2 bag(s) to LeRobot Dataset v3 (videos/images + state + actions)."
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    log = _setup_logger(cfg.out_dir, cfg.data_dir_name)

    log.info("=== so101_rosbag2lerobot_dataset ===")
    log.info("Output dir: %s", cfg.out_dir)
    log.info("Bags root: %s", cfg.bags_root)
    log.info("Repo id: %s | FPS: %s | Videos: %s", cfg.repo_id, cfg.fps, cfg.use_videos)
    log.info("Image streams: %s", ", ".join(f"{n}:{s.topic}" for n, s in cfg.images.items()))
    log.info("Sync ref: %s | Tol: %.1f ms | Downsample: %d",
             cfg.sync_reference, cfg.sync_tolerance_s * 1000.0, cfg.downsample_by)

    conv = RosbagToLeRobotConverter(cfg, logger=log)
    total = conv.convert()
    log.info("Done. Total frames written: %d", total)

if __name__ == "__main__":
    main()