# so101_rosbag2lerobot_dataset

## Overview
so101_rosbag2lerobot_dataset converts ROS 2 bag recordings into [LeRobot](https://github.com/huggingface/lerobot) datasets. It wraps the ingestion of image, state, and action streams into a reproducible pipeline with synchronization diagnostics and versioned output directories, making it straightforward to publish high-quality robot demonstrations.

## Installation
```bash
python -m pip install so101-rosbag2lerobot-dataset
```
The package depends on `rosbag2_py` and related ROS 2 Python packages. Install them through your ROS 2 distribution or apt environment before installing this project.

For local development:
```bash
pip install -e .[dev]
pre-commit install
```

## Conversion
1. Create a YAML configuration describing the bag location, ROS topics, and dataset layout (see `config/` for examples).
2. Run the CLI with the configuration:
   ```bash
   so101-rosbag2lerobot --config path/to/config.yaml
   ```
3. Conversion produces versioned `data/` and `logs/` sub-directories inside `out_dir`. The logs contain synchronization reports for each bag.

## Visualization
Use the included `so101_rosbag2lerobot_dataset.sample` script to explore the resulting parquet data:
```bash
python -m so101_rosbag2lerobot_dataset.sample path/to/lerobot/file.parquet
```
For more advanced visualization, import the dataset with `pandas` or the LeRobot tooling to render videos and evaluate trajectories.

## Contribute
Contributions are welcome! Please:
1. Fork the repository and create a feature branch.
2. Install development dependencies with `pip install -e .[dev]`.
3. Run `pre-commit run --all-files` and `pytest` before submitting a pull request.

## Acknowledgment
This project builds upon the excellent work of the [LeRobot](https://github.com/huggingface/lerobot) team and the ROS 2 community for providing robust bagging and serialization utilities.

## License
This repository is released under the terms of the MIT License. See [LICENSE](LICENSE) for details.
