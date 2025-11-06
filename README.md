# so101_rosbag2lerobot_dataset

## Overview
so101_rosbag2lerobot_dataset converts ROS 2 bag recordings into [LeRobot](https://github.com/huggingface/lerobot) datasets. It wraps the ingestion of image, state, and action streams into a reproducible pipeline with synchronization diagnostics and versioned output directories, making it straightforward to publish high-quality robot demonstrations.

## Dependencies

- ROS2 Humble from the [Official Link](https://docs.ros.org/en/humble/Installation.html)
- [LeRobot](https://github.com/huggingface/lerobot)
- Hugging Face hub [user access token](https://huggingface.co/docs/hub/en/security-tokens)

## Installation
```bash
cd so101-rosbag2lerobot-dataset
python -m pip install so101-rosbag2lerobot-dataset
```

For local development:
```bash
cd so101-rosbag2lerobot-dataset
pip install -e .[dev]
pre-commit install
```

## Conversion
1. Create a YAML configuration describing the bag location, ROS topics, and dataset layout (see `config/` for examples).
2. Run the CLI with the configuration:
   ```bash
   so101-rosbag2lerobot --config path/to/config.yaml
   ```
3. Conversion produces versioned `data/` and `logs/` sub-directories inside `./output`. The logs contain synchronization reports for each bag.

## Visualization

After converting your ROS bags to LeRobot format, you can visualize and explore your dataset using the LeRobot CLI tools.

Visualize your converted dataset with:

```bash
lerobot-dataset-viz --repo-id REPO_ID \
 --episode-index INDEX \
 --root ROOT \
 --mode local
```

**[NOTE] For now this project support only local dataset visualization and not from the hub.**


## Contribute

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Install development dependencies with `pip install -e .[dev]`.
3. Run `pre-commit run --all-files` and `pytest` before submitting a pull request.

## Acknowledgment

This project builds upon the excellent work of the [LeRobot](https://github.com/huggingface/lerobot) team and the ROS 2 community for providing robust bagging and serialization utilities.

## License

This repository is released under the terms of the MIT License. See [LICENSE](LICENSE) for details.

## V1.0.1 ROADMAP
