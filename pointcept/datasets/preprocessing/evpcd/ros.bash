#!/bin/bash

# デフォルト値の設定
DEFAULT_PATH_BAG="$HOME/Pointcept/data/evpcd/rosbag/"
DEFAULT_TOPIC_CLOUD="/iv_points"
DEFAULT_OUTPUT_PATH="$HOME/Pointcept/data/evpcd/bin/"

# 引数の受け取り（なければデフォルト）
PATH_BAG="${1:-$DEFAULT_PATH_BAG}"
TOPIC_CLOUD="${2:-$DEFAULT_TOPIC_CLOUD}"
OUTPUT_PATH="${3:-$DEFAULT_OUTPUT_PATH}"

echo "📂 ROSBAG:       ${PATH_BAG}"
echo "📡 TOPIC:        ${TOPIC_CLOUD}"
echo "📦 OUTPUT BIN:   ${OUTPUT_PATH}"

# ROS 2環境の読み込み
source ~/Pointcept/pointcept/datasets/preprocessing/evpcd/ros/install/setup.bash

# 点群変換ノードの実行
ros2 run rosbag2_to_bin rosbag2_to_bin_node \
  --ros-args \
  -p path_bag:="${PATH_BAG}" \
  -p topic_cloud:="${TOPIC_CLOUD}" \
  -p output_path:="${OUTPUT_PATH}"

# .pkl の自動生成
python3 ~/Pointcept/pointcept/datasets/preprocessing/evpcd/preprocess_evpcd.py "${OUTPUT_PATH}"
