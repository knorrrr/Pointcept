import os
import pickle
import sys

def generate_pkl(bin_dir):
    if not os.path.exists(bin_dir):
        print(f"❌ Error: '{bin_dir}' does not exist。")
        return

    data_list = []

    for fname in sorted(os.listdir(bin_dir)):
        if not fname.endswith(".bin"):
            continue
        token = os.path.splitext(fname)[0]
        lidar_path = os.path.join(os.path.dirname(os.path.abspath(bin_dir)), "bin", fname)
        # print(lidar_path)
        data_list.append({
            "lidar_path": lidar_path,
            "lidar_token": token
        })

    # 保存先ディレクトリ: bin_dir の親ディレクトリに info フォルダを作成
    parent_dir = os.path.dirname(bin_dir.rstrip("/"))
    info_dir = os.path.join(parent_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    # 保存先ファイル名
    save_path = os.path.join(info_dir, "info.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(data_list, f)

    print(f"✅ Saved {len(data_list)} entries to {save_path}")

# 実行エントリポイント
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pkl_from_bin.py <bin_directory>")
        sys.exit(1)

    bin_dir = sys.argv[1]
    generate_pkl(bin_dir)
