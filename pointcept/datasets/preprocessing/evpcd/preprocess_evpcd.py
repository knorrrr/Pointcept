import os
import pickle
import sys

def generate_pkl(bin_dir):
    if not os.path.exists(bin_dir):
        print(f"❌ Error: '{bin_dir}' does not exist。")
        return

    data_list = []

    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    if len(bin_files) < 2:
        print("❌ Error: .binファイルが2つ以上必要です。")
        return

    usable_len = len(bin_files) - (len(bin_files) % 2)  # 奇数なら1つ減らす

    for i in range(0, usable_len, 2):
        lidar_fname = bin_files[i]
        pred_fname = bin_files[i + 1]
        lidar_token = os.path.splitext(lidar_fname)[0]

        lidar_path = os.path.join(os.path.abspath(bin_dir), lidar_fname)
        pred_lidar_path = os.path.join(os.path.abspath(bin_dir), pred_fname)

        data_list.append({
            "lidar_path": lidar_path,
            "pred_lidar_path": pred_lidar_path,
            "lidar_token": lidar_token
        })

    if len(bin_files) % 2 == 1:
        print(f"⚠️ binファイル数が奇数なので、最後の1つ（{bin_files[-1]}）は無視されました。")

    parent_dir = os.path.dirname(bin_dir.rstrip("/"))
    info_dir = os.path.join(parent_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

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
