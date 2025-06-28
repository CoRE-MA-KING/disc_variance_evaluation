import pyrealsense2 as rs
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DiscTracker:
    def __init__(self):
        # ディスクごとのデータを辞書で保存: {ディスクID: [(x, y), (x, y), ...]}
        self.disc_data = {}

    def add_measurement(self, disc_id, x, y):
        # 測定データを追加
        if disc_id not in self.disc_data:
            self.disc_data[disc_id] = []
        self.disc_data[disc_id].append((x, y))

    def get_average(self, disc_id):
        # 平均座標を返す
        if disc_id not in self.disc_data or not self.disc_data[disc_id]:
            return None  # データがない場合
        x_vals = [p[0] for p in self.disc_data[disc_id]]
        y_vals = [p[1] for p in self.disc_data[disc_id]]
        avg_x = sum(x_vals) / len(x_vals)
        avg_y = sum(y_vals) / len(y_vals)
        return (avg_x, avg_y)

    def get_all_averages(self):
        # 全ディスクの平均をまとめて返す
        return {
            disc_id: self.get_average(disc_id)
            for disc_id in self.disc_data
        }

    def __str__(self):
        return f"DiscTracker({self.disc_data})"

def plot_disc_averages(tracker):
    averages = tracker.get_all_averages()
    if not averages:
        print("平均値がありません")
        return

    x = [avg[0] for avg in averages.values()]
    y = [avg[1] for avg in averages.values()]
    labels = [str(disc_id) for disc_id in averages.keys()]

    plt.scatter(x, y)

    for i, label in enumerate(labels):
        plt.text(x[i], y[i], f"ID {label}", fontsize=9, ha='right')
    xmin=-1000
    xmax=1000
    zmin=0
    zmax=2000
    plt.xlim(xmin, xmax)  # 横軸の範囲
    plt.ylim(zmin, zmax)  # 縦軸の範囲
    plt.title("Disc Averages")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# RealSenseパイプライン初期化
pipeline = rs.pipeline()
config = rs.config()

# 深度ストリーム有効化（640x480, 30FPS）
config.enable_stream(rs.stream.depth, 848, 100, rs.format.z16, 100)

# ストリーミング開始
pipeline.start(config)

profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 初期フレーム（背景）保存用
bg_depth = None

disc_flag = 0  # 今回のフレームでディスクが検出されたか
pre_disc_flag = 0  # 前回のフレームでディスクが検出されていたか
disc_count = 0

tracker = DiscTracker()

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # numpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data())

        # 初期フレーム保存（背景）
        if bg_depth is None:
            bg_depth = depth_image
            continue

        # 差分取得（深度差）
        diff = cv2.absdiff(depth_image, bg_depth)

        # 閾値処理（動いた部分を白く）
        _, fg_mask = cv2.threshold(diff, 1000, 255, cv2.THRESH_BINARY)

        # 8bit変換（輪郭検出のため）
        fg_mask = np.uint8(fg_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 輪郭検出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        disc_flag = 0  # 毎フレーム初期化

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3000 < area < 60000:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                
                # 通過検知フラグを立てる
                disc_flag = 1

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                depth = depth_frame.get_distance(cx, cy)
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                if point[2]<0.0001:
                    break
                #print(f"検出位置: 2D=({cx}, {cy}), 深度={int(depth*1000):4d}mm, 3D=({(point[0]*1000):.2f},{(point[1]*1000):.2f},{(point[2]*1000):.2f})")
                print(f"検出位置: ({(point[0]*1000):6.1f},{(point[1]*1000):6.1f},{(point[2]*1000):6.1f}), disc_count{disc_count}")
                tracker.add_measurement(disc_count, point[0]*1000, point[2]*1000)

                # デバッグ表示用画像作成（カラーなし）
                show = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.circle(show, (cx, cy), 10, (0, 255, 0), -1)
                cv2.imshow("Depth with Detection", show)


        if disc_flag == 1 and pre_disc_flag == 0:
            disc_count += 1

        # 現在の状態を記録しておく
        pre_disc_flag = disc_flag        


        cv2.imshow("Depth Foreground Mask", fg_mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            plot_disc_averages(tracker)


finally:
    pipeline.stop()
    cv2.destroyAllWindows()