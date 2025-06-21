import pyrealsense2 as rs
import cv2
import numpy as np

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
        _, fg_mask = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)

        # 8bit変換（輪郭検出のため）
        fg_mask = np.uint8(fg_mask)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # 輪郭検出
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3000 < area < 60000:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                depth = depth_frame.get_distance(cx, cy)
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)
                print(f"検出位置: 2D=({cx}, {cy}), 深度={depth:.3f}m, 3D={point}")

                # デバッグ表示用画像作成（カラーなし）
                show = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                cv2.circle(show, (cx, cy), 10, (0, 255, 0), -1)
                cv2.imshow("Depth with Detection", show)

        cv2.imshow("Depth Foreground Mask", fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()