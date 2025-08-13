#sign_to_text/dataset/make_data.py
import cv2
import os

def capture_sign_images():
    label = input("Nhập ký hiệu (vd: A, B, C, hello): ").strip()
    data_split = input("Chọn tập dữ liệu (train/val): ").strip().lower()
    assert data_split in ['train', 'val'], "Chỉ được nhập 'train' hoặc 'val'"

    warmup_frames = 60
    max_images = 500
    resize_scale = 0.3

    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(root_dir, data_split, "char", label)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    frame_count = 0
    saved_count = 0

    print(f"Chuẩn bị chụp ký hiệu '{label}' cho tập {data_split}.")
    print("Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

        frame_count += 1
        cv2.imshow('Capture Sign Language', frame)

        if frame_count >= warmup_frames and saved_count < max_images:
            saved_count += 1
            img_path = os.path.join(save_dir, f"{label}_{saved_count:04d}.png")
            cv2.imwrite(img_path, frame)
            print(f"Đã lưu: {img_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Người dùng yêu cầu thoát.")
            break

        if saved_count >= max_images:
            print("Đã chụp đủ ảnh. Dừng chương trình.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_sign_images()
