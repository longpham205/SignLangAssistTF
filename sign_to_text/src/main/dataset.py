#sign_to_text/src/main/dataset.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)
import glob
import tensorflow as tf

import config as cf

# ===============================
# Helper cho ảnh tĩnh (static images)
# ===============================
def preprocess_image(path, label, img_size=cf.IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scale [0,1]
    img = tf.image.resize(img, img_size)
    # Chuẩn hóa theo ImageNet mean/std
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img, label


def load_static_dataset(root_dir, split='train', batch_size=cf.BATCH_SIZE_STT_STATIC, shuffle=True):
    data_dir = os.path.join(root_dir, split,"char")
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    print("classes",classes)
    print("class_to_idx",class_to_idx)

    all_files = []
    all_labels = []

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        all_files.extend(files)
        all_labels.extend([class_to_idx[cls]] * len(files))

    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    
    print("dataset",dataset)
    
    for f, l in dataset.take(1):
        print("f ,type",f.numpy(), type(f.numpy()))
    
    dataset = dataset.map(lambda f, l: preprocess_image(tf.cast(f, tf.string), tf.cast(l, tf.int32)), num_parallel_calls=tf.data.AUTOTUNE)
    
    print("dataset after map",dataset)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_to_idx



# ===============================
# Helper cho video dataset
# ===============================
def decode_video(path, seq_len=cf.SEQ_LEN, img_size=cf.IMG_SIZE):
    # TensorFlow đọc video không native tốt như OpenCV
    # Ở đây ta sẽ fallback gọi OpenCV trong tf.py_function
    def _read_video(video_path):
        import cv2
        cap = cv2.VideoCapture(video_path.decode('utf-8'))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        # Uniform sample/pad seq_len
        T = len(frames)
        if T == 0:
            blank = tf.zeros((img_size[0], img_size[1], 3), dtype=tf.uint8)
            frames = [blank.numpy()] * seq_len
        idxs = tf.linspace(0.0, tf.cast(max(T-1,0), tf.float32), seq_len)
        idxs = tf.cast(idxs, tf.int32).numpy()

        sampled = []
        for i in idxs:
            f = frames[i]
            sampled.append(f)

        return tf.stack(sampled)

    video_tensor = tf.py_function(_read_video, [path], tf.uint8)
    video_tensor.set_shape((seq_len, img_size[0], img_size[1], 3))
    video_tensor = tf.image.convert_image_dtype(video_tensor, tf.float32)

    # Normalize like ImageNet
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    video_tensor = (video_tensor - mean) / std
    return video_tensor


def preprocess_video(path, label, seq_len=cf.SEQ_LEN, img_size=cf.IMG_SIZE):
    video = decode_video(path, seq_len, img_size)
    return video, label


def load_video_dataset(root_dir, split='train', batch_size=cf.BATCH_SIZE_STT_VIDEO, shuffle=True):
    data_dir = os.path.join(root_dir, split)
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_files = []
    all_labels = []

    for cls in classes:
        pattern = os.path.join(data_dir, cls, '*.mp4')
        files = glob.glob(pattern)
        all_files.extend(files)
        all_labels.extend([class_to_idx[cls]] * len(files))

    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))
    dataset = dataset.map(lambda f, l: preprocess_video(f, l), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, class_to_idx
