# sign_to_text/src/main/infer.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)
import collections
import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
import config as cf

from sign_to_text.models.mobinetV2 import create_mobilenetv2, create_mobilenet_lstm

def load_static_model():
    num_classes = None
    if not os.path.exists(cf.CKPT_StT_STATIC_BEST ):
        raise FileNotFoundError("Static model checkpoint not found.")
    # Load class_to_idx from config or separate file if you saved it
    # Here you may need to hardcode or load from file
    # For demo, assume class_to_idx saved as JSON or you know class count
    # Suppose num_classes = 30 for example
    # Better: store and load class_to_idx from separate file during training
    
    # We'll just load model weights and assume num_classes known
    # For demo, just load model by rebuilding with number classes
    # TODO: Adjust to load real class_to_idx mapping if needed
    
    # Just rebuild model with correct num_classes
    # For demonstration, let's set dummy num_classes=30 (change as needed)
    num_classes = 30
    
    model = create_mobilenetv2(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_weights(cf.CKPT_StT_STATIC_BEST)
    return model

def load_video_model():
    num_classes = 30  # Change accordingly
    model = create_mobilenet_lstm(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_weights(cf.CKPT_StT_VIDEO_BEST)
    return model

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(frame_rgb, cf.IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = img / 255.0
    # Normalize by ImageNet mean/std
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img

def build_video_tensor(frames_bgr):
    if len(frames_bgr) == 0:
        return None
    # Uniformly sample SEQ_LEN frames
    idxs = np.linspace(0, max(len(frames_bgr)-1, 0), num=cf.SEQ_LEN, dtype=np.int32)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    sampled = []
    for i in idxs:
        img = tf.image.resize(frames_rgb[i], cf.IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std
        sampled.append(img)
    seq = tf.stack(sampled)  # shape (SEQ_LEN, H, W, C)
    return seq

def detect_motion(frames_bgr, thresh=cf.MOTION_THRESH, min_frames=cf.MOTION_MIN_FRAMES):
    motion_count = 0
    prev_gray = None
    for f in frames_bgr:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            _, m = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            score = (m.sum() / 255)
            if score > thresh:
                motion_count += 1
        prev_gray = gray
    return motion_count >= min_frames

def majority_vote(deq):
    if not deq:
        return None
    counter = collections.Counter(deq)
    return counter.most_common(1)[0][0]

def main():
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print("Device:", device)

    with tf.device(device):
        static_model = load_static_model()
        video_model = load_video_model()

        # TODO: You need to load actual class_to_idx and idx_to_class mapping used for models
        # For demo, we create dummy idx_to_class mapping (change accordingly)
        idx_to_class = {i: f"Class_{i}" for i in range(30)}

    cap = cv2.VideoCapture(0)
    fps = cf.FPS
    max_buf = int(cf.WINDOW_SECONDS * fps)
    frame_buffer = collections.deque(maxlen=max_buf)
    pred_buffer = collections.deque(maxlen=cf.PRED_SMOOTHING)

    last_static_infer = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_id += 1
        frame_disp = frame.copy()
        frame_buffer.append(frame)

        use_video = False
        if len(frame_buffer) >= min(12, max_buf):
            if detect_motion(list(frame_buffer)):
                use_video = True

        pred_label = None
        conf = None

        if use_video and len(frame_buffer) >= cf.SEQ_LEN:
            x = build_video_tensor(list(frame_buffer))
            if x is not None:
                x = tf.expand_dims(x, axis=0)  # (1, T, H, W, C)
                logits = video_model(x, training=False)
                prob = tf.nn.softmax(logits[0])
                top = tf.argmax(prob).numpy()
                conf = float(prob[top].numpy())
                pred_label = idx_to_class.get(top, None)
        else:
            if frame_id - last_static_infer >= cf.STATIC_INFER_EVERY:
                last_static_infer = frame_id
                x = preprocess_frame(frame)
                x = tf.expand_dims(x, axis=0)  # (1, H, W, C)
                logits = static_model(x, training=False)
                prob = tf.nn.softmax(logits[0])
                top = tf.argmax(prob).numpy()
                conf = float(prob[top].numpy())
                pred_label = idx_to_class.get(top, None)

        if pred_label is not None:
            pred_buffer.append(pred_label)
            smooth_label = majority_vote(pred_buffer)
        else:
            smooth_label = majority_vote(pred_buffer)

        text = f"MODE: {'VIDEO' if use_video else 'STATIC'}  PRED: {smooth_label if smooth_label else '-'}"
        cv2.putText(frame_disp, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('SignLangAssist (motion-first)', frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
