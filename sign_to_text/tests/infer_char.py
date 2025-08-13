# sign_to_text/src/main/infer_char.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

import collections
import cv2
import tensorflow as tf
import numpy as np
import config as cf

from sign_to_text.models.mobinetV2 import create_mobilenetv2_model

def count_classes_in_dataset():
    train_char_dir = os.path.join(cf.DATA_DIR_StT, 'train', 'char')
    classes = [d for d in os.listdir(train_char_dir) if os.path.isdir(os.path.join(train_char_dir, d))]
    classes = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return class_to_idx

def load_static_model():
    if not os.path.exists(cf.CKPT_StT_STATIC_BEST):
        raise FileNotFoundError("Static model checkpoint not found.")

    class_to_idx = count_classes_in_dataset()
    num_classes = len(class_to_idx)

    model = create_mobilenetv2_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_weights(cf.CKPT_StT_STATIC_BEST)
    return model, class_to_idx

def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(frame_rgb, cf.IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img

def majority_vote(deq):
    if not deq:
        return None
    counter = collections.Counter(deq)
    return counter.most_common(1)[0][0]

def main():
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print("Device:", device)

    with tf.device(device):
        model, class_to_idx = load_static_model()
        idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("Model loaded.")
    
    cap = cv2.VideoCapture(0)
    pred_buffer = collections.deque(maxlen=cf.PRED_SMOOTHING)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            x = preprocess_frame(frame)
            x = tf.expand_dims(x, axis=0)  # (1, H, W, C)

            logits = model(x, training=False)
            prob = tf.nn.softmax(logits[0])
            top_idx = tf.argmax(prob).numpy()
            conf = float(prob[top_idx])
            pred_label = idx_to_class.get(top_idx, None)

            pred_buffer.append(pred_label)
            smooth_label = majority_vote(pred_buffer)

            text = f"PRED: {smooth_label if smooth_label else '-'}  CONF: {conf:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('SignLangAssist - Char Inference', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
