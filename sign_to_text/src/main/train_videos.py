# sign_to_text/src/main/train_videos.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)
import time
import copy
import tensorflow as tf
from tensorflow.keras import optimizers, losses, callbacks
from sklearn.metrics import accuracy_score
import config as cf

from sign_to_text.models.mobinetV2 import create_mobilenet_lstm
from sign_to_text.src.main.video_dataset import load_video_dataset  # bạn phải có file này

def main():
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print("Device:", device)

    with tf.device(device):
        train_ds, class_to_idx = load_video_dataset(cf.VIDEO_TRAIN_DIR_StT, batch_size=cf.BATCH_SIZE_STT_VIDEO, shuffle=True)
        val_ds, _ = load_video_dataset(cf.VIDEO_VAL_DIR_StT, batch_size=cf.BATCH_SIZE_STT_VIDEO, shuffle=False)
        num_classes = len(class_to_idx)

        model = create_mobilenet_lstm(
            num_classes,
            pretrained=True,
            freeze_backbone=True,
            seq_len=cf.SEQ_LEN,
            input_shape=(*cf.IMG_SIZE, 3)
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=cf.LR),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        best_ckpt_path = cf.CKPT_StT_VIDEO_BEST

        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        tensorboard_cb = callbacks.TensorBoard(log_dir=cf.LOG_DIR_StT_VIDEO)

        # Train with frozen backbone
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cf.UNFREEZE_EPOCH,
            callbacks=[checkpoint_cb, tensorboard_cb]
        )

        # Unfreeze backbone for fine-tuning
        print("Unfreezing backbone for fine-tuning.")
        model.base_model.trainable = True  # chỉ unfreeze backbone trong base_model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=cf.FT_LR),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cf.EPOCHS,
            initial_epoch=cf.UNFREEZE_EPOCH,
            callbacks=[checkpoint_cb, tensorboard_cb]
        )

        # Save final model weights
        final_ckpt_path = cf.CKPT_StT_VIDEO_FINAL
        model.save_weights(final_ckpt_path)
        print("Training complete. Best weights saved.")

if __name__ == '__main__':
    os.makedirs(cf.CHECKPOINT_DIR, exist_ok=True)
    main()
