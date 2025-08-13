# sign_to_text/src/main/train_chars.py
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)
import time
import copy
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import config as cf

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, root_dir)

from sign_to_text.models.mobinetV2 import create_mobilenetv2_model
from sign_to_text.src.main.dataset import load_static_dataset  # giữ nguyên nếu có sẵn

def plot_history(history, save_dir):
    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_plot.png')
    plt.savefig(save_path)
    
    plt.show()
    print(f"Saved training plot to: {save_path}")

def get_run_folder(base_dir):
    # Format ngày tháng
    date_str = time.strftime("%Y-%m-%d")
    day_dir = os.path.join(base_dir, date_str)
    os.makedirs(day_dir, exist_ok=True)

    # Đếm xem đã có bao nhiêu lần train trong ngày này
    runs = [d for d in os.listdir(day_dir) if os.path.isdir(os.path.join(day_dir, d)) and d.startswith("run")]
    run_number = len(runs) + 1
    run_dir = os.path.join(day_dir, f"run{run_number}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print("Device:", device)

    with tf.device(device):
        train_ds, class_to_idx = load_static_dataset(cf.DATA_DIR_StT, split='train', batch_size=cf.BATCH_SIZE_STT_STATIC, shuffle=True)
        val_ds, _ = load_static_dataset(cf.DATA_DIR_StT, split='val', batch_size=cf.BATCH_SIZE_STT_STATIC, shuffle=False)
        num_classes = len(class_to_idx)

        print("Creating model...")        
        model = create_mobilenetv2_model(num_classes, pretrained=True, freeze_backbone=True)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=cf.LR),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        best_ckpt_path = cf.CKPT_StT_STATIC_BEST

        checkpoint_cb = callbacks.ModelCheckpoint(
            filepath=best_ckpt_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        tensorboard_cb = callbacks.TensorBoard(log_dir=cf.LOG_DIR_StT_STATIC)

        # Train with frozen backbone
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cf.UNFREEZE_EPOCH,
            callbacks=[checkpoint_cb, tensorboard_cb]
        )

        # Unfreeze backbone for fine-tuning
        print("Unfreezing backbone for fine-tuning.")
        model.trainable = True
        model.compile(
            optimizer=optimizers.Adam(learning_rate=cf.FT_LR),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cf.EPOCHS,
            initial_epoch=cf.UNFREEZE_EPOCH,
            callbacks=[checkpoint_cb, tensorboard_cb]
        )

        # Save final model weights
        final_ckpt_path = cf.CKPT_StT_STATIC_FINAL
        model.save_weights(final_ckpt_path)
        print("Training complete. Best weights saved.")

        # Merge lịch sử 2 giai đoạn
        history = {}
        for k in history1.history.keys():
            history[k] = history1.history[k] + history2.history[k]

        # Tạo thư mục lưu kết quả theo ngày và lần chạy
        save_dir = os.path.join(cf.RESULTS_STT_CHAR_DIR, get_run_folder(''))
        plot_history(type('History', (), {'history': history})(), save_dir)

if __name__ == '__main__':
    main()
    print("Done!")