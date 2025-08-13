import os

# ========================
# Base paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# Data cho Sign-to-Text (ảnh tĩnh)
# ========================
DATA_DIR_StT = os.path.join(BASE_DIR, 'sign_to_text', 'dataset')
TRAIN_DIR_StT = os.path.join(DATA_DIR_StT, 'train')
VAL_DIR_StT = os.path.join(DATA_DIR_StT, 'val')

# ========================
# Data cho Sign-to-Text (video động <3s)
# ========================
VIDEO_DATA_DIR_StT = os.path.join(BASE_DIR, 'sign_to_text', 'dataset_video')
VIDEO_TRAIN_DIR_StT = os.path.join(VIDEO_DATA_DIR_StT, 'train')
VIDEO_VAL_DIR_StT   = os.path.join(VIDEO_DATA_DIR_StT, 'val')

# ========================
# Data cho Text-to-Sign
# ========================
DATA_DIR_TtS = os.path.join(BASE_DIR, 'text_to_sign', 'dataset')
TRAIN_DIR_TtS = os.path.join(DATA_DIR_TtS, 'train')
VAL_DIR_TtS   = os.path.join(DATA_DIR_TtS, 'val')

# ========================
# Checkpoints (tách riêng, dùng cho TF: lưu .h5)
# ========================
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint')
CKPT_StT_STATIC_BEST  = os.path.join(CHECKPOINT_DIR, 'stt_static_best.weights.h5')
CKPT_StT_STATIC_FINAL = os.path.join(CHECKPOINT_DIR, 'stt_static_final.weights.h5')
CKPT_StT_VIDEO_BEST   = os.path.join(CHECKPOINT_DIR, 'stt_video_best.weights.h5')
CKPT_StT_VIDEO_FINAL  = os.path.join(CHECKPOINT_DIR, 'stt_video_final.weights.h5')
CKPT_TtS_BEST         = os.path.join(CHECKPOINT_DIR, 'tts_best.h5')      # (dành cho mô hình TtS)
CKPT_TtS_FINAL        = os.path.join(CHECKPOINT_DIR, 'tts_final.h5')

# ========================
# Results + Logs
# ========================
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
RESULTS_StT_DIR = os.path.join(RESULTS_DIR, 'sign_to_text')
RESULTS_STT_CHAR_DIR = os.path.join(RESULTS_StT_DIR, 'char')
RESULTS_STT_WORD_DIR = os.path.join(RESULTS_StT_DIR, 'word')
RESULTS_TtS_DIR = os.path.join(RESULTS_DIR, 'text_to_sign')
RESULTS_TTS_VIDEO_DIR = os.path.join(RESULTS_TtS_DIR, 'video')

RUNS_DIR = os.path.join(BASE_DIR, 'runs')
LOG_DIR_StT_STATIC = os.path.join(RUNS_DIR, 'stt_static')
LOG_DIR_StT_VIDEO  = os.path.join(RUNS_DIR, 'stt_video')
LOG_DIR_TtS        = os.path.join(RUNS_DIR, 'tts')

# ========================
# Hyper-params (dùng chung)
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE_STT_STATIC = 32
BATCH_SIZE_STT_VIDEO  = 8
NUM_WORKERS = 4
PIN_MEMORY = True
EPOCHS = 20
LR = 1e-3
FT_LR = 1e-4
UNFREEZE_EPOCH = 5
SEED = 42

# ========================
# Video params (sequence)
# ========================
FPS = 30
WINDOW_SECONDS = 3
SEQ_LEN = 32
FEATURE_DIM = 1280
LSTM_HIDDEN = 512
LSTM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.2

# ========================
# Motion detection (real-time)
# ========================
MOTION_THRESH = 2000
MOTION_MIN_FRAMES = 6
PRED_SMOOTHING = 5
STATIC_INFER_EVERY = 4

# Tạo thư mục cần thiết
for d in [
    CHECKPOINT_DIR,
    RESULTS_STT_CHAR_DIR, RESULTS_STT_WORD_DIR,
    RESULTS_TTS_VIDEO_DIR,
    LOG_DIR_StT_STATIC, LOG_DIR_StT_VIDEO, LOG_DIR_TtS
]:
    os.makedirs(d, exist_ok=True)
