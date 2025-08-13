#sign_to_text/models/mobinetV2.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_mobilenetv2_model(num_classes, pretrained=True, freeze_backbone=True, input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet' if pretrained else None,
        pooling='avg'
    )
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_backbone)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model


class MobileNetLSTM(tf.keras.Model):
    def __init__(self, num_classes,
                 feature_dim=1280,
                 hidden_size=512,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.2,
                 pretrained=True,
                 freeze_backbone=True,
                 seq_len=32,
                 input_shape=(224, 224, 3)):
        super().__init__()

        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights='imagenet' if pretrained else None
        )
        self.base_model.trainable = not freeze_backbone

        self.pool = layers.GlobalAveragePooling2D()

        self.lstm_layers = []
        for i in range(num_layers):
            lstm = layers.Bidirectional(
                layers.LSTM(hidden_size, dropout=dropout, return_sequences=(i != num_layers - 1))
            ) if bidirectional else layers.LSTM(hidden_size, dropout=dropout, return_sequences=(i != num_layers - 1))
            self.lstm_layers.append(lstm)

        self.dropout = layers.Dropout(dropout)
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        # inputs shape: (batch, seq_len, H, W, C)
        batch_size = tf.shape(inputs)[0]

        # flatten batch and seq_len for feature extraction
        x = tf.reshape(inputs, (-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]))  # (B*T, H, W, C)
        features = self.base_model(x, training=training)
        pooled = self.pool(features)  # (B*T, feature_dim)
        # reshape back to (B, T, feature_dim)
        seq_features = tf.reshape(pooled, (batch_size, self.seq_len, self.feature_dim))

        # pass through stacked LSTM layers
        x_lstm = seq_features
        for lstm in self.lstm_layers:
            x_lstm = lstm(x_lstm, training=training)

        x_lstm = self.dropout(x_lstm, training=training)
        logits = self.classifier(x_lstm)  # (B, num_classes)
        return logits


def create_mobilenet_lstm(num_classes, **kwargs):
    return MobileNetLSTM(num_classes, **kwargs)
