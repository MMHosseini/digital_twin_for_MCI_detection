import tensorflow as tf
from data_handler import DataHandler
from control_panel import ControlPanel

controller = ControlPanel()
data_handler = DataHandler()


class AudioModelHandler:
    def vocal_twin_loss(self, true_prosody, pred_prosody, true_wavlm, pred_wavlm, lambda_prosody=1.0, lambda_wavlm_mse=1.0, lambda_wavlm_cos=0.5):
        prosody_loss = tf.reduce_mean(tf.square(true_prosody - pred_prosody))
        wavlm_mse_loss = tf.reduce_mean(tf.square(true_wavlm - pred_wavlm))
        pred_wavlm_norm = tf.math.l2_normalize(pred_wavlm, axis=1)
        true_wavlm_norm = tf.math.l2_normalize(true_wavlm, axis=1)
        cosine_sim = tf.reduce_sum(pred_wavlm_norm * true_wavlm_norm, axis=1)
        wavlm_cos_loss = 1.0 - tf.reduce_mean(cosine_sim)

        total_loss = lambda_prosody * prosody_loss + lambda_wavlm_mse * wavlm_mse_loss + lambda_wavlm_cos * wavlm_cos_loss

        return total_loss, {
            "total_loss": total_loss,
            "prosody_loss": prosody_loss,
            "wavlm_mse_loss": wavlm_mse_loss,
            "wavlm_cos_loss": wavlm_cos_loss,
        }

    @tf.function
    def train_step(self, model, optimizer, text_emb, avg_xvector, true_prosody, true_wavlm):
        with tf.GradientTape() as tape:
            pred_prosody, pred_wavlm = model([text_emb, avg_xvector], training=True)
            total_loss, loss_dict = self.vocal_twin_loss(true_prosody=true_prosody, pred_prosody=pred_prosody, true_wavlm=true_wavlm, pred_wavlm=pred_wavlm)
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss_dict

    @tf.function
    def val_step(self, model, text_emb, avg_xvector, true_prosody, true_wavlm):
        pred_prosody, pred_wavlm = model([text_emb, avg_xvector], training=False)
        total_loss, loss_dict = self.vocal_twin_loss(true_prosody=true_prosody, pred_prosody=pred_prosody, true_wavlm=true_wavlm, pred_wavlm=pred_wavlm)
        return loss_dict

    def train_model(self, model, train_dataset, val_dataset, epochs=20, lr=1e-3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for epoch in range(epochs):
            train_losses = []
            for batch in train_dataset:
                text_emb, avg_xvector, prosody, wavlm = batch
                loss_dict = self.train_step(model, optimizer, text_emb, avg_xvector, prosody, wavlm)
                train_losses.append(loss_dict["total_loss"].numpy())

            val_losses = []
            for batch in val_dataset:
                text_emb, avg_xvector, prosody, wavlm = batch
                loss_dict = self.val_step(model, text_emb, avg_xvector, prosody, wavlm)
                val_losses.append(loss_dict["total_loss"].numpy())

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={sum(train_losses) / len(train_losses):.4f} | "
                f"val_loss={sum(val_losses) / len(val_losses):.4f}"
            )

    def train_audio_mimicker(self):
        text_embs, avg_xvectors, prosody_targets, wavlm_targets = data_handler.load_data_for_audio_mimic()
        val_text_embs, val_avg_xvectors, val_prosody_targets, val_wavlm_targets = data_handler.load_data_for_audio_mimic()
        train_dataset = tf.data.Dataset.from_tensor_slices((text_embs, avg_xvectors, prosody_targets, wavlm_targets)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_text_embs, val_avg_xvectors, val_prosody_targets, val_wavlm_targets)).batch(32)

        model = VocalTwinNet(text_dim=768, xvector_dim=512, prosody_dim=20, wavlm_dim=768, hidden_dims=(512, 256), dropout=0.2)
        dummy_text = tf.zeros((2, 768))
        dummy_xvec = tf.zeros((2, 512))
        _ = model([dummy_text, dummy_xvec])
        self.train_model(model=model, train_dataset=train_dataset, val_dataset=val_dataset, epochs=20, lr=1e-3)


class VocalTwinNet(tf.keras.Model):
    def __init__(self, text_dim: int, xvector_dim: int, prosody_dim: int, wavlm_dim: int, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        self.hidden_layers = []
        for h in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(h, activation="relu"))
            self.hidden_layers.append(tf.keras.layers.Dropout(dropout))

        self.prosody_head = tf.keras.layers.Dense(prosody_dim, name="prosody_head")
        self.wavlm_head = tf.keras.layers.Dense(wavlm_dim, name="wavlm_head")

    def call(self, inputs, training=False, mask=False):
        """
        inputs:
            text_emb:    [B, text_dim]
            avg_xvector: [B, xvector_dim]
        """
        text_emb, avg_xvector = inputs
        x = tf.concat([text_emb, avg_xvector], axis=1)

        for layer in self.hidden_layers:
            x = layer(x, training=training)

        pred_prosody = self.prosody_head(x)
        pred_wavlm = self.wavlm_head(x)
        return pred_prosody, pred_wavlm


