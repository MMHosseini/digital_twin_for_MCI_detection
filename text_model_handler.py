import os
import tensorflow as tf
from openai import models
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.python.distribute.device_util import current

from control_panel import ControlPanel
from data_handler import DataHandler

controller = ControlPanel()
data_handler = DataHandler()

class TextModelHandler:
    def train_model(self):
        participants = os.listdir(controller.train_data_path)
        features = ['embedding_text', 'sentiment_text']

        for participant in participants:
            train_condition, train_question, train_answer, train_mci = data_handler.load_data(subset='train', features=features, participant=participant,
                                                                                              respondent='participant_transcript')
            train_set = data_handler.make_tf_dataset(train_question, train_condition, train_answer, train_mci, batch_size=controller.batch_size, shuffle=True)

            validation_condition, validation_question, validation_answer, validation_mci = data_handler.load_data(subset='validation', features=features,
                                                                                                                  participant=participant,
                                                                                                                  respondent='participant_transcript')
            validation_set = data_handler.make_tf_dataset(validation_question, validation_condition, validation_answer, validation_mci,
                                                          batch_size=controller.batch_size, shuffle=True)
            input_dim = train_question.shape[1]
            condition_dim = train_condition.shape[1]
            latent_dim = 128

            del train_condition
            del train_question
            del train_answer
            del train_mci
            del validation_condition
            del validation_question
            del validation_answer
            del validation_mci

            cvae_handler = CVAEModel()
            model = cvae_handler.create_model(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim)

            learning_rate = controller.learning_rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model_path = os.path.join(controller.model_path, participant)
            os.makedirs(model_path, exist_ok=True)

            num_batch = len(train_set)
            epochs = controller.epochs
            best_model_loss = float('inf')
            for epoch in range(epochs):
                total_loss = 0
                average_loss = 0
                batch_index = 0
                for question, condition, answer, mci_score in train_set:
                    with tf.GradientTape() as tape:
                        reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=True)
                        loss = self.claculate_loss(answer, reconstruction, z_mean, z_log_var, mci_score, mci_prediction)
                    model_gradient = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
                    batch_index += 1

                    total_loss += loss
                    average_loss = total_loss / batch_index
                    # tf.print("EPOCH: ", epoch, '/', epochs, ", BATCH: ", batch_index, '/', num_batch, ', BATCH_LOSS: ', loss, ', EPOCH_LOSS: ', average_loss)

                save_path = os.path.join(model_path, 'CVAE_epoch_' + str(int(epoch)) + '_loss_' + str(float(average_loss))[:6] + '.h5')
                model.save(save_path)

                current_model_loss = self._evaluate_model(model, validation_set)
                if current_model_loss < best_model_loss:
                    best_model_loss = current_model_loss
                    save_path = os.path.join(model_path, 'best.h5')
                    model.save(save_path)
                    print('best_model_loss', best_model_loss)

    def validate_model(self):
        participants = os.listdir(controller.validation_data_path)
        subset = 'validation'
        features = ['embedding_text', 'sentiment_text']
        respondents = ['participant_transcript', 'simulated_transcript_sft', 'simulated_transcript_raw_gpt']
        respondents = ['participant_transcript']

        for participant in participants:
            for respondent in respondents:
                validation_condition, validation_question, validation_answer, validation_mci = data_handler.load_data(subset, features, participant, respondent)
                validation_set = data_handler.make_tf_dataset(validation_question, validation_condition, validation_answer,
                                                              validation_mci, batch_size=controller.batch_size, shuffle=False)
                num_batch = len(validation_set)

                models_path = os.path.join(controller.model_path, participant)
                model_path = os.path.join(models_path, 'best.h5')
                model = load_model(model_path)

                del validation_condition
                del validation_question
                del validation_answer
                del validation_mci

                total_mci_loss = 0
                for question, condition, answer, mci_score in validation_set:
                    reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=False)
                    mci_loss = tf.reduce_mean(tf.abs(mci_prediction - mci_score)) * 30
                    total_mci_loss += mci_loss
                print('participant:', participant, ', respondent:' + respondent + '-->  mci diff is:', float(total_mci_loss) / num_batch)


    def claculate_loss(self, real, prediction, z_mean, z_log_var, mci_true, mci_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(real - prediction))
        kl_divergence = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        mci_loss = 100 * tf.reduce_mean(tf.square(mci_true - mci_pred))
        loss = reconstruction_loss + kl_divergence + mci_loss
        return loss

    def _evaluate_model(self, model, validation_set):
        num_batch = len(validation_set)
        total_loss = 0
        for question, condition, answer, mci_score in validation_set:
            reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=True)
            loss = tf.reduce_mean(tf.abs(mci_prediction - mci_score)) * 30
            # loss = tf.reduce_mean(tf.abs(reconstruction - answer))
            total_loss += loss
        current_model_loss = float(total_loss) / num_batch
        return current_model_loss

class CVAEModel:
    def create_model(self, input_dim = 768, condition_dim = 7, latent_dim = 64):
        # Encoder
        input_layer = Input(shape=(input_dim,), name='input_layer')
        condition_layer = Input(shape=(condition_dim,), name='condition_layer')
        encoder_input_concat = layers.Concatenate(name='encoder_input_concat')([input_layer, condition_layer])
        encoder_layer_1 = layers.Dense(256, activation='relu', name='encoder_layer_1')(encoder_input_concat)
        encoder_layer_2 = layers.Dense(128, activation='relu', name='encoder_layer_2')(encoder_layer_1)

        # Latent space - mean and log-variance
        z_mean = layers.Dense(latent_dim, name='z_mean')(encoder_layer_2)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(encoder_layer_2)

        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        # Latent layer (z)
        z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # MCI score predictor
        mci_predictor_input_concat = layers.Concatenate(name='mci_predictor_input_concat')([z, condition_layer])
        mci_predictor_layer_1 = layers.Dense(32 + condition_dim, activation='relu', name='mci_predictor_layer_1')(mci_predictor_input_concat)
        # mci_predictor_input_concat = z
        # mci_predictor_layer_1 = layers.Dense(32, activation='relu', name='mci_predictor_layer_1')(mci_predictor_input_concat)
        mci_predictor_output_layer = layers.Dense(1, activation='sigmoid', name='mci_predictor_output_layer')(mci_predictor_layer_1)

        # Decoder
        decoder_input_concat = layers.Concatenate(name='decoder_input_concat')([z, condition_layer])
        decoder_layer_1 = layers.Dense(128 + condition_dim, activation='relu', name='decoder_layer_1')(decoder_input_concat)
        decoder_layer_2 = layers.Dense(256, activation='relu', name='decoder_layer_2')(decoder_layer_1)
        decoder_output_layer = layers.Dense(input_dim, name='output_layer')(decoder_layer_2)

        # Define VAE model with encoder and decoder
        cvae_model = Model(inputs=[input_layer, condition_layer], outputs=[decoder_output_layer, mci_predictor_output_layer, z_mean, z_log_var], name='cvae_model')
        cvae_model.summary()
        return cvae_model