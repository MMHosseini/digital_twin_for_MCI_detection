import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from control_panel import ControlPanel
from data_handler import DataHandler
from plot_handler import PlotHandler

controller = ControlPanel()
data_handler = DataHandler()
plot_handler = PlotHandler()

class MCIHandler:
    def report_train_and_validation_loss(self):
        participants = os.listdir(controller.train_data_path)
        # features = ['embedding_text', 'sentiment_text', 'prosody_audio', 'wavlm_audio']
        features = ['embedding_text', 'sentiment_text']

        for participant in participants:
            train_loss = []
            validation_loss = []
            train_condition, train_question, train_answer, train_mci = data_handler.load_mci_data('train', features, participant,
                                                                                                  'participant_transcript')
            train_array = [train_question, train_condition, train_answer, train_mci]
            train_set = data_handler.make_tf_dataset(train_array, batch_size=controller.batch_size, shuffle=True)

            validation_condition, validation_question, validation_answer, validation_mci = data_handler.load_mci_data('validation', features,
                                                                                                                      participant, 'participant_transcript')
            validation_array = [validation_question, validation_condition, validation_answer, validation_mci]
            validation_set = data_handler.make_tf_dataset(validation_array, batch_size=controller.batch_size, shuffle=True)

            input_dim = train_question.shape[1]
            condition_dim = train_condition.shape[1]
            latent_dim = 128
            output_dim = train_answer.shape[1]

            del train_condition
            del train_question
            del train_answer
            del train_mci
            del validation_condition
            del validation_question
            del validation_answer
            del validation_mci

            cvae_handler = CVAEModel()
            model = cvae_handler.create_model(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim, output_dim=output_dim)

            learning_rate = controller.learning_rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            model_path = os.path.join(controller.model_path, participant)
            os.makedirs(model_path, exist_ok=True)

            epochs = 20
            for epoch in range(epochs):
                total_train_loss = 0
                total_train_reconstruction_loss = 0
                total_train_kl_divergence = 0
                total_train_mci_loss = 0
                train_batch_index = 0
                for question, condition, answer, mci_score in train_set:
                    with tf.GradientTape() as tape:
                        reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=True)
                        loss, reconstruction_loss, kl_divergence, mci_loss = self.claculate_loss(answer, reconstruction, z_mean, z_log_var, mci_score, mci_prediction)
                    model_gradient = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
                    train_batch_index += 1
                    total_train_loss += float(loss)
                    total_train_reconstruction_loss += float(reconstruction_loss)
                    total_train_kl_divergence += float(kl_divergence)
                    total_train_mci_loss += float(mci_loss)

                average_train_loss = total_train_loss / train_batch_index
                average_train_reconstruction_loss = total_train_reconstruction_loss / train_batch_index
                average_train_kl_divergence = total_train_kl_divergence / train_batch_index
                average_train_mci_loss = total_train_mci_loss / train_batch_index
                train_loss.append([average_train_loss, average_train_reconstruction_loss, average_train_kl_divergence, average_train_mci_loss])

                total_validation_loss = 0
                total_validation_reconstruction_loss = 0
                total_validation_kl_divergence = 0
                total_validation_mci_loss = 0
                validation_batch_index = 0
                for question, condition, answer, mci_score in validation_set:
                    reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=True)
                    loss, reconstruction_loss, kl_divergence, mci_loss = self.claculate_loss(answer, reconstruction, z_mean, z_log_var, mci_score,
                                                                                             mci_prediction)
                    validation_batch_index += 1
                    total_validation_loss += float(loss)
                    total_validation_reconstruction_loss += float(reconstruction_loss)
                    total_validation_kl_divergence += float(kl_divergence)
                    total_validation_mci_loss += float(mci_loss)

                average_validation_loss = total_validation_loss / validation_batch_index
                average_validation_reconstruction_loss = total_validation_reconstruction_loss / validation_batch_index
                average_validation_kl_divergence = total_validation_kl_divergence / validation_batch_index
                average_validation_mci_loss = total_validation_mci_loss / validation_batch_index
                validation_loss.append([average_validation_loss, average_validation_reconstruction_loss, average_validation_kl_divergence, average_validation_mci_loss])

            train_loss = np.array(train_loss).T
            validation_loss = np.array(validation_loss).T
            labels = ['All', 'Reconstruction', 'KL-Divergence', 'MCI']
            plot_handler.plot_cvae_losses(train_loss, controller.plot_path, file_name='cvae_train_loss_'+ participant + '.svg', labels=labels)
            plot_handler.plot_cvae_losses(validation_loss, controller.plot_path, file_name='cvae_validation_loss_'+ participant + '.svg', labels=labels)

    def train_cvae_model(self):
        self._train_cvae_model(features=['embedding_text', 'sentiment_text'], feature_type='text')
        self._train_cvae_model(features=['prosody_audio', 'wavlm_audio'], feature_type='audio')
        self._train_cvae_model(features=['embedding_text', 'sentiment_text', 'prosody_audio', 'wavlm_audio'], feature_type='text_and_audio')

    def _train_cvae_model(self, features, feature_type):
        participants = os.listdir(controller.train_data_path)
        for participant in participants:
            train_condition, train_question, train_answer, train_mci = data_handler.load_mci_data('train', features, participant,
                                                                                                           'participant_transcript')
            train_array = [train_question, train_condition, train_answer, train_mci]
            train_set = data_handler.make_tf_dataset(train_array, batch_size=controller.batch_size, shuffle=True)

            validation_condition, validation_question, validation_answer, validation_mci = data_handler.load_mci_data('validation', features,
                                                                                                                               participant, 'participant_transcript')
            validation_array = [validation_question, validation_condition, validation_answer, validation_mci]
            validation_set = data_handler.make_tf_dataset(validation_array, batch_size=controller.batch_size, shuffle=True)

            input_dim = train_question.shape[1]
            condition_dim = train_condition.shape[1]
            latent_dim = 128
            output_dim = train_answer.shape[1]

            del train_condition
            del train_question
            del train_answer
            del train_mci
            del validation_condition
            del validation_question
            del validation_answer
            del validation_mci

            cvae_handler = CVAEModel()
            model = cvae_handler.create_model(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim, output_dim=output_dim)

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
                        loss, reconstruction_loss, kl_divergence, mci_loss = self.claculate_loss(answer, reconstruction, z_mean, z_log_var, mci_score, mci_prediction)
                    model_gradient = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(model_gradient, model.trainable_variables))
                    batch_index += 1

                    total_loss += loss
                    average_loss = total_loss / batch_index
                    # tf.print("EPOCH: ", epoch, '/', epochs, ", BATCH: ", batch_index, '/', num_batch, ', BATCH_LOSS: ', loss, ', EPOCH_LOSS: ', average_loss)

                # save_path = os.path.join(model_path, 'CVAE_epoch_' + str(int(epoch)) + '_loss_' + str(float(average_loss))[:6] + '.h5')
                # model.save(save_path)

                current_model_loss = self._evaluate_model(model, validation_set)
                if current_model_loss < best_model_loss:
                    best_model_loss = current_model_loss
                    save_path = os.path.join(model_path, 'best_' + feature_type + '_modality_mci_predictor.h5')
                    model.save(save_path)
                    print('best model mci error for modality:', feature_type, ', participant:', participant, '-->', best_model_loss)

    def validate_cvae_model(self):
        self._validate_cvae_model(features=['embedding_text', 'sentiment_text'], feature_type='text')
        self._validate_cvae_model(features=['prosody_audio', 'wavlm_audio'], feature_type='audio')
        self._validate_cvae_model(features=['embedding_text', 'sentiment_text', 'prosody_audio', 'wavlm_audio'], feature_type='text_and_audio')

    def _validate_cvae_model(self, features, feature_type):
        participants = os.listdir(controller.validation_data_path)
        subset = 'validation'
        respondents = ['participant_transcript', 'simulated_transcript_sft', 'simulated_transcript_raw_gpt']

        for participant in participants:
            for respondent in respondents:
                validation_condition, validation_question, validation_answer, validation_mci = data_handler.load_mci_data(subset, features,
                                                                                                                                   participant, respondent)
                validation_array = [validation_question, validation_condition, validation_answer, validation_mci]
                validation_set = data_handler.make_tf_dataset(validation_array, batch_size=controller.batch_size, shuffle=True)
                num_batch = len(validation_set)

                model_path = os.path.join(controller.model_path, participant, 'best_' + feature_type + '_modality_mci_predictor.h5')
                model = load_model(model_path)

                del validation_condition
                del validation_question
                del validation_answer
                del validation_mci

                total_mci_loss = 0
                total_reconstruction_loss = 0
                all_mci_scores = []
                all_mci_predictions = []
                for question, condition, answer, mci_score in validation_set:
                    reconstruction, mci_prediction, z_mean, z_log_var = model([question, condition], training=False)
                    mci_loss = tf.reduce_mean(tf.abs(mci_prediction - mci_score)) * 30
                    all_mci_scores += (mci_score[:, 0].numpy() * 30.0).tolist()
                    all_mci_predictions += (mci_prediction[:, 0].numpy() * 30.0).tolist()
                    reconstruction_loss = tf.reduce_mean(tf.square(reconstruction - answer))
                    total_mci_loss += mci_loss
                    total_reconstruction_loss += reconstruction_loss
                mci_diff = float(total_mci_loss) / num_batch
                reconstruction_diff = float(total_reconstruction_loss) / num_batch
                mci_std = np.std(np.abs(np.array(all_mci_scores) - np.array(all_mci_predictions)))
                print('participant:', participant, ', feature type', feature_type, ', respondent:', respondent, '-->  mci diff:', mci_diff,
                      '-->  mci std:', mci_std, '--> reconstruction diff:', reconstruction_diff)
                
                save_path = os.path.join(controller.result_path, feature_type + '_based_mci_prediction.txt')
                with open(save_path, "a") as f:
                    f.write(f"participant: {participant}, respondent: {respondent} --> mci diff: {mci_diff} --> mci std: {mci_std} --> "
                            f"reconstruction diff: {reconstruction_diff}\n")
                    f.write("============================================\n")

    def claculate_loss(self, real, prediction, z_mean, z_log_var, mci_true, mci_pred):
        reconstruction_loss = 1 * tf.reduce_mean(tf.square(real - prediction))
        kl_divergence = 1 * -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        mci_loss = 100 * tf.reduce_mean(tf.square(mci_true - mci_pred))
        loss = reconstruction_loss + kl_divergence + mci_loss
        return loss, reconstruction_loss, kl_divergence, mci_loss

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
    def create_model(self, input_dim = 768, condition_dim = 7, latent_dim = 64, output_dim = 768):
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
        decoder_output_layer = layers.Dense(output_dim, name='output_layer')(decoder_layer_2)

        # Define VAE model with encoder and decoder
        cvae_model = Model(inputs=[input_layer, condition_layer], outputs=[decoder_output_layer, mci_predictor_output_layer, z_mean, z_log_var], name='cvae_model')
        # cvae_model.summary()
        return cvae_model