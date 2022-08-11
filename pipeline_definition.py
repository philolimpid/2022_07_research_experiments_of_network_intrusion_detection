import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import h5py
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from time import process_time
import sklearn
import seaborn as sns
from tensorflow import keras
import matplotlib.ticker as ticker
import matplotlib.dates as mdate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import general_parameters

np.random.seed(general_parameters.random_seed)

plt.rcParams['font.sans-serif'] = ['STSONG']

class data_block():
    def __init__(self):
        print('done')
        self.train = [0,1]
        self.val = [0,1]
        self.test = [0,1]

class ANN_pipeline():
    def __init__(self, model_name, data_block, input_feature_list, scaled_feature,
                 data_type, resampling_type, resampling_ratio, version):
        self.model_name = model_name
        self.data_block = data_block
        self.input_feature_list = input_feature_list
        self.scaled_feature = scaled_feature
        self.data_type = data_type
        self.resampling_type = resampling_type
        self.resampling_ratio = resampling_ratio
        self.version = version

        self.output_numbers = np.linalg.matrix_rank(data_block.train[1])

    def build_and_compile_model(self):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=self.output_numbers, activation='softmax')
        ])

        self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.optimizers.Adam(learning_rate=0.001),
                           metrics=[tf.metrics.Recall()])

    def fit(self, epoch_num=10):
        project_dir, output_describe, model_name= \
            general_parameters.project_dir, r'\trained_model\check_point_for_temp_use_',self.model_name
        checkpoint_filepath = project_dir + output_describe + '_' + model_name
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.fit_start_time = process_time()
        self.history = self.model.fit(self.data_block.train[0],
                                      self.data_block.train[1],
                                      batch_size=100,
                                      epochs=epoch_num,
                                      validation_data=self.data_block.val,
                                      callbacks=[model_checkpoint_callback, tensorboard_callback],
                                      shuffle=True
                                      )
        self.model.load_weights(checkpoint_filepath)
        self.fit_end_time = process_time()

    def save_model_and_history(self):
        self.model.save_weights(
            general_parameters.project_dir + r'\trained_model\model_' + '_' + self.version + '_' + self.model_name +
            '_'+ self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        pd.DataFrame.from_dict(self.history.history).to_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv',index=False)
        pd.DataFrame([self.fit_start_time, self.fit_end_time]).to_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv', index=False)

    def load_model_and_history(self):
        self.model.load_weights(
            general_parameters.project_dir + r'\trained_model\model_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.history = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.history.history = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.time_result = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.fit_start_time = self.time_result.iloc[0, 0]
        self.fit_end_time = self.time_result.iloc[1, 0]

    def get_the_classification_results(self):
        model_output = self.model(self.data_block.test[0].to_numpy())
        self.output = tf.argmax(model_output, axis=1).numpy().tolist()
        self.label = tf.argmax(self.data_block.test[1], axis=1).numpy().tolist()

    def evaluate_the_results(self):
        from sklearn import metrics

        micro_accuracy = metrics.accuracy_score(self.label, self.output)
        micro_precision = metrics.precision_score(self.label, self.output, average='micro')
        macro_precision = metrics.precision_score(self.label, self.output, average='macro')

        micro_recall = metrics.recall_score(self.label, self.output, average='micro')
        macro_recall = metrics.recall_score(self.label, self.output, average='macro')

        micro_f1 = metrics.f1_score(self.label, self.output, average='micro')
        macro_f1 = metrics.f1_score(self.label, self.output, average='macro')

        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(self.label, self.output)

        from sklearn.metrics import classification_report
        print(classification_report(self.label, self.output,output_dict=True))

        fine_grained_evaluation = pd.DataFrame(
            classification_report(self.label, self.output, output_dict=True)).drop(['accuracy','macro avg','weighted avg'],axis=1).T
        fine_grained_evaluation = pd.melt(fine_grained_evaluation.reset_index(),id_vars='index')
        fine_grained_evaluation = fine_grained_evaluation.astype({'index':'int32'})
        fine_grained_evaluation = pd.merge(fine_grained_evaluation,fine_grained_evaluation[fine_grained_evaluation['variable']=='support'],
                                           how='left',on='index')
        fine_grained_evaluation.drop(['variable_y'],axis=1,inplace=True)
        fine_grained_evaluation = fine_grained_evaluation[fine_grained_evaluation['variable_x'] != 'support']
        fine_grained_evaluation = fine_grained_evaluation.rename(columns={'index':'class','variable_x':'metrics','value_x':'value','value_y':'support'})


        t = {}
        t[1] = ['micro', 'accuracy', micro_accuracy ,len(self.data_block.test[0])]
        t[2] = ['micro', 'precision', micro_precision ,len(self.data_block.test[0])]
        t[3] = ['micro', 'recall', micro_recall ,len(self.data_block.test[0])]
        t[4] = ['micro', 'f1-score', micro_f1 ,len(self.data_block.test[0])]
        t[5] = ['macro', 'precision', macro_precision ,len(self.data_block.test[0])]
        t[6] = ['macro', 'recall', macro_recall ,len(self.data_block.test[0])]
        t[7] = ['macro', 'f1-score', macro_f1 ,len(self.data_block.test[0])]

        self.metrics = pd.DataFrame(t).T
        print(self.metrics)

        self.confusion_matrix = pd.DataFrame(self.confusion_matrix)
        print(self.confusion_matrix)

        self.metrics.columns = ['class','metrics','value','support']
        self.metrics = pd.concat([self.metrics,fine_grained_evaluation], axis=0)
        for i in ['model_name','data_type','resampling_type','resampling_ratio','version']:
            self.metrics[i] = eval('self.'+i)
        self.metrics.to_csv(
            general_parameters.project_dir + '\experiment_output\metrics' + '_' + self.version +
            '_' + self.model_name + '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv',
            encoding='utf_8_sig',index=False)
        all_experiment_results = pd.read_csv(general_parameters.project_dir + r'\experiment_output\all_experiment_results.csv',
                                    encoding='utf_8_sig')
        all_experiment_results = pd.concat([all_experiment_results,self.metrics],axis=0)
        all_experiment_results.to_csv(general_parameters.project_dir + r'\experiment_output\all_experiment_results.csv',
                                    encoding='utf_8_sig',index=False)

        self.confusion_matrix.to_csv(
            general_parameters.project_dir + '\experiment_output\confusion_matrix' + '_' + self.version +
            '_' + self.model_name + '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv',
            encoding='utf_8_sig')

    def get_training_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        ax.plot(self.history['val_loss'][0:], marker='o', markersize=3, color='red')
        ax.plot(self.history['loss'][0:], marker='s', markersize=3, color='blue')
        # plt.title('model loss')
        plt.ylabel('损失函数值')
        plt.xlabel('时期')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.legend(loc='upper right', fontsize='medium')
        plt.show()

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\training_history', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png')
        plt.show()
class DTR_pipeline(ANN_pipeline):

    def change_data_block_shape(self):
        self.data_block.train[1] = tf.argmax(self.data_block.train[1], axis=1).numpy()
        self.data_block.val[1] = tf.argmax(self.data_block.val[1], axis=1).numpy()
        self.data_block.test[1] = tf.argmax(self.data_block.test[1], axis=1).numpy()
    def build_and_compile_model(self):
        from sklearn import tree
        self.model = tree.DecisionTreeRegressor()
    def fit(self):
        self.fit_start_time = process_time()
        self.model.fit(self.data_block.train[0], self.data_block.train[1])
        self.fit_end_time = process_time()
    def get_the_classification_results(self):
        self.output = self.model.predict(self.data_block.test[0]).tolist()
        self.output = [round(x) for x in self.output]
        self.label = self.data_block.test[1].tolist()

class AE_pipeline(ANN_pipeline):
    def build_and_compile_model(self):
        latent_dim = 2

        class Autoencoder(tf.keras.models.Model):
            def __init__(self, latent_dim, input_feature_list):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                self.encoder = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=32, activation='relu'),
                    tf.keras.layers.Dense(units=8, activation='relu'),
                    tf.keras.layers.Dense(units=2, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=8, activation='relu'),
                    tf.keras.layers.Dense(units=32, activation='relu'),
                    tf.keras.layers.Dense(units=len(input_feature_list), activation='relu'),
                ])

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        self.model = Autoencoder(latent_dim,self.input_feature_list)
        self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    def fit(self, epoch_num=10):
        project_dir, output_describe, model_name= \
            general_parameters.project_dir, r'\trained_model\check_point_for_temp_use_',self.model_name
        checkpoint_filepath = project_dir + output_describe + '_' + model_name
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.fit_start_time = process_time()
        self.history = self.model.fit(self.data_block.train[0],
                                      self.data_block.train[0],
                                      batch_size=100,
                                      epochs=epoch_num,
                                      #validation_data=self.data_block.val,
                                      callbacks=[model_checkpoint_callback, tensorboard_callback],
                                      shuffle=True
                                      )
        self.model.load_weights(checkpoint_filepath)
        self.fit_end_time = process_time()

    def save_model_and_history(self):
        self.model.save_weights(
            general_parameters.project_dir + r'\trained_model\model_' + '_' + self.version + '_' + self.model_name +
            '_'+ self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        pd.DataFrame.from_dict(self.history.history).to_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv',index=False)
        pd.DataFrame([self.fit_start_time, self.fit_end_time]).to_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv', index=False)

    def load_model_and_history(self):
        self.model.load_weights(
            general_parameters.project_dir + r'\trained_model\model_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.history = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.history.history = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\history_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.time_result = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.fit_start_time = self.time_result.iloc[0, 0]
        self.fit_end_time = self.time_result.iloc[1, 0]

class VAE_pipeline():
    def __init__(self, model_name, data_block, input_feature_list, scaled_feature,
                 data_type, resampling_type, resampling_ratio, version):
        self.model_name = model_name
        self.data_block = data_block
        self.input_feature_list = input_feature_list
        self.scaled_feature = scaled_feature
        self.data_type = data_type
        self.resampling_type = resampling_type
        self.resampling_ratio = resampling_ratio
        self.version = version

        self.output_numbers = np.linalg.matrix_rank(data_block.train[1])

    def build_model(self):
        latent_size = 3

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=2*latent_size, activation='relu'),
            tfpl.DistributionLambda(
                lambda t:tfd.MultivariateNormalDiag(
                    loc=t[..., :latent_size],
                    scale_diag=tf.math.exp(t[..., latent_size:])
                )
            )
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=2*len(self.input_feature_list), activation='relu'),
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=t[..., :len(self.input_feature_list)],
                    scale_diag=tf.math.exp(t[..., len(self.input_feature_list):])))
        ])

        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_size),scale_diag=tf.ones(latent_size)+10)

    def compile_and_fit_model(self, epoch_num=10):
        def loss(x, encoding_dist, sampled_decoding_dist, prior):
            return tf.reduce_mean(
                tfd.kl_divergence(encoding_dist, prior) - sampled_decoding_dist.log_prob(x)
            )

        self.fit_start_time = process_time()
        num_epochs = epoch_num
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        for i in range(num_epochs):
            for train_batch in np.expand_dims(np.array(self.data_block.train[0]),axis=0):
                with tf.GradientTape() as tape:
                    encoding_dist = self.encoder(train_batch)
                    sampled_z = encoding_dist.sample()
                    sampled_decoding_dist = self.decoder(sampled_z)
                    current_loss = loss(train_batch, encoding_dist, sampled_decoding_dist, self.prior)
                #print(current_loss.numpy())

                grads = tape.gradient(current_loss, self.encoder.trainable_variables +
                                      self.decoder.trainable_variables)
                #print(grads)
                opt.apply_gradients(zip(grads, self.encoder.trainable_variables
                                        + self.decoder.trainable_variables))

            print('-ELBO after epoch {}: {:.0f}'.format(i + 1, current_loss.numpy()))


        self.fit_end_time = process_time()

    def reconstruction(self):
        for i in range(3):
            reconstruction_sample = np.expand_dims(np.array(self.data_block.train[0].iloc[i]), axis=0)
            approx_posterior = self.encoder(reconstruction_sample)
            decoding_dist = self.decoder(approx_posterior.sample()).mean()

            fig, ax = plt.subplots(1, 1)
            ax.plot(reconstruction_sample[0])
            ax.plot(decoding_dist.numpy()[0])
            print(reconstruction_sample)
            print(decoding_dist.numpy())

    def generate_new_samples(self, num_samples=10):
        z = self.prior.sample(num_samples)
        self.new_samples = self.decoder(z).mean()
        fig, ax = plt.subplots(1, 1)
        for i in range(10):
            ax.plot(self.new_samples[i])

        fig, ax = plt.subplots(1,1)
        for i in range(10):
            ax.plot(np.array(self.data_block.train[0].iloc[i]))
        print(self.new_samples.numpy())

    def save_model_and_history(self):
        self.encoder.save_weights(
            general_parameters.project_dir + r'\trained_model\encoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.decoder.save_weights(
            general_parameters.project_dir + r'\trained_model\decoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        pd.DataFrame([self.fit_start_time, self.fit_end_time]).to_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv', index=False)

    def load_model_and_history(self):
        self.encoder.load_weights(
            general_parameters.project_dir + r'\trained_model\encoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.decoder.load_weights(
            general_parameters.project_dir + r'\trained_model\decoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.time_result = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.fit_start_time = self.time_result.iloc[0, 0]
        self.fit_end_time = self.time_result.iloc[1, 0]







class VAE_2_pipeline(ANN_pipeline):
    def build_model(self):
        def encoder_layers(inputs, latent_dim):
            x = tf.keras.layers.Conv1D(filters=32,kernel_size=3,strides=2,padding='same',activation='relu',name='encode_conv1')(inputs)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu',
                                       name='encode_conv1')(inputs)
            batch_2 = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Flatten(name='encode_flatten')(batch_2)
            x = tf.keras.layers.Dense(20,activation='relu',name='encode_dense')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            mu = tf.keras.layers.Dense(latent_dim,name='latent_mu')(x)
            sigma = tf.keras.layers.Dense(latent_dim, name='latent_sigma')(x)

            return mu, sigma, batch_2.shape

        class Sampling(tf.keras.layers.Layer):
            def call(self, inputs):
                mu, sigma = inputs
                batch = tf.shape(mu)[0]
                dim = tf.shape(mu)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return mu + tf.exp(0.5 * sigma) * epsilon

        def encoder_model(LATENT_DIM, input_shape):
            inputs = tf.keras.layers.Input(shape=input_shape)
            mu, sigma, conv_shape = encoder_layers(inputs, latent_dim=LATENT_DIM)
            z = Sampling()((mu,sigma))
            model = tf.keras.Model(inputs, outputs=[mu, sigma, z])
            return model, conv_shape


        latent_size = 2

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=2*latent_size, activation='relu'),
            tfpl.DistributionLambda(
                lambda t:tfd.MultivariateNormalDiag(
                    loc=t[..., :latent_size],
                    scale_diag=tf.math.exp(t[..., latent_size:])
                )
            )
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(units=8, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=2*len(self.input_feature_list), activation='relu'),
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=t[..., :len(self.input_feature_list)],
                    scale_diag=tf.math.exp(t[..., len(self.input_feature_list):])))
        ])

        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros(latent_size))

    def compile_and_fit_model(self, epoch_num=10):
        def loss(x, encoding_dist, sampled_decoding_dist, prior):
            return tf.reduce_sum(
                tfd.kl_divergence(encoding_dist, prior) - sampled_decoding_dist.log_prob(x)
            )

        self.fit_start_time = process_time()
        num_epochs = epoch_num
        opt = tf.keras.optimizers.Adam()
        for i in range(num_epochs):
            for train_batch in np.expand_dims(np.array(self.data_block.train[0]),axis=0):
                with tf.GradientTape() as tape:
                    encoding_dist = self.encoder(train_batch)
                    sampled_z = encoding_dist.sample()
                    sampled_decoding_dist = self.decoder(sampled_z)
                    current_loss = loss(train_batch, encoding_dist, sampled_decoding_dist, self.prior)

                grads = tape.gradient(current_loss, self.encoder.trainable_variables +
                                      self.decoder.trainable_variables)
                #print(grads)
                opt.apply_gradients(zip(grads, self.encoder.trainable_variables
                                        + self.decoder.trainable_variables))

            print('-ELBO after epoch {}: {:.0f}'.format(i + 1, current_loss.numpy()))
        self.fit_end_time = process_time()

    def reconstruction(self):
        reconstruction_sample = np.expand_dims(np.array(self.data_block.train[0].iloc[30]),axis=0)
        approx_posterior = self.encoder(reconstruction_sample)
        decoding_dist = self.decoder(approx_posterior.sample()).mean()

        fig, ax = plt.subplots(1,1)
        ax.plot(reconstruction_sample,labels='reconstruction')
        ax.plot(decoding_dist.numpy(), labels='reconstruction')
        print(reconstruction_sample)
        print(decoding_dist.numpy())

    def generate_new_samples(self,num_samples=6):
        z = self.prior.sample(num_samples)
        self.new_samples = self.decoder(z).mean()
        print(self.new_samples.numpy())


    def save_model_and_history(self):
        self.encoder.save_weights(
            general_parameters.project_dir + r'\trained_model\encoder_' + '_' + self.version + '_' + self.model_name +
            '_'+ self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.decoder.save_weights(
            general_parameters.project_dir + r'\trained_model\decoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        pd.DataFrame([self.fit_start_time, self.fit_end_time]).to_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv', index=False)

    def load_model_and_history(self):
        self.encoder.load_weights(
            general_parameters.project_dir + r'\trained_model\encoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.decoder.load_weights(
            general_parameters.project_dir + r'\trained_model\decoder_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio))
        self.time_result = pd.read_csv(
            general_parameters.project_dir + r'\trained_model\training_time_' + '_' + self.version + '_' + self.model_name +
            '_' + self.data_type + '_' + self.resampling_type + '_' + str(self.resampling_ratio) + '.csv')
        self.fit_start_time = self.time_result.iloc[0, 0]
     