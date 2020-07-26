import os
import sys
import math
import glob
import wave
import shutil
import IPython
import numpy as np
import pandas as pd
import multiprocessing
from random import randint
import matplotlib.image as mpimg
from itertools import zip_longest
from datetime import datetime, time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score

import requests as r
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, LSTM, Conv1D

from models import *
from classes import *

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

data_dir = './output_uni'
# data_dir = './new_data'

# data_dir = './mmsad/'
pos_train_paths, neg_train_paths,\
    pos_val_paths, neg_val_paths, sh_paths = DataGenerator.split_paths(
        data_dir)

print("lenght of positive paths = ", len(pos_train_paths))
print("lenght of negative paths = ", len(neg_train_paths))
print("lenght of positive validation paths = ", len(pos_val_paths))
print("lenght of negative validation paths = ", len(neg_val_paths))
print("lenght of shaikh paths = ", len(sh_paths))


all_pos = list(pos_train_paths) + list(pos_val_paths)
all_neg = list(neg_train_paths) + list(neg_val_paths)


sh_test_batch = np.expand_dims(
    get_clip('./new_data/shaikh/Husary_128kbps.111003.wav'), axis=0)

pp = glob.glob('./tb/*')


es = EarlyStopping(monitor='val_accuracy', mode='max',
                   patience=10, restore_best_weights=True)


Loss = contrastive_loss
input_shape = (20, 250)
optimizer3 = Adam(lr=.001, amsgrad=True)
dropout = .1
reg_parameter = 1e-3
kernel_size = 7
filt = 4096
lstm_cell = 256
dense_cell = 256
epochs = 100
cnt = 0
kf = KFold(n_splits=10)


def train(lstm_cell, dense_cell, cnt):
    def submodel():
        input = Input(shape=input_shape)

        x = Conv1D(filters=filt, kernel_size=kernel_size, kernel_regularizer=l2(reg_parameter),
                   bias_regularizer=l2(reg_parameter), activation='relu')(input)

        x = LSTM(lstm_cell,  kernel_regularizer=l2(reg_parameter),
                 bias_regularizer=l2(reg_parameter), return_sequences=True,)(x)

        x = LSTM(lstm_cell,  kernel_regularizer=l2(reg_parameter),
                 bias_regularizer=l2(reg_parameter), return_sequences=True,)(x)

        x = Dropout(dropout)(x)
        x = Flatten()(x)

        x = Dense(dense_cell,
                  kernel_regularizer=l2(reg_parameter),
                  bias_regularizer=l2(reg_parameter))(x)

        return Model(input, x)

    model = get_model(submodel, input_shape)
    BASE_NAME = "2_lstm"

    splitted_name = BASE_NAME.split('__')[0]
    for i in pp:
        if splitted_name in (i.split('__')[0].split("\\")[1]):
            break
    else:
        print(BASE_NAME)
        ii = 1
        f1_scores_val = []
        loss = []
        val_loss = []
        train_accuracy = []
        val_accuracy = []
        for train_index, test_index in kf.split(np.arange(min(len(all_pos), len(all_neg)))):
            pos_train_paths, pos_val_paths = np.array(all_pos)[train_index],\
                np.array(all_pos)[test_index]
            neg_train_paths, neg_val_paths = np.array(all_neg)[train_index],\
                np.array(all_neg)[test_index]

            train_generator = DataGenerator(sh_paths, pos_train_paths, neg_train_paths, 0,
                                            [1, 2], combine_augmentation=False, train=True)
            val_generator = DataGenerator(sh_paths, pos_val_paths, neg_val_paths, 0,
                                          [], combine_augmentation=False, train=False)

            model.compile(
                loss=Loss, optimizer=optimizer3, metrics=[accuracy])

            NAME = str(ii) + BASE_NAME
            print('-' * 100)
            print(NAME)
            print('-' * 100)

            # logdir = os.path.join("new_logs", "{}".format(NAME))
            # tensorboard = tf.keras.callbacks.TensorBoard(
            #     logdir, histogram_freq=1)

            history = model.fit(train_generator,
                                validation_data=val_generator,
                                batch_size=16,
                                callbacks=[
                                    # tensorboard,
                                    es
                                ],
                                workers=16,
                                shuffle=True,
                                epochs=epochs)

            print_preds(model, sh_test_batch, sh_paths, [], 'Sh')
            _, __ = print_preds(model, sh_test_batch, pos_train_paths,
                                neg_train_paths, 'Train')
            preds_pos, preds_neg = print_preds(model, sh_test_batch,
                                               pos_val_paths, neg_val_paths, 'Val')

            x = list(preds_pos) + list(preds_neg)
            y = [1 for i in range(len(pos_val_paths))] + \
                [0 for i in range(len(pos_val_paths))]

            print(history.history)

            f1 = f1_score(x, y)
            print(f1)
            f1_scores_val.append(f1)
            loss.append(history.history['loss'])
            val_loss.append(history.history['val_loss'])
            train_accuracy.append(history.history['accuracy'])
            val_accuracy.append(history.history['val_accuracy'])
            ii += 1

            tf.keras.backend.clear_session()

        cnt += 1

        logdir = "./tb/" + NAME
        writer = tf.summary.create_file_writer(logdir)

        f1_scalar = sum(f1_scores_val) / len(f1_scores_val)

        best_train_acc = list(map(lambda x: max(x), train_accuracy))
        accuracy_scalar = sum(best_train_acc) / len(best_train_acc)

        best_val_acc = list(map(lambda x: max(x), val_accuracy))
        val_accuracy_scalar = sum(best_val_acc) / len(best_val_acc)

        stripped = ([y for y in suby if y is not None]
                    for suby in zip_longest(*loss))
        loss_scalar = list(map(lambda x: sum(x)/len(x), stripped))

        stripped_val = ([y for y in suby if y is not None]
                        for suby in zip_longest(*val_loss))
        val_loss_scalar = list(map(lambda x: sum(x)/len(x), stripped_val))

        print('-' * 100)
        print('F1:')
        print('-' * 100)
        print(f1_scores_val)
        print('-' * 100)
        print(f1_scalar)

        print('-' * 100)
        print("Train Loss:")
        print('-' * 100)
        print(loss_scalar)
        print('-' * 100)
        print("Val Loss:")
        print('-' * 100)
        print(val_loss_scalar)

        print('-' * 100)
        print("Train Accuracy:")
        print(best_train_acc)
        print('-' * 100)
        print(accuracy_scalar)
        print('-' * 100)

        print('-' * 100)
        print("Val Accuracy:")
        print(best_val_acc)
        print('-' * 100)
        print(val_accuracy_scalar)
        print('-' * 100)

        with writer.as_default():
            tf.summary.scalar('f1', f1_scalar, step=cnt)
            tf.summary.scalar('Mean train accuracy',
                              accuracy_scalar, step=cnt)
            tf.summary.scalar('Mean val accuracy',
                              val_accuracy_scalar, step=cnt)

            for step in range(len(loss_scalar)):
                tf.summary.scalar('loss', loss_scalar[step], step=step)
                tf.summary.scalar(
                    'val_loss', val_loss_scalar[step], step=step)

            for step in range(len(best_train_acc)):
                tf.summary.scalar('train_accuracy',
                                  best_train_acc[step], step=step)
                tf.summary.scalar(
                    'val_accuracy', best_val_acc[step], step=step)


if __name__ == "__main__":
    p = multiprocessing.Process(
        target=train, args=(lstm_cell, dense_cell, cnt))
    p.start()
    p.join()
