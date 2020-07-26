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


def load_data(data_dir):
    pos_train_paths, neg_train_paths,\
        pos_val_paths, neg_val_paths, sh_paths = DataGenerator.split_paths(
            data_dir)

    all_pos = list(pos_train_paths) + list(pos_val_paths)
    all_neg = list(neg_train_paths) + list(neg_val_paths)

    print("length of shaikh paths = ", len(sh_paths))
    print("length of positive paths = ", len(all_pos))
    print("length of negative paths = ", len(all_neg))

    train_generator = DataGenerator(sh_paths, all_pos, all_neg, 0,
                                    [1, 2], combine_augmentation=False, train=True, max_pad_len=200)
    return train_generator, sh_paths, all_pos, all_neg


def train(train_generator, sh_paths, all_pos, all_neg, sh_test_batch):
    checkpoint_dir = './last_checkpoint_narab'
    checkpoint_prefix = os.path.join(
        checkpoint_dir, 'last_200_' + "_ckpt_{epoch}")
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix, verbose=1, save_best_only=True, save_weights_only=True, monitor='loss')

    model = get_model(submodel)
    model.compile(
        loss=contrastive_loss, optimizer=Adam(lr=.001, amsgrad=True), metrics=[accuracy])

    tensorboard = tf.keras.callbacks.TensorBoard(
        "last_model_naran", histogram_freq=1)

    history = model.fit(train_generator,
                        batch_size=16,
                        callbacks=[tensorboard, checkpoint_callback],
                        workers=16,
                        shuffle=True,
                        epochs=100)

    print_preds(model, sh_test_batch, sh_paths, [], 'Sh')
    preds_pos, preds_neg = print_preds(model, sh_test_batch, all_pos,
                                       all_neg, 'Train')

    f1 = get_f1(preds_pos, preds_neg)
    print(history.history)
    print(f1)


def main():
    sh_test_batch = np.expand_dims(
        get_clip('./new_data/shaikh/Husary_128kbps.111003.wav'), axis=0)

    train_generator, sh_paths, all_pos, all_neg = load_data(
        data_dir='./new_data')

    train(train_generator, sh_paths, all_pos, all_neg, sh_test_batch)


main()
