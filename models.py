from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, f1_score
import librosa

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, LSTM, Conv1D

threshold = 0.5
max_pad_len = 200


def get_f1(preds_pos, preds_neg):
    x = list(preds_pos) + list(preds_neg)
    y = [1 for i in range(len(preds_pos))] + \
        [0 for i in range(len(preds_neg))]

    return f1_score(x, y)


def get_clip(file_path):
    wave, sr = librosa.load(file_path, mono=True)
    mfcc = librosa.feature.mfcc(np.asfortranarray(wave), sr=sr,
                                n_mfcc=20)
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    mfccx = max(mfcc.min(), mfcc.max(), key=abs)
    mfcc = mfcc/mfccx
    return mfcc


def print_preds(model, sh_test_batch, pos_paths, neg_paths, clip_type):
    pos_clips = get_clips(pos_paths)
    y_ones = [1 for i in range(len(pos_clips))]
    preds_pos, actual_pos = get_predictions(model,
                                            pos_clips, y_ones,
                                            sh_test_batch)
    print('-' * 100)
    print('Positive {}: '.format(clip_type))
    print('-' * 100)
    print(preds_pos)
    print(actual_pos)
    print('-' * 100)

    if clip_type != 'Sh':
        neg_clips = get_clips(neg_paths)
        y_zeros = [0 for i in range(len(neg_clips))]
        preds_neg, actual_neg = get_predictions(model,
                                                neg_clips, y_zeros,
                                                sh_test_batch)

        print('Negative {}: '.format(clip_type))
        print('-' * 100)
        print(preds_neg)
        print(actual_neg)
        print('-' * 100)

        return preds_pos, preds_neg


def get_clips(clips_paths):
    clips = []
    for path in clips_paths:
        clips.append(get_clip(path))
    return np.array(clips)


def euclidean_distance(vects):
    '''
        https://en.wikipedia.org/wiki/Euclidean_distance
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''
        Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = .75
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''
        Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


def classify(y_true, y_pred):
    prediction = y_pred.ravel()
    # print(prediction)
    if prediction < threshold:
        return 1
    return 0


def submodel(input_shape=(20, max_pad_len)):
    input = Input(shape=input_shape)

    x = Conv1D(filters=4096, kernel_size=7, kernel_regularizer=l2(1e-3),
               bias_regularizer=l2(1e-3), activation='relu')(input)

    x = LSTM(256,  kernel_regularizer=l2(1e-3),
             bias_regularizer=l2(1e-3), return_sequences=True,)(x)

    x = Dropout(.1)(x)
    x = Flatten()(x)

    x = Dense(256,
              kernel_regularizer=l2(1e-3),
              bias_regularizer=l2(1e-3))(x)

    return Model(input, x)


def get_model(base_network, input_shape=(20, max_pad_len)):
    base_network = base_network()

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [processed_a, processed_b])
    model = Model([input_a, input_b], distance)
    return model


def test_with_all_clips(model, test_clip, sh_clips):
    pred = 0
    for clip in sh_clips:
        true_clip = np.expand_dims(clip, axis=0)
        single_pred = model.predict(
            [true_clip, np.expand_dims(test_clip, axis=0)])
        pred += single_pred
        print(single_pred)
    return pred/len(sh_clips)


def get_predictions(model, X_test, y_test, sh_test_batch, test_with_all=False, sh_clips=[]):
    preds = np.zeros((0, 1))
    acutual_preds = np.zeros((0, 1))

    for ii in range(len(y_test)):
        if (test_with_all):
            pred = test_with_all_clips(model, X_test[ii, :], sh_clips)
        else:
            pred = model.predict(
                [sh_test_batch, np.expand_dims(X_test[ii, :], axis=0)])
        print('-' * 100)
        print(pred)
        print('-' * 100)
        preds = np.vstack((preds, classify(y_test[ii], pred)))
        acutual_preds = np.vstack((acutual_preds, pred))
    return preds.flatten(), np.around(acutual_preds.flatten(), 3)


def print_statistics(cm):
    tn, fp, fn, tp = cm.ravel()
    # TP
    print("TP: " + str(tp))
    # TN
    print("TN: " + str(tn))
    # FP
    print("FP: " + str(fp))
    # FN
    print("FN: " + str(fn))
    # TPR
    recall = tp/(tp+fn)
    print("TPR/recall: " + str(recall))
    # TNR
    specificity = tn/(tn+fp)
    print("TNR/specificity: " + str(specificity))
    # PPV
    precision = tp/(tp+fp)
    print("PPV/precision: " + str(precision))
    # NPV
    npv = tn/(tn+fn)
    print("NPV/negative predictive value: " + str(npv))
    # FNR
    miss_rate = 1-recall
    print("FNR/false negative rate: " + str(miss_rate))
    # FPR
    fall_out = 1-specificity
    print("FPR/false positive rate: " + str(fall_out))
    # FDR
    fdr = 1-precision
    print("FDR/false discovery rate: " + str(fdr))
    # FOR
    fomr = 1-npv
    print("FOR/false ommission rate: " + str(fomr))
    # F1
    f1 = 2*((precision*recall)/(precision+recall))
    print("F1 score: " + str(f1))
    # accuracy
    acc = (tp+tn)/(tp+tn+fp+fn)
    print("Accuracy: " + str(acc))
    # Matthews correlation coefficient (MCC)
    mcc = (tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    print("MCC/Matthews correlation coefficient: " + str(mcc))
    # Informedness or Bookmaker Informedness (BM)
    bm = recall+specificity-1
    print("BM/Bookmaker Informedness: " + str(bm))
    # Markedness (MK)
    mk = precision+npv-1
    print("MK/Markedness: " + str(mk))

    return fall_out, recall


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    s = [['TN', 'FP'], ['FN', 'TP']]
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, (str(s[i][j])+" = "+str(format(cm[i][j], fmt))), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_confusion_matrix(trues, preds):
    cm = confusion_matrix(trues, preds)
    fpr, tpr = print_statistics(cm)
    plt.figure()
    plot_confusion_matrix(
        cm, classes=['Negative', 'Positive'], title='Confusion matrix')
    plt.show()
