from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--train", dest="TRAIN_FILE", required=True,
                    help="Input file path for training", metavar="FILE")
parser.add_argument("--test", dest="TEST_FILE", required=True,
                    help="Input file path for testing", metavar="FILE")
parser.add_argument("--out", dest="OUT_PATH", default="output",
                    help="Output file path", metavar="FILE")

parser.add_argument("--max_len", dest="MAX_LENGTH", type=int, required=True,
                    help="Maximum length for training/testing dataset")

parser.add_argument("--nfeature", dest="NUM_FEATURE", type=int, default=20,
                    help="Number of features. Default value is 20")

parser.add_argument("--batch", dest="BATCH_SIZE", type=int, default=32,
                    help="Batch size for each training process. Default value is 32")

parser.add_argument("--epoch", dest="EPOCHS", type=int, default=100,
                    help="Number of epoches. Default value is 100")

parser.add_argument("--wsizes", dest="WINDOW_SIZES", nargs='+', type=int, default=[8,16,24,32,40,48],
                    help="Kernel sizes, input array. E.g.: --wsizes 8 16 24 32 40 48")

parser.add_argument("--nfilter", dest="NUM_FILTER", default=256, type=int,
                    help="Number of filters for each evolution layer. Default value is 256")
parser.add_argument("--nhidden", dest="NUM_HIDDEN", default=1024, type=int,
                    help="Number of nodes for fully connected layer. Default value is 1024")
parser.add_argument("--nclass", dest="NUM_CLASS", default=2, type=int,
                    help="Number of classes. Default value is 2")

args = parser.parse_args()

import csv
import pandas as pd
import numpy as np

def load_ds(file_path):
  NUM_SAMPLES = 0
  with open(file_path) as file:
    NUM_SAMPLES = sum(1 for row in file)

  data = np.zeros((NUM_SAMPLES, args.MAX_LENGTH * args.NUM_FEATURE), dtype=np.float32 )
  labels = np.zeros((NUM_SAMPLES, 1), dtype=np.uint8 )

  with open(file_path) as file:
    file = csv.reader(file, delimiter = ',')
    m = 0
    for row in file:
      labels[m] = int(row[0])
      data[m] = np.array(row[1:]).astype('float32')
      m += 1
      print(f"\rReading {file_path}...\t{m}/{NUM_SAMPLES}", end='')
  print('\tDone')
  return data, labels


x_train, y_train = load_ds(args.TRAIN_FILE)
x_test, y_test = load_ds(args.TEST_FILE)

# Add a channels dimension
x_train = np.reshape( x_train, [-1,1, args.MAX_LENGTH, args.NUM_FEATURE] )
x_test = np.reshape( x_test, [-1,1, args.MAX_LENGTH, args.NUM_FEATURE] )

print(f"Train shape: {x_train.shape}")
print(f"Test shape: {x_test.shape}")

print(f"Train label shape: {y_train.shape}")
print(f"Test label shape: {y_test.shape}")

# Convert to categorical labels
import tensorflow as tf
y_train = tf.keras.utils.to_categorical(y_train,args.NUM_CLASS)
y_test = tf.keras.utils.to_categorical(y_test,args.NUM_CLASS)

import os
os.mkdir(args.OUT_PATH)

import os
import tensorflow as tf
import math

from sklearn import metrics
from sklearn.metrics import roc_curve
from tensorflow.keras import Model, layers
def val_binary(epoch,logs):

  pred = model.predict(x_test)

  fpr, tpr, thresholds = roc_curve(y_test[:,1], pred[:, 1])
  # calculate the g-mean for each threshold
  gmeans = np.sqrt(tpr * (1-fpr))
  # locate the index of the largest g-mean
  ix = np.argmax(gmeans)
  print(f'Best Threshold={thresholds[ix]}, G-Mean={gmeans[ix]}')
  threshold = thresholds[ix]

  y_pred = (pred[:, 1] >= threshold).astype(int)

  fout = open(f'{args.OUT_PATH}/training.csv','a')
  
  TN, FP, FN, TP =  metrics.confusion_matrix(y_test[:,1], y_pred).ravel()

  Sens = TP/(TP+FN) if TP+FN > 0 else 0.0
  Spec = TN/(FP+TN) if FP+TN > 0 else 0.0
  Acc = (TP+TN)/(TP+FP+TN+FN)
  MCC = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) if TP+FP > 0 and FP+TN > 0 and TP+FN and TN+FN else 0.0
  F1 = 2*TP/(2*TP+FP+FN)
  print(f'{epoch + 1},TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}\n')
  fout.write(f'{epoch + 1},{TP},{FP},{TN},{FN},{Sens:.4f},{Spec:.4f},{Acc:.4f},{MCC:.4f}\n')
  fout.close()


from model import mCNN

model = mCNN(
    num_filters=args.NUM_FILTER,
    num_hidden=args.NUM_HIDDEN,
    window_sizes=args.WINDOW_SIZES)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(1, 1,args.MAX_LENGTH, args.NUM_FEATURE))
model.summary()

model.fit(
    x_train,
    y_train,
    batch_size=args.BATCH_SIZE,
    epochs=args.EPOCHS,
    validation_data=(x_test, y_test),
    callbacks=[
      tf.keras.callbacks.LambdaCallback(on_epoch_end=val_binary),
      tf.keras.callbacks.ModelCheckpoint(args.OUT_PATH + '/weights.{epoch:02d}', save_weights_only=True),
      tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ]
)