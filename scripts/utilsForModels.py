from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import math
from sklearn.metrics import confusion_matrix
from pathlib import Path


def get_data_generators(bitmaps_dir, batch_size = 64, val_split = .20):

  image_shape = (224,224)

  datagen_kwargs = dict(rescale=1./255, validation_split=val_split)
  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

  test_gen = valid_datagen.flow_from_directory(
      bitmaps_dir, 
      subset="validation", 
      color_mode="grayscale",
      batch_size=batch_size,
      class_mode='binary',
      shuffle=True,
      target_size=image_shape
  )

  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

  train_gen = train_datagen.flow_from_directory(
      bitmaps_dir, 
      subset="training", 
      color_mode="grayscale",
      batch_size=batch_size,
      class_mode='binary',
      shuffle=True,
      target_size=image_shape)

  # print some info about data
  for image_batch, label_batch in train_gen:
    break
  print(image_batch.shape) 
  print(label_batch.shape)
  
  return train_gen, test_gen


def create_model_feedforward():
  model = models.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),  
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
  ])
  return model



def create_model():  
  # simple CNN model
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 1))) #  Since your image_ordering_dim is set to "tf", if you reshape your data to the tensorflow style it should work: IMG_W, IMG_H, colorchanel
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(32, (3, 3)))
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2)))

  model.add(layers.Conv2D(64, (3, 3)))
  model.add(layers.Activation('relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2))),

  model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
  model.add(layers.Dense(64))
  model.add(layers.Activation('relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1))
  model.add(layers.Activation('sigmoid'))
  # simple CNN model
  # model = models.Sequential()
  # model.add(layers.Conv2D(32, (2, 2), input_shape=(224, 224, 1))) #  Since your image_ordering_dim is set to "tf", if you reshape your data to the tensorflow style it should work: IMG_W, IMG_H, colorchanel
  # model.add(layers.Dense(1))
  # model.add(layers.Activation('sigmoid'))

  model.compile(
      optimizer=tf.keras.optimizers.Adagrad(),
      loss=losses.BinaryCrossentropy(),
      metrics=['accuracy']
  ) 

  return model



def train_model(model, saveName, train_gen, test_gen, batch_size = 64):

  print("Fit model on training data")
  model.fit_generator(
        train_gen,
        steps_per_epoch=2000 // batch_size,
        epochs=5,
        validation_data=test_gen,
        validation_steps=800 // batch_size)

  #Path('trainedWeights/'+saveName).mkdir(parents=True, exist_ok=True)
  #model.save_weights('trainedWeights/'+saveName)  # always save your weights after training or during training



def evaluate_the_model(model, test_dir):

  image_shape = (224, 224)
  datagen_kwargs = dict(rescale=1./255, validation_split=.30)
  valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
  validation_generator = valid_datagen.flow_from_directory(
        test_dir, 
        subset="validation", 
        color_mode="grayscale",
        batch_size=1,  # change batch size to 1
        class_mode='binary',
        shuffle=True,
        target_size=image_shape
  )

  filenames = validation_generator.filenames
  nb_samples = len(filenames)

  # https://stackoverflow.com/questions/45806669/how-to-use-predict-generator-with-imagedatagenerator
  Y_pred = model.predict_generator(validation_generator, steps=nb_samples)
  y_pred = np.argmax(Y_pred, axis=1)
  print('Confusion Matrix')
  labels = ['Flow', 'No flow']
  #Confution Matrix and Classification Report
  cm = confusion_matrix(validation_generator.classes, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

  disp.plot(cmap=plt.cm.Blues)
  plt.show()

  print('Classification Report')
  print(tf.keras.metrics.classification_report(validation_generator.classes, y_pred, target_names=labels))



def split_corelated_csv_for_classes(corelated_csv_dir):
  df = pd.read_csv(corelated_csv_dir)  
  
  df_0 = df[df['flow'] <= 0]
  df_1 = df[df['flow'] == 1]

  savePath = '/'.join(corelated_csv_dir.split("/")[:-1])
  print("save splitted csv in classes to: ", savePath)

  df_0.to_csv(savePath+'/_noflow.csv', index=False)  
  df_1.to_csv(savePath+'/_flow.csv', index=False)
  
  return savePath+'/_noflow.csv', savePath+'/_flow.csv'



def get_bitmaps_from_csvs(flow_file_dir, noflow_file_dir):
  IMG_WIDTH_HEIGHT = 224
  image_scale = round(IMG_WIDTH_HEIGHT / 2.1333)
  reshapeObj = Rescale(output_size=image_scale, round_values_to=0, bit_map_normalize=True, img_h=IMG_WIDTH_HEIGHT, img_w=IMG_WIDTH_HEIGHT)
  df_0 = pd.read_csv(noflow_file_dir)
  df_1 = pd.read_csv(flow_file_dir)
  savePathFlow = '/'.join(flow_file_dir.split("/")[:-1])+'/bitmaps/flow'
  savePathNoflow = '/'.join(flow_file_dir.split("/")[:-1])+'/bitmaps/noflow'
  # Utwórz ścieżki
  Path(savePathFlow).mkdir(parents=True, exist_ok=True)
  Path(savePathNoflow).mkdir(parents=True, exist_ok=True)
  
  print("Generating bitmaps (noflow)...")
  row = 1 # 0 row is header, so I start from 1
  while row < len(df_0):
    coords = df_0.iloc[row, :]
    coords = np.array(coords)
    # Taki slice array, bo pierwsza kolumna to  index unnamed, a ostatnie 3 to time_ms, flow, time_precision
    coords = coords[1:-3]
    coords = coords.astype('float').reshape(-1,2)
    sample =  {'inputs': coords, 'targets': 0} # taki już wymusiłem obiekt wejściowy, więc niech sobie jest...
    sample = reshapeObj(sample)

    bitmap = sample['inputs']
    #zapisz bitmape do pliku...
    image = Image.fromarray(bitmap)
    if image.mode != 'L':
      image = image.convert('L')
    image.save(savePathNoflow+'/'+str(row)+'.bmp', format='bmp')
    row+=1

  print("Generating bitmaps (flow)...")
  row = 1 # Pierwszy wiersz to nagłówki, więc zaczynam od 1, a nie od 0
  while row < len(df_1):
    coords = df_0.iloc[row, :]
    coords = np.array(coords)
    # Taki slice array, bo pierwsza kolumna to  index unnamed, a ostatnie 3 to time_ms, flow, time_precision
    coords = coords[1:-3]
    coords = coords.astype('float').reshape(-1, 2)
    sample =  {'inputs': coords, 'targets': 0} # taki już wymusiłem obiekt wejściowy, więc niech sobie jest...
    sample = reshapeObj(sample)
    bitmap = sample['inputs']
    
    #zapisz bitmape do pliku...
    image = Image.fromarray(bitmap)
    if image.mode != 'L':
      image = image.convert('L')
    image.save(savePathFlow+'/'+str(row)+'.bmp', format='bmp')
    row+=1

  return 0  





class Rescale(object):

  def __init__(self, output_size, round_values_to = -1, bit_map_normalize = True, img_h=50, img_w=50):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size
    self.round_values_to = round_values_to
    self.bit_map_normalize = bit_map_normalize
    self.width_height_for_emtpty = img_h # or img_w => not matter since I use both equal
    self.img_w = img_w
    self.img_h = img_h

  def __bitmap_normalizse(self, max_w, max_h, landmarks):
    df = pd.DataFrame(np.zeros((int(max_w), int(max_h))))
    for land in landmarks:
      df[land[0]][land[1]] = 255
    return df.to_numpy()

  def __custom_image_size_normalization(self, inputs):
    # add empty row at the bottom...
    if (inputs.shape[0] < self.img_h):
      add_row_count = self.img_h - inputs.shape[0]
      inputs = np.pad(inputs, ((0, add_row_count), (0,0)), 'constant')

    # slice array - from bottom
    elif (inputs.shape[0] > self.img_h):
      inputs = inputs[:self.img_h , :]
      
    # add empty COLUMN from the LEFT side of array (left, beacause zeros are more common in that area)      
    if (inputs.shape[1] < self.img_w):
      add_col_count = self.img_w - inputs.shape[1]
      inputs = np.pad(inputs, ((0, 0), (add_col_count,0)), 'constant')

    # slice array - from left side 
    elif (inputs.shape[1] > self.img_w):
      slice_index_col = inputs.shape[1] - self.img_w
      inputs = inputs[: , slice_index_col:]

    return inputs    


  def __call__(self, sample):
      
    landmarks, targets = sample['inputs'], sample['targets']

    # When face was not detected, them set all landmarks to zeros, and target = 0
    array_sum = np.sum(landmarks)
    array_has_nan = np.isnan(array_sum)
    if (array_has_nan):
      landmarks = np.zeros((self.width_height_for_emtpty, self.width_height_for_emtpty))
      targets = 0
      return {'inputs': landmarks, 'targets': targets}

    # find max "Width" and "Height" in landmarks
    if landmarks.size != 0:
      w = np.max(landmarks[:][0])
      h = np.max(landmarks[:][1])
    else:
      w = self.img_w  # eventually just skip iteration when empty
      h = self.img_h

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size
    new_h, new_w = int(new_h), int(new_w)

    # h and w are swapped for landmarks because for images,
    # x and y axes are axis 1 and 0 respectively
    
    landmarks = landmarks * [new_w / w, new_h / h]
    if (self.round_values_to > -1):
      landmarks = np.around(landmarks, self.round_values_to)

    # NORMALIZE INTO "BIT MAP"
    if (self.bit_map_normalize):
      if landmarks.size != 0:
        new_max_w = np.max(landmarks) + 1
        new_max_h = np.max(landmarks) + 1
      else:
        new_max_w = self.img_w
        new_max_h = self.img_h
      landmarks = self.__bitmap_normalizse(new_max_w, new_max_h, landmarks)

    landmarks = self.__custom_image_size_normalization(landmarks)

    return {'inputs': landmarks, 'targets': targets}
