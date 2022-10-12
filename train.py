#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import os

def get_label(file_path, dpos, labels):
    file_name=tf.strings.split(file_path, os.path.sep)[-2]
    digit=tf.strings.substr(file_name, dpos,1)
    return digit==labels

def process_path(file_path, dpos, labels):
    label = get_label(file_path, dpos, labels)
    img=load_image(file_path)
    return img, label

def process_test(file_path):
    img=load_image(file_path)
    file_name=tf.strings.split(file_path, os.path.sep)[-1]
    datetimestr=tf.strings.substr(file_name, 3, 14)
    return img, datetimestr

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(min(25,len(image_batch))):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

def train_model_for_digit(globpath, digit, labels, epochs=40):
    ds_load  = tf.data.Dataset.list_files(globpath)
    ds_label = ds_load .map(lambda x: process_path(x, digit, labels), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_batch = ds_label.shuffle(1000).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(66,92,3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(labels))
    ])
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(ds_batch, epochs=epochs)
    return model


#
# Load data

models=train_model_for_digit('train/*/*.jpg', 0, ['m','p'                                ], epochs=10)
model1=train_model_for_digit('train/*/*.jpg', 1, ['0','1','2','3','4','5','6','7','8','9'], epochs=10)
model2=train_model_for_digit('train/*/*.jpg', 2, ['0','1','2','3','4','5','6','7','8','9'], epochs=25)
model3=train_model_for_digit('train/*/*.jpg', 4, ['0','1','2','3','4','5','6','7','8','9'], epochs=30)

#
# Test!

ds_test  = tf.data.Dataset.list_files('test/*.jpg')
ds_test_img   = ds_test.map(process_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test_batch = ds_test_img.shuffle(1000).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Add more trainings
image_batch, datetime_batch = next(iter(ds_test_batch))

predicts_batch = np.argmax(models.predict(image_batch), axis=1)
predict1_batch = np.argmax(model1.predict(image_batch), axis=1)
predict2_batch = np.argmax(model2.predict(image_batch), axis=1)
predict3_batch = np.argmax(model3.predict(image_batch), axis=1)
predict_batch=(predict1_batch*10+predict2_batch+predict3_batch/10)*(np.array([-1,+1])[predicts_batch])

image_batch   =image_batch   .numpy()
datetime_batch=datetime_batch.numpy()

for i in range(len(image_batch)):
    plt.clf()
    plt.imshow(image_batch[i])
    plt.title(predict_batch[i])
    plt.axis('off')
    plt.draw()
    plt.show(False)

    print('Correct classification?')
    x=input()
    if x=='':
        continue
    if x=='x':
        break

    x=float(x)
    x='{}{:04.1f}'.format('p' if x>0 else 'm',abs(x))
    print(x)

    dirtrainname='train/{}'.format(x)
    if not os.path.exists(dirtrainname):
        os.mkdir(dirtrainname)

    fname='01-{}-snapshot.jpg'.format(datetime_batch[i].decode('utf-8'))
    pathtrainname='{}/{}'.format(dirtrainname,fname)
    if os.path.exists(pathtrainname):
        print('Already training on this...')
        continue

    os.symlink(os.path.abspath('images/{}'.format(fname)), pathtrainname)
plt.close()

show_batch(image_batch, predict_batch)
plt.savefig('test.pdf')

# Make plot
for image_batch, datetime_batch in iter(ds_test_batch):
    datetimestr = [datetime.datetime.strptime(x.decode('utf-8'), '%Y%m%d%H%M%S') for x in datetime_batch.numpy()]

    predicts_batch = np.argmax(models.predict(image_batch), axis=1)    
    predict1_batch = np.argmax(model1.predict(image_batch), axis=1)
    predict2_batch = np.argmax(model2.predict(image_batch), axis=1)
    predict3_batch = np.argmax(model3.predict(image_batch), axis=1)
    predict_batch=(predict1_batch*10+predict2_batch+predict3_batch/10)*(np.array([-1,+1])[predicts_batch])    

    plt.plot(datetimestr, predict_batch,'.')
plt.show()
    
