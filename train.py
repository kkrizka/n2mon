#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
import os
import hashlib

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_label(file_path):
    dir_name=tf.strings.split(file_path, os.path.sep)[-2]
    number=tf.strings.to_number(dir_name)
    return number

def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_path(file_path):
    label = get_label(file_path)
    img = load_image(file_path)
    return img, label

def process_test(file_path):
    img=load_image(file_path)
    return img, file_path

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(min(25,len(image_batch))):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(label_batch[n])
        plt.axis('off')
    plt.show()

def train_model(globpath, epochs=40, checkpoint='chk'):
    ds_load  = tf.data.Dataset.list_files(globpath)
    ds_label = ds_load .map(lambda x: process_path(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_batch = ds_label.shuffle(1000).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(240,320,3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(10),
        #tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    if os.path.exists(f'{checkpoint}.index'):
        model.load_weights(checkpoint)

    # Run training
    model.fit(ds_batch, epochs=epochs)
    model.save_weights(checkpoint)

    return model


#
# Load data

model=train_model('train/*/*.jpg', epochs=10)

model.save('mymodel')

#
# Test!

ds_test  = tf.data.Dataset.list_files('test/*.jpg')
ds_test_img   = ds_test.map(process_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test_batch = ds_test_img.shuffle(1000).batch(128).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Add more trainings
image_batch, path_batch = next(iter(ds_test_batch))
predict_batch = model.predict(image_batch)

image_batch=image_batch   .numpy()
path_batch =path_batch.numpy()

for i in range(len(image_batch)):
    fig,ax=plt.subplots()
    ax.imshow(image_batch[i])
    ax.set_title(predict_batch[i][0])
    #ax.axis('off')
    fig.savefig('test.png')

    print('Correct classification?')
    x=input()
    if x=='':
        continue
    if x=='x':
        os.unlink(path_batch[i])
        continue

    x=float(x)
    x=f'{x:02.1f}'
    print(x)

    dirtrainname='train/{}'.format(x)
    if not os.path.exists(dirtrainname):
        os.mkdir(dirtrainname)

    fname=md5(path_batch[i])+'.jpg'
    pathtrainname='{}/{}'.format(dirtrainname,fname)
    if os.path.exists(pathtrainname):
        print('Already training on this...')
        continue

    os.rename(path_batch[i], pathtrainname)

# show_batch(image_batch, predict_batch)
# plt.savefig('test.pdf')

# # Make plot
# for image_batch, datetime_batch in iter(ds_test_batch):
#     datetimestr = [datetime.datetime.strptime(x.decode('utf-8'), '%Y%m%d%H%M%S') for x in datetime_batch.numpy()]

#     predicts_batch = np.argmax(models.predict(image_batch), axis=1)    
#     predict1_batch = np.argmax(model1.predict(image_batch), axis=1)
#     predict2_batch = np.argmax(model2.predict(image_batch), axis=1)
#     predict3_batch = np.argmax(model3.predict(image_batch), axis=1)
#     predict_batch=(predict1_batch*10+predict2_batch+predict3_batch/10)*(np.array([-1,+1])[predicts_batch])    

#     plt.plot(datetimestr, predict_batch,'.')
# plt.show()
    
