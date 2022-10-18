#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time, datetime
import os
import urllib.request
import shutil

import n2mon

#
# Load data

model=tf.keras.models.load_model('mymodel')
print(model)

df=pd.DataFrame(columns=['time','n2level'])

fig_img,ax_img=plt.subplots()
fig_ts ,ax_ts =plt.subplots()

webcamurl='http://epweb2.ph.bham.ac.uk/user/thomas/tracker/trh/lastsnap.jpg'
lastmd5=None
while True:
    resp=urllib.request.urlretrieve(webcamurl)
    imgpath=resp[0]

    # Dummy check to make sure that the image is completely copied.
    # Download the same file twice in a row.
    mymd5=n2mon.md5(imgpath)
    if lastmd5!=mymd5:
        lastmd5=mymd5
        continue

    img=n2mon.load_image(imgpath)
    n2level = model.predict(np.expand_dims(img,axis=0))

    df=pd.concat([df,pd.DataFrame([{'time':datetime.datetime.now(), 'n2level':n2level[0][0]}])])
    print(df)

    ax_img.cla()
    ax_img.imshow(img)
    ax_img.set_title(n2level[0][0])
    fig_img.savefig('img.png')

    ax_ts.cla()
    ax_ts.plot(df['time'],df['n2level'],'.')
    ax_ts.set_xlabel('Time')
    ax_ts.set_ylabel('N2 Pressure [???]')
    fig_ts.savefig('timeseries.png')

    # Save the image for later training
    fname=mymd5+'.jpg'
    pathtrainname='{}/{}'.format('test',fname)
    if os.path.exists(pathtrainname):
        print('Already training on this...')
    else:
        shutil.move(imgpath, pathtrainname)

    time.sleep(20)
