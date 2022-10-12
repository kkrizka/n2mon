#!/usr/bin/env python

import os
import glob
from PIL import Image

for imgpath in glob.glob('climatecamera/*snapshot.jpg'):
    imgname=os.path.basename(imgpath)
    if os.path.exists('images/{}'.format(imgname)):
        continue
    print(imgpath)

    img=Image.open(imgpath)
    imgcrop=img.crop((772,48,772+92,48+66))
    imgcrop.save('images/{}'.format(imgname),'JPEG')

