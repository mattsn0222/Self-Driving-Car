import numpy as np
import cv2
import matplotlib.image as mpimg

file1=open('image_raw_sample', 'r')
img_count=0
while True:
    line = file1.readline();
    if not line:
        break
    if line[:7]=="data: [":
        lines = line.strip()[7:-1].split(',')
        print(len(lines))
        arr=np.array(lines, np.uint8)
        img = cv2.cvtColor(np.reshape(arr, (-1,800)), cv2.COLOR_BayerGB2RGB)
        mpimg.imsave('img_{:04}.png'.format(img_count), img)
        img_count+=1

file1.close()
    