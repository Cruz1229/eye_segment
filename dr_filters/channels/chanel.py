import cv2
from matplotlib import pyplot as plt
import numpy as np



def __read_image(image_file):
    image_src = cv2.imread(image_file)
    
    #image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src



def channels(image_file,r=False, g=False ,b=False ,with_plot=False):
    if type(image_file) == str:
        image_src = __read_image(image_file=image_file)
    else:
        image_src = image_file

    r_channel, g_channel, b_channel = cv2.split(image_src)
    
    channels_list = []

    if r:
        channels_list.append(r_channel)
    if g:
        channels_list.append(g_channel)
    if b:
        channels_list.append(b_channel)
    

    if len(channels_list) == 0:
        channels_list.append(r_channel)
    

    image_merge = cv2.merge(channels_list)
    
    
    if with_plot:
        fig = plt.figure(figsize=(10, 20))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis("off")
        ax2.title.set_text("Enabled channels")

        ax1.imshow(image_src, cmap=None)
        ax2.imshow(image_merge, cmap=None)
        #return True

    return image_merge