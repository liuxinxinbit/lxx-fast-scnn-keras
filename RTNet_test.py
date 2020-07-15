from RTNet import RTNet
import numpy as np
from tensorflow.keras.models import load_model, save_model
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
import imgviz
import time

def random_crop_or_pad(image, size=(448, 512)):
    if image.shape[0] > size[0]:
        crop_random_y = random.randint(0, image.shape[0] - size[0])
        image = image[crop_random_y:crop_random_y + size[0], :, :]
    else:
        zeros = np.zeros(
            (size[0], image.shape[1], image.shape[2]), dtype=np.float32)
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)

    if image.shape[1] > size[1]:
        crop_random_x = random.randint(0, image.shape[1] - size[1])
        image = image[:, crop_random_x:crop_random_x + size[1], :]
    else:
        zeros = np.zeros((image.shape[0], size[1], image.shape[2]))
        zeros[:image.shape[0], :image.shape[1], :] = image
        image = np.copy(zeros)

    return image


rtnet = RTNet()

# rtnet.train(epochs=10, steps_per_epoch=500, batch_size=4)
# rtnet.save()

# rtnet.load()
# start = time.time()
# for flag in range(500):
#     print(str(flag).zfill(5))
#     image = np.float32(Image.open("marine_data/11/images/"+str(flag+1).zfill(5)+".jpg"))/255
#     image = random_crop_or_pad(image)
#     plt.subplot(1, 2, 1)
#     plt.imshow(image)
    
#     prediction = rtnet.predict(image)
#     # print("1  ",time.time()-start)
#     result = np.zeros((448, 512))

#     a = prediction[0,:,:, 0]
#     b = prediction[0,:,:, 1]
#     c=  prediction[0,:,:, 2]
#     # result[np.multiply(a > b, a > c)] = 0
#     result[np.multiply(b > a, b > c)] = 1
#     result[np.multiply(c > a, c > b)] = 2


#     print("2  ",flag/(time.time()-start))
#     plt.subplot(1, 2, 2)
#     plt.imshow(result)
#     plt.pause(0.01)
#     plt.clf()
