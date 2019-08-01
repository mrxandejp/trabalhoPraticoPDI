# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:38:21 2019

@author: Alexandre Santos Marques
"""

from IPython.display import Image

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np
import skimage
from skimage import data, io
from statistics import median

import os
path = os.getcwd()
print ("Current working directory %s" % path)

os.chdir(path)

'''
FUNÇÕES
'''
def display_rgb(photo, rgb):
  display_r = photo[:,:,0]
  display_g = photo[:,:,1]
  display_b = photo[:,:,2]
  if rgb == 'r':
    plt.imshow(display_r, 'Reds')
  elif rgb == 'g':
    plt.imshow(display_g, 'Greens')
  elif rgb == 'b':
    plt.imshow(display_b, 'Blues')   
  else:
    plt.figure(figsize=(20,10))

    plt.subplot(1, 3, 1)
    plt.imshow(display_r, 'Reds')

    plt.subplot(1, 3, 2)
    plt.imshow(display_g, 'Greens')

    plt.subplot(1, 3, 3)
    plt.imshow(display_b, 'Blues')

def from_rgb_to_yiq(photo):
  shape = photo.shape
  yiq = np.zeros(photo.shape)
  for i in range(0,shape[0]):
    for j in range(0,shape[1]):
      yiq[i,j,0] = (0.299*photo[i,j,0] + 0.587*photo[i,j,1] + 0.114*photo[i,j,2]) #Y
      yiq[i,j,1] = (0.596*photo[i,j,0] - 0.274*photo[i,j,1] - 0.322*photo[i,j,2]) #I
      yiq[i,j,2] = (0.211*photo[i,j,0] - 0.523*photo[i,j,1] + 0.312*photo[i,j,2]) #Q
      
  return yiq

def from_yiq_to_rgb(photo):
  shape = photo.shape
  rgb = np.zeros(photo.shape, dtype = int)
  for i in range(0,shape[0]): 
    for j in range(0,shape[1]):
      rgb[i,j,0] = min((1.000*photo[i,j,0] + 0.956*photo[i,j,1] + 0.621*photo[i,j,2]),255)
      rgb[i,j,1] = min((1.000*photo[i,j,0] - 0.272*photo[i,j,1] - 0.647*photo[i,j,2]),255)
      rgb[i,j,2] = min((1.000*photo[i,j,0] - 1.106*photo[i,j,1] + 1.703*photo[i,j,2]),255)
      
  return rgb

def brightness_level(photo, brightness = 1):
  shape = photo.shape
  shinebright = np.zeros(photo.shape, dtype = int)
  for i in range(0,shape[0]): 
    for j in range(0,shape[1]):
      shinebright[i,j,0] = min(255, brightness * photo[i,j,0])
      shinebright[i,j,1] = min(255, brightness * photo[i,j,1])
      shinebright[i,j,2] = min(255, brightness * photo[i,j,2])
      
  return shinebright

def mask_from_file(file_name):
    file = open(file_name,"r") # abre o arquivo input.txt em modo de leitura (read)
    if file.mode == 'r':
        m = int(file.readline())
        n = int(file.readline())
        mask = np.zeros(shape=[m,n], dtype=int)
        m_aux = 0
        for lines in file:
            value = lines.split()
            for i in range(0, n):
                mask[m_aux, i] = int(value[i])
            m_aux+=1
    
    return m, n, mask
   
def negative(photo):
    shape = photo.shape
    negative = np.zeros(photo.shape, dtype = int)
    for i in range(0,shape[0]): 
        for j in range(0,shape[1]):
            negative[i,j,0] = 255-photo[i,j,0]
            negative[i,j,1] = 255-photo[i,j,1]
            negative[i,j,2] = 255-photo[i,j,2]
  
    return negative


def rebater(mask):
    mask=np.flipud(np.fliplr(mask))
    return mask

def convolution(photo,m,n,mask):
    mask = rebater(mask)
    shape = photo.shape
    #shape = [6,6]
    conv = np.zeros([shape[0]-(m-1),shape[1]-(n-1),3], dtype = int)
    for i in range(0,shape[0]-(m-1)): 
        for j in range(0,shape[1]-(n-1)):
            for i_m in range(0,m):
                for j_n in range(0,n):
                    conv[i,j,0]+=mask[i_m,j_n]*photo[i+i_m,j+j_n,0]
                    conv[i,j,1]+=mask[i_m,j_n]*photo[i+i_m,j+j_n,1]
                    conv[i,j,2]+=mask[i_m,j_n]*photo[i+i_m,j+j_n,2]
        for c in range(0,3):
            if(conv[i,j,c]<0):
                conv[i,j,c]=0
            if(conv[i,j,c]>255):
                conv[i,j,c]=255
    return conv

def convolution_median(photo,m,n):
    list_m0 = []
    list_m1 = []
    list_m2 = []
    shape = photo.shape
    #shape = [6,6]
    conv = np.zeros([shape[0]-(m-1),shape[1]-(n-1),3], dtype = int)
    for i in range(0,shape[0]-(m-1)): 
        for j in range(0,shape[1]-(n-1)):
            for i_m in range(0,m):
                for j_n in range(0,n):
                    list_m0.append(photo[i+i_m,j+j_n,0])   
                    list_m1.append(photo[i+i_m,j+j_n,1]) 
                    list_m2.append(photo[i+i_m,j+j_n,2])
            conv[i,j,0]=np.median(list_m0)
            conv[i,j,1]=np.median(list_m1)
            conv[i,j,2]=np.median(list_m2)
            list_m0.clear()
            list_m1.clear()
            list_m2.clear()
    return conv
         
'''
MAIN
'''

m, n, mask = mask_from_file("input.txt")

img_original = image.imread('./imagens_trab/lena256color.jpg')

img = img_original.copy()
img

# Para poder editar a imagem
img.flags 
plt.imshow(img)

img.shape

display_rgb(img, 'all')

img_yiq = from_rgb_to_yiq(img)

plt.imshow(img_yiq)

img_rgb = from_yiq_to_rgb(img_yiq)

plt.imshow(img_rgb)

negativo=negative(img)

plt.imshow(negativo)

img_brilhosa2 = brightness_level(img, 2)
img_brilhosa3 = brightness_level(img, 3)
img_brilhosa5 = brightness_level(img, 5)

plt.figure(figsize=(20,10))

plt.subplot(1, 4, 1)
plt.imshow(img)

plt.subplot(1, 4, 2)
plt.imshow(img_brilhosa2)

plt.subplot(1, 4, 3)
plt.imshow(img_brilhosa3)

plt.subplot(1, 4, 4)
plt.imshow(img_brilhosa5)
plt.show()

'''
TESTE CONVOLUÇÃO USANDO MÉDIA
'''
media = [[1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]]

conv_media=convolution(img,m,n,media)

plt.imshow(conv_media)

'''
TESTE CONVOLUÇÃO USANDO SOBEL X
'''
sobel_x = [[-1,0,1],
        [-2,0,2],
        [-1,0,1]]

conv_sobel_x=convolution(img,m,n,sobel_x)

plt.imshow(conv_sobel_x)

'''
TESTE CONVOLUÇÃO USANDO SOBEL Y
'''
sobel_y = [[1,2,1],
        [0,0,0],
        [-1,-2,-1]]

conv_sobel_y=convolution(img,m,n,sobel_y)

plt.imshow(conv_sobel_y)

'''
TESTE MEDIANA
'''
conv_mediana = convolution_median(img,m,n)

plt.imshow(conv_mediana)

plt.imshow(img)