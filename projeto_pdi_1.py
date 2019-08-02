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
def save_image(opcao, img, image_name):
    save = input("Deseja salvar a imagem? (y/n): ")  
    save_path = './imagens_salvas/'
    opcoes = {'1': '_original.png', '2': '_yiq.png', 'r' : '_red.png', 
              'g': '_green.png', 'b': '_blue.png', 'n':'_negativo.png', 'nr': '_negativoRed.png',
              'ng': '_negativoGreen.png', 'nb': '_negativoBlue.png',
              'b' : '_brilho.png','br' : '_brilhoRed.png', 'bg' : '_brilhoGreen.png', 'bb' : '_brilhoBlue.png'}
    if save == 'y':
        plt.imsave( save_path + image_name + opcoes[opcao], img)
        return print("Imagem salva com sucesso!")
    elif save == 'n':
        return print("Sem salvar!")

def menu_banda_rgb(img, image_name):
    print("\nDigite a opcao para qual banda deseja visualizar: ")
    print("r - Red")
    print("g - Green")
    print("b - Blue")
    print("a - Todos")
    op = input("Opção: ")
    display_rgb(img, op, image_name)
    return print("Menu banda rgb fechado!")

def menu_negativo(img, image_name):
    print("\nDigite opcao de negativo: ")
    print("n - Negativo em todos os canais")
    print("nr - negativo em canal específico Red")
    print("ng - negativo em canal específico Green")
    print("nb - negativo em canal específico Blue")
    op = input("Opção: ")
    if op == 'n':
        negativo=negative(img)
        plt.imshow(negativo)
        save_image(op, negativo, image_name)
    else:
        img_negativa = negative_ch(img,op) 
        plt.imshow(img_negativa)
        save_image(op, img_negativa, image_name)

def menu_brightness_level(img, image_name):
    print("\nDigite opcao de brilho: ")
    print("b - Brilho em todos os canais")
    print("br - brilho em canal específico Red")
    print("bg - brilho em canal específico Green")
    print("bb - brilho em canal específico Blue")
    op = input("Opção: ")
    if op == 'b':
        level = input("Digite o nivel de brilho (ex: 2): ")
        img_brilhosa = brightness_level(img, float(level))
        plt.imshow(img_brilhosa)
        save_image(op, img_brilhosa, image_name +'_'+ level)
    else:
        level = input("Digite o nivel de brilho (ex: 2): ")
        img_brilhosa_canal = brightness_level_ch(img, float(level), op)
        plt.imshow(img_brilhosa_canal)
        save_image(op, img_brilhosa_canal, image_name +'_'+ level)

def display_rgb(photo, rgb, image_name):
  display_r = np.zeros(photo.shape)
  display_g = np.zeros(photo.shape)
  display_b = np.zeros(photo.shape)
  display_r[:,:,0] = photo[:,:,0]
  display_g[:,:,1] = photo[:,:,1]
  display_b[:,:,2] = photo[:,:,2]
  if rgb == 'r':
    plt.imshow(display_r, 'Reds')
    save_image(rgb, display_r, image_name)
  elif rgb == 'g':
    plt.imshow(display_g, 'Greens')
    save_image(rgb, display_g, image_name)
  elif rgb == 'b':
    plt.imshow(display_b, 'Blues')
    save_image(rgb, display_b, image_name)
  else:
    plt.figure(figsize=(20,10))

    plt.subplot(1, 3, 1)
    plt.imshow(display_r, 'Reds')
    save_image('r', display_r, image_name)
    
    plt.subplot(1, 3, 2)
    plt.imshow(display_g, 'Greens')
    save_image('g', display_g, image_name)

    plt.subplot(1, 3, 3)
    plt.imshow(display_b, 'Blues')
    save_image('b', display_b, image_name)

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
            rgb[i,j,0] = (1.000*photo[i,j,0] + 0.956*photo[i,j,1] + 0.621*photo[i,j,2])
            rgb[i,j,1] = (1.000*photo[i,j,0] - 0.272*photo[i,j,1] - 0.647*photo[i,j,2])
            rgb[i,j,2] = (1.000*photo[i,j,0] - 1.106*photo[i,j,1] + 1.703*photo[i,j,2])
            for c in range(0,3):
                if(rgb[i,j,c]<0):
                    rgb[i,j,c]=0
                if(rgb[i,j,c]>255):
                    rgb[i,j,c]=255
  
    return rgb


def brightness_level(photo, brightness):
  shape = photo.shape
  shinebright = np.zeros(photo.shape, dtype = int)
  for i in range(0,shape[0]): 
    for j in range(0,shape[1]):
      shinebright[i,j,0] = min(255, brightness * photo[i,j,0])
      shinebright[i,j,1] = min(255, brightness * photo[i,j,1])
      shinebright[i,j,2] = min(255, brightness * photo[i,j,2])
      
  return shinebright


def brightness_level_ch(photo, brightness, channel):
  brightness_ops = {'br': 0, 'bg':1, 'bb':2}  
  shape = photo.shape
  shinebright = np.zeros(photo.shape, dtype = int)
  for i in range(0,shape[0]): 
    for j in range(0,shape[1]):
        shinebright[i,j,0] = photo[i,j,0]
        shinebright[i,j,1] = photo[i,j,1]
        shinebright[i,j,2] = photo[i,j,2]
        shinebright[i,j,brightness_ops[channel]] = min(255, brightness * photo[i,j,brightness_ops[channel]])
    
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

def negative_ch(photo,channel):
    negative_ops = {'nr': 0, 'ng':1, 'nb':2}
    shape = photo.shape
    negative = np.zeros(photo.shape, dtype = int)
    for i in range(0,shape[0]): 
        for j in range(0,shape[1]):
            negative[i,j,0] = photo[i,j,0]
            negative[i,j,1] = photo[i,j,1]
            negative[i,j,2] = photo[i,j,2]
            negative[i,j,negative_ops[channel]] = 255-photo[i,j,negative_ops[channel]]
            
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

while True:
    m, n, mask = mask_from_file("input.txt")
    
    print("\nDigite a opcao desejada: ")
    print("1 - Inserir a imagem (Caso não tenha inserido, ou deseja mudar a imagem)")
    print("2 - Conversão RGB-YIQ-RGB ")
    print("3 - Exibir de bandas individuais RGB")
    print("4 - Negativo")
    print("5 - Controle de brilho multiplicativo")
    print("6 - Convolução m x n")
    print("7 - Filtro mediana m x n")
    print("0 - Sair")
    opcao = input("Opção: ")    
    if opcao == '0':
        print("\nEncerrando o programa...")
        break
    elif opcao == '1':
        [image_name, ext] = input("Digite o nome da imagem (Ex: lena256color.jpg): ").split('.')
        img_original = image.imread('./imagens_trab/' + image_name +'.'+ ext)
        img = img_original.copy()
        if ext == 'png':
            img = img*255
    elif opcao == '2':
        img_yiq = from_rgb_to_yiq(img)
        plt.imshow(img_yiq)
        save_image(opcao, img_yiq, image_name)
        img_rgb = from_yiq_to_rgb(img_yiq)
        plt.imshow(img_rgb)
    elif opcao == '3':
        menu_banda_rgb(img, image_name)
    elif opcao == '4':
        menu_negativo(img, image_name)
    elif opcao == '5':
        menu_brightness_level(img, image_name)

m, n, mask = mask_from_file("input.txt")

img_original = image.imread('./imagens_trab/CNN1.png')

img = img_original.copy()
img

img = img*255

# Para poder editar a imagem
img.flags 
plt.imshow(img)

img.shape


'''
Exibição de bandas individuais (R,G e B) em tons de R, G e B 
'''
display_rgb(img, 'all')

'''
Conversão RGB-YIQ-RGB
'''
img_yiq = from_rgb_to_yiq(img)

plt.imshow(img_yiq)

img_rgb = from_yiq_to_rgb(img_yiq)

plt.imshow(img_rgb)

'''
Negativo
'''
negativo=negative(img)
plt.imshow(negativo)
save_image('n', negativo, 'teste')

'''
Negativo em canal específico
'''
#Canal RED
img_negativa_R = negative_ch(img,0) 
plt.imshow(img_negativa_R)
#Canal GREEN
img_negativa_G = negative_ch(img,1) 
plt.imshow(img_negativa_G)
#Canal BLUE 
img_negativa_B = negative_ch(img,2) 
plt.imshow(img_negativa_B)

'''
Controle de brilho multiplicativo
'''
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
Controle de brilho multiplicativo em canal específico 
'''
img_brilhosa_red = brightness_level_ch(img, 2, 2)
plt.imshow(img_brilhosa_red)

'''
Convolução usando a média
'''
media = [[1/9,1/9,1/9],
        [1/9,1/9,1/9],
        [1/9,1/9,1/9]]

conv_media=convolution(img,m,n,media)

plt.imshow(conv_media)

'''
Convolução usando sobel X
'''
sobel_x = [[-1,0,1],
        [-2,0,2],
        [-1,0,1]]

conv_sobel_x=convolution(img,m,n,sobel_x)

plt.imshow(conv_sobel_x)

'''
Convolução usando sobel Y
'''
sobel_y = [[1,2,1],
        [0,0,0],
        [-1,-2,-1]]

conv_sobel_y=convolution(img,m,n,sobel_y)

plt.imshow(conv_sobel_y)

'''
Filtro da mediana 
'''
conv_mediana = convolution_median(img,m,n)

plt.imshow(conv_mediana)

'''
Convolução com mascara lida por arquivo
'''
conv_mask = convolution(img,m,n,mask)
plt.imshow(conv_mask)

plt.imshow(img)