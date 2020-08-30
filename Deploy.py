import streamlit as st
import altair as altc
import pandas as pd
import numpy as np
import os, urllib, cv2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


import tensorflow as tf
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True



import keras
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,UpSampling2D,Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate, add
from skimage.transform import resize
from keras import backend as K


page_bg_img = '''
<style>
body {
    background-image: url("https://i.pinimg.com/originals/eb/7e/06/eb7e06bd05368b84635aa4f4ce81ddd8.jpg");
    background-size: cover;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache
def segnet():
    input1=Input((256,256,3))
    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(input1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    c1=BatchNormalization()(conv2)
    drop1 = Dropout(0.1)(c1)
    pool1 =MaxPooling2D(pool_size=(2, 2))(drop1)
    
    conv1=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    c2=BatchNormalization()(conv2)
    drop2 = Dropout(0.1)(c2)
    pool2 =MaxPooling2D(pool_size=(2, 2))(drop2) 
    
    conv1=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c3=BatchNormalization()(conv3)
    drop3 = Dropout(0.1)(c3)
    pool3 =MaxPooling2D(pool_size=(2, 2))(drop3) 
    
    conv1=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c4=BatchNormalization()(conv3)
    drop4 = Dropout(0.1)(c4)
    pool4 =MaxPooling2D(pool_size=(2, 2))(drop4) 
    
    conv1=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool4)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    c5=BatchNormalization()(conv3)
    drop5 = Dropout(0.1)(c5)
    pool5 =MaxPooling2D(pool_size=(2, 2))(drop5) 
    

    
    up1 =Conv2D(1024,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(pool5))
    merge1 = concatenate([c5,up1], axis =3)
    
    conv1=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge1)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(1024,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)
    
    
    up2 =Conv2D(512,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge2 = concatenate([c4,up2], axis =3)
    
    conv1=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge2)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)
    

    up3 =Conv2D(256,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge3 = concatenate([c3,up3], axis =3)

    conv1=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge3)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    conv3=Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch2)
    batch3=BatchNormalization()(conv3)
    batch3 = Dropout(0.2)(batch3)
    

    up4 =Conv2D(128,2, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch3))
    merge4 = concatenate([c2,up4], axis =3) 

    conv1=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(merge4)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    batch2 = Dropout(0.2)(batch2)
    
    
    up5 =Conv2D(64,1, activation = 'relu', padding = 'same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(batch2))
    merge5 = concatenate([c1,up5], axis =3) 

    conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge5)
    batch1=BatchNormalization()(conv1)
    conv2=Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(batch1)
    batch2=BatchNormalization()(conv2)
    
    
    output=Conv2D(1,(1,1),activation='sigmoid')(batch2)
    
    model=Model(input1,output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.load_weights('final_1.h5')
    return model




def portrait_seg(im):
    im=np.reshape(im,(1,256,256,3))
    model=segnet()
    pred_img = model.predict(im)
    return np.reshape(pred_img,(256,256))

def convert_to_img(img,pred):
    pre=pred<0.5
    pre=pre.astype(int)
    pre=pre*255
    ind=np.where(pre==0)
    mask=img.copy()
    mask[ind[0],ind[1],:]=0
    return mask



st.set_option('deprecation.showfileUploaderEncoding', False)

st.title('Portrait Segmentation')
st.sidebar.title("Portrait Segmentation:")
st.sidebar.markdown('Portrait segmentation refers to the process of segmenting a person in an image from its background.')



uploaded_file=st.file_uploader('Upload the Image',type=['jpg','png'])


uploaded_text = st.text_input('Enter Image URL')
bp=st.button('proceed')
if bp:
    if uploaded_text is not None:
        temp=st.success('Processing the Image')
        urllib.request.urlretrieve(uploaded_text,"sample.png")
        src_img   = Image.open('sample.png')
        img       = src_img.resize((256,256))
        label_img = portrait_seg(img)
        final_img = convert_to_img(np.array(img),label_img)
        st.image([img,final_img],caption=['Original Image','Portrait Segment Image'] ,width=300, use_column_width=False)
        temp.empty()

if uploaded_file is not None:
    s=st.success('Processing the Image')
    src_img   = Image.open(uploaded_file)
    img       = src_img.resize((256,256))
    label_img = portrait_seg(img)
    final_img = convert_to_img(np.array(img),label_img)
    s.success('Done.....!!!!')
    s1=st.image([img,final_img],caption=['Original Image','Portrait Segment Image'] ,width=300, use_column_width=False)        
    s.empty()

def back_grd(im1,pre):
    ind1=np.where(pre!=0)
    backg=im1.copy()
    backg=np.array(backg)
    backg[ind1[0],ind1[1],:]=0
    return backg

agree = st.sidebar.checkbox("add background")
back_grds = ["flag.jpg","wall.jpg","wall1.jpg","flag1.jpg","wall4.jpg","wall5.jpg"]

@st.cache  
def back_pick(n):
    b1=Image.open(back_grds[n-1])
    b1=b1.resize((256,256))
    return b1


if agree:
    pick_img = st.sidebar.radio("Which Background?", [x for x in range(1, len(back_grds)+1)])
    st.sidebar.image(back_grds,width=100)
    b1=back_pick(pick_img)
    pre=label_img<0.5
    pre=pre.astype(int)
    pre=pre*255
    backg= back_grd(np.array(b1),pre)
    im=backg+final_img
    s1.image([img,im],caption=['Original Image','Modified Image'] ,width=300, use_column_width=False)











