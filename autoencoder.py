''' 
This is the implement of a basic autoencoder by keras
'''
# coding: utf-8

# In[1]:

from __future__ import print_function
from keras.layers import Input, Dense
from keras.models import Model


# In[2]:

# this is the size of our encoded representations
encoding_dim = 32


# In[3]:

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded  = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoced = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoced)


# In[4]:

print(autoencoder.summary())


# In[5]:

# create an encoder model
encoder = Model(input=input_img, output=encoded)
# create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# retrive the layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


# In[6]:

print(encoder.summary(), decoder.summary())


# In[7]:

# train autoencoder model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[8]:

# prepare dataset
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape, x_test.shape)


# In[9]:

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape, x_test.shape)


# In[10]:

# train on dataset
autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))


# In[11]:

# encode and decode some digits
import matplotlib.pyplot as plt

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# In[12]:

# show some images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[13]:

print(encoded_imgs.shape)


# In[14]:

plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[15]:

score = autoencoder.evaluate(x_test, x_test, verbose=0)
print(score)

