'''
This is a convolution denoising autoencoder implemented with keras.
'''
# coding: utf-8

# In[1]:

# load mnist dataset
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)


# In[2]:

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
print(x_train.shape, x_test.shape)


# In[3]:

# show some figures
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (10, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[4]:

# add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
print(x_train_noisy.shape, x_test_noisy.shape)


# In[5]:

# normalization
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)


# In[6]:

# show some figures
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_train_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[7]:

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
# define autoencoder model
input_img = Input(shape=(1, 28, 28))

# encoder part
x = Conv2D(28, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_img)
x = MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_first')(x)
x = Conv2D(28, kernel_size=(3, 3), activation='relu', padding='same',data_format='channels_first')(x)
encoded = MaxPooling2D(pool_size=(2, 2), padding='same', data_format='channels_first')(x)

# decoder part
x = Conv2D(28, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(encoded)
x = UpSampling2D(size=(2, 2), data_format='channels_first')(x)
x = Conv2D(28, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x)
x = UpSampling2D(size=(2, 2), data_format='channels_first')(x)
decoded = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same', data_format='channels_first')(x)

autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[8]:

print(autoencoder.summary())


# In[15]:

from keras.callbacks import TensorBoard
# train our model
history = autoencoder.fit(x_train_noisy, x_train, batch_size=128, epochs=20, shuffle=True,
                validation_data=(x_test_noisy, x_test))


# In[30]:

x_rec = autoencoder.predict(x_test_noisy)
print(x_rec.shape)


# In[31]:

print(history.history.keys())


# In[39]:

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# In[45]:

plt.figure(figsize=(10, 2))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_rec[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[ ]:



