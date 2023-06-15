#!/usr/bin/env python
# coding: utf-8

# To start with deep learning, the very basic project that you can build is to predict the next digit in a sequence.

# In[14]:


import numpy as np
from tensorflow import keras 
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[15]:


sequences = [
    [1, 2, 3, 4, 5, 6],
    [5, 6, 7, 8, 9, 10],
    [2, 4, 6, 8, 10, 12],
]

input_sequences = [seq[:-1] for seq in sequences]
output_sequences = [seq[-1] for seq in sequences]


# In[17]:


x_train = np.array(input_sequences)
y_train = np.array(output_sequences)

model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10, batch_size=1)

input_sequence = [3, 6, 9]
padded_sequence = pad_sequences([input_sequence[:-1]], maxlen=5, padding='pre')
next_digit = model.predict(padded_sequence)[0][0]
print("Next digit prediction:", round(next_digit))


# Create a sequence like a list of odd numbers and then build a model and train it to predict the next digit in the sequence. 
# 

# In[21]:


sequence = [i for i in range(1, 20, 2)]

input_sequence = sequence[:-1]
output_sequence = sequence[1:]

x_train = np.array(input_sequence)
y_train = np.array(output_sequence)

model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10, batch_size=1)

input_digit = 19
next_digit = model.predict(np.array([input_digit]))[0][0]
print("Next digit prediction:", round(next_digit))


# Task: - A simple neural network with 2 layers would be sufficient to build the model.
# 

# In[22]:


model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=10, batch_size=1)

input_digit = 19
next_digit = model.predict(np.array([input_digit]))[0][0]
print("Next digit prediction:", round(next_digit))


# In[ ]:




