#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk(r"C:\Users\SANTANU\Downloads\archive (10)\NEU Metal Surface Defects Data"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train_dir = r"C:\Users\SANTANU\Downloads\archive (10)\NEU Metal Surface Defects Data\train"
test_dir = r"C:\Users\SANTANU\Downloads\archive (10)\NEU Metal Surface Defects Data\test"
valid_dir = r"C:\Users\SANTANU\Downloads\archive (10)\NEU Metal Surface Defects Data\valid"


# In[3]:


train_datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=20,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


#get the images from train datagen
train_generator = train_datagen.flow_from_directory(train_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=True)
valid_generator = test_datagen.flow_from_directory(valid_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)

test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(200, 200),
                                              batch_size=32,
                                              class_mode='categorical',
                                              shuffle=False)


# In[4]:


#checking for batch size
for image_batch , labels_batch in train_generator :
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# In[5]:


#checking for class names
class_names = train_generator.class_indices
class_names = list(class_names.keys())
print(class_names)


# In[6]:


#making functions
def get_sample_image(generator):
    images, labels = next(generator)
    image = images[0]
    label_index = np.argmax(labels[0])
    label_name = class_names[label_index]

    return image, label_name


def sample_images(generator, nrows=3, ncols=3):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    
    for i in range(nrows*ncols):
        image, label_name = get_sample_image(generator)
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]
        ax.imshow(image)
        ax.set_title(label_name)
        ax.axis('off')

    plt.show()


# In[7]:


sample_images(train_generator, nrows=3, ncols=3)


# # model using cnn

# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.applications.efficientnet import EfficientNetB7


# In[9]:


model1 = Sequential([ Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(6 ,activation='softmax')])


# In[10]:


model1.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[11]:


model1.summary()


# In[12]:


history = model1.fit(train_generator,
                    epochs=15,
                    batch_size=32,
                    validation_data=valid_generator)


# In[113]:


def plot_history(history,metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_"+metric],"")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_"+metric])
    plt.show()
plot_history(history,"accuracy")
plot_history(history,"loss")


# In[22]:


pip install scikeras


# In[23]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import StackingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# In[24]:


# Define CNN model function
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))  # Assuming 6 classes for metal surface detection

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[25]:


# Create instances of CNN models
base_model1 = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=1)
base_model2 = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=1)
base_model3 = KerasClassifier(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=1)


# In[20]:


base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top = False , weights = 'imagenet' ,
                                                               input_shape = (200,200,3), pooling= 'max')
model2 = Sequential([ Conv2D(32, (2, 2), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (2, 2), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(6 ,activation='softmax')])


# In[107]:


# Define stacking classifier with base models and meta-classifier
stacking_models = [('cnn1', base_model1), ('cnn2', base_model2), ('cnn3', base_model3)]
meta_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
stacking_classifier = StackingClassifier(estimators=stacking_models, final_estimator=meta_classifier)


# 
# 

# In[116]:


history2 = model2.fit(train_generator,
                    epochs=20,
                    batch_size=32,
                    validation_data=valid_generator)


# In[117]:


def plot_history(history, metric):
    plt.plot(history2.history[metric])
    plt.plot(history2.history['val_'+metric], '')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()
plot_history(history, 'accuracy')
plot_history(history, 'loss')


# In[118]:


result = model2.evaluate(test_generator)
print("Test loss, Test accuracy : ", result)


# # ensembling both models

# In[119]:


import numpy as np

# Assuming preds_base_model and preds_model2 are the predictions from the respective models

# Reshape the predictions to a compatible shape for averaging
preds_base_model_reshaped = preds_base_model.reshape(-1, 72, 6)
preds_model2_reshaped = preds_model2.reshape(-1, 72, 6)

# Average predictions for each sample
avg_preds_base_model = np.mean(preds_base_model_reshaped, axis=0)
avg_preds_model2 = np.mean(preds_model2_reshaped, axis=0)

# Average the averaged predictions from both models
final_predictions = (avg_preds_base_model + avg_preds_model2) / 2

# Evaluate the ensemble method's performance
# Assuming test_labels contains the true labels for the test set
ensemble_loss, ensemble_accuracy = model2.evaluate(test_generator)
print("Ensemble Test loss, Ensemble Test accuracy : ", ensemble_loss, ensemble_accuracy)


# In[ ]:





# In[120]:


# Generate predictions for test data using model1 and model2
test_preds_model1 = model1.predict(test_generator)
test_preds_model2 = model2.predict(test_generator)

 #Assuming test_labels contains the true labels for the test set
# Replace this with your actual labels
test_labels = load_test_labels()  # Replace with your function or method to load test labels


# In[121]:


# Combine predictions from model1 and model2
stacking_input = np.concatenate([test_preds_model1, test_preds_model2], axis=1)

# Train a meta-model (e.g., RandomForestClassifier) using stacking_input and test_labels
from sklearn.ensemble import RandomForestClassifier

meta_model = RandomForestClassifier(n_estimators=100)
meta_model.fit(stacking_input, test_labels)

# Evaluate the stacking ensemble method
ensemble_accuracy = meta_model.score(stacking_input, test_labels)
print(f"Stacking Ensemble Accuracy: {ensemble_accuracy}")


# # visualisation

# In[123]:


images, labels = next(test_generator)    

indices = np.random.choice(range(len(images)), size=9)
images = images[indices]
labels = labels[indices]

predictions = model1.predict(images)


class_names=list(test_generator.class_indices.keys())


plt.figure(figsize=(10,10))
    
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
        
    image = images[i]
        
    if image.shape[-1] == 1:
        image = np.squeeze(image) 
        
    plt.imshow(image)
        
    predicted_label = np.argmax(predictions[i])
        
    if predicted_label == np.argmax(labels[i]):
        color='blue'
        result_text="Correct"
            
    else:
        color='red'
        result_text="Incorrect"

    label_text="True: "+ class_names[np.argmax(labels[i])] + ", Pred: " + class_names[predicted_label] + f" ({result_text})"        
            
    plt.xlabel(label_text,color=color)


# In[124]:


mages, labels = next(test_generator)    

indices = np.random.choice(range(len(images)), size=9)
images = images[indices]
labels = labels[indices]

predictions = model2.predict(images)


class_names=list(test_generator.class_indices.keys())


plt.figure(figsize=(10,10))
    
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
        
    image = images[i]
        
    if image.shape[-1] == 1:
        image = np.squeeze(image) 
        
    plt.imshow(image)
        
    predicted_label = np.argmax(predictions[i])
        
    if predicted_label == np.argmax(labels[i]):
        color='blue'
        result_text="Correct"
            
    else:
        color='red'
        result_text="Incorrect"

    label_text="True: "+ class_names[np.argmax(labels[i])] + ", Pred: " + class_names[predicted_label] + f" ({result_text})"        
            
    plt.xlabel(label_text,color=color)


# In[125]:


import pandas as pd

# Define the accuracy and loss values for both models
model1_accuracy = history.history['accuracy'][-1]
model1_loss = history.history['loss'][-1]
model2_accuracy = history2.history['accuracy'][-1]
model2_loss = history2.history['loss'][-1]


# Create a dictionary with the accuracy and loss values
data = {'Model': ['Model 1', 'Model 2'],
        'Accuracy': [model1_accuracy, model2_accuracy],
        'Loss': [model1_loss, model2_loss]}

df = pd.DataFrame(data)
df


# In[ ]:




