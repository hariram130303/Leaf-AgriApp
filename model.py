#model.py
# Importing Keras libraries and packages
from keras.models import Sequential # type: ignore
from keras.layers import Convolution2D # type: ignore
from keras.layers import MaxPooling2D # type: ignore
from keras.layers import Flatten # type: ignore
from keras.layers import Dense # type: ignore
from keras.layers import Dropout # type: ignore
from keras.layers import BatchNormalization # type: ignore
from keras.models import load_model
import numpy as np

# Initializing the CNN
classifier = Sequential()

# Convolution Step 1
classifier.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(224, 224, 3), activation = 'relu'))

# Max Pooling Step 1
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Convolution Step 2
classifier.add(Convolution2D(256, 11, strides = (1, 1), padding='valid', activation = 'relu'))

# Max Pooling Step 2
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding='valid'))
classifier.add(BatchNormalization())

# Convolution Step 3
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 4
classifier.add(Convolution2D(384, 3, strides = (1, 1), padding='valid', activation = 'relu'))
classifier.add(BatchNormalization())

# Convolution Step 5
classifier.add(Convolution2D(256, 3, strides=(1,1), padding='valid', activation = 'relu'))

# Max Pooling Step 3
classifier.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid'))
classifier.add(BatchNormalization())

# Flattening Step
classifier.add(Flatten())

# Full Connection Step
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1000, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 38, activation = 'softmax'))
classifier.summary()

classifier.save('Model.keras')

#Loading Weights To The Model**
classifier.load_weights('Model.keras')

#Fine Tuning By Freezing Some Layers Of Our Model**
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
from keras import layers
for i, layer in enumerate(classifier.layers):
   print(i, layer.name)

   # we chose to train the top 2 conv blocks, i.e. we will freeze
# the first 8 layers and unfreeze the rest:
print("Freezed layers:")
for i, layer in enumerate(classifier.layers[:20]):
    print(i, layer.name)
    layer.trainable = False

#Model Summary After Freezing**
#trainable parameters decrease after freezing some bottom layers   
classifier.summary()

#Compiling the Model**
# Compiling the Model
from keras import optimizers
classifier.compile(optimizer='adam', 
                   loss='sparse_categorical_crossentropy', 
                   metrics=['accuracy'])


#Image Preprocessing
# image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Use correct base path to your actual folder
base_dir = r"D:\New_folder\hari_practice\Deep_Learning\Success\Codeclause-main\New folder\Crop Disease Identification"

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 128

# Use COLOR images path
train_dir = os.path.join(base_dir, 'train', 'color')
valid_dir = os.path.join(base_dir, 'valid', 'color')

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size=(224, 224),
                                                 batch_size=batch_size,
                                                 class_mode='sparse')

valid_set = valid_datagen.flow_from_directory(valid_dir,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              class_mode='sparse')


class_dict = training_set.class_indices
print(class_dict)

li = list(class_dict.keys())
print(li)

train_num = training_set.samples
valid_num = valid_set.samples

# checkpoint
from keras.callbacks import ModelCheckpoint # type: ignore
weightpath = "best_weights.weights.h5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

#fitting images to CNN
history = classifier.fit(training_set,
                         steps_per_epoch=train_num//batch_size,
                         validation_data=valid_set,
                         epochs=2,
                         validation_steps=valid_num//batch_size,
                         callbacks=callbacks_list)
#saving model
filepath="AlexNetModel.hdf5"
classifier.save(filepath)

#Visualising Training Progress**
#plotting training values
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

from sklearn.metrics import classification_report, confusion_matrix

# Predict on validation set
Y_pred = classifier.predict(valid_set)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(valid_set.classes, y_pred))
print('Classification Report')
print(classification_report(valid_set.classes, y_pred, target_names=valid_set.class_indices.keys()))

#accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
#loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Predicting New Test Image(s)**
# predicting an image
from tensorflow.keras.preprocessing import image#type:ignore
from tensorflow.keras.utils import load_img, img_to_array

image_path = r"D:\New_folder\hari_practice\Deep_Learning\Success\Codeclause-main\Crop Disease Identification\train\Tomato___Early_blight\fdc50285-836e-4707-a17d-388b8179c87c___RS_Erly.B 7812.JPG"
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

print("Following is our prediction:")
prediction = classifier.predict(img)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
predicted_index = np.argmax(prediction)
class_name = li[predicted_index]


##Another way
# img_class = classifier.predict_classes(img)
# img_prob = classifier.predict_proba(img)
# print(img_class ,img_prob )


#ploting image with predicted class name        
plt.figure(figsize = (4,4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()
