import tensorflow as tf #import tensorflow libraries
print (tf.__version__) #print version -- was causing an error in the beginning due to version
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import json

dataset_path = 'Documents/unoCardDetectionUsingCNN/myDataset' #getting the path to the dataset

train_datagen = ImageDataGenerator( #image processing
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224), #set image size to 224*224
    batch_size=50,
    class_mode='categorical',
    shuffle=True,
    seed=42
)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224,224,3)))
#Add more layers
model.add(layers.Flatten())
model.add(layers.Dense(53, activation='softmax'))

#adding another layer
#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(54, activation='softmax'))

#and another layer :D
#model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(54, activation='softmax'))


model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

your_epochs = 50
your_steps_per_epoch = len(train_generator)

# Model Training
history = model.fit(
    train_generator,
    epochs=your_epochs,
    steps_per_epoch=your_steps_per_epoch,
    verbose=1
)
print(type(history))