import tensorflow as tf #import tensorflow libraries
print (tf.__version__) #print version -- was causing an error in the beginning due to version
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
from keras.models import load_model

import json

dataset_path = 'myDataset' #getting the path to the dataset

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
model.add(layers.Conv2D(32, (3, 3), activation='relu')) #layer 2
model.add(layers.Conv2D(32, (3, 3), activation='relu')) #layer 3
model.add(layers.Flatten())
model.add(layers.Dense(53, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

your_epochs = 80
your_steps_per_epoch = len(train_generator)

# Model Training
history = model.fit(
    train_generator,
    epochs=your_epochs,
    steps_per_epoch=your_steps_per_epoch,
    verbose=1
)
print(type(history))



# Save the model architecture to JSON
model_json = model.to_json()
with open("UNO_CNN_Model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("UNO_CNN_Model.h5")
print("Saved model to disk")

# Optionally, you can also save the entire model (architecture + weights)
model.save("UNO_CNN_Model_complete.h5")

# Load the model later if needed
loaded_model = load_model("UNO_CNN_Model_complete.h5")