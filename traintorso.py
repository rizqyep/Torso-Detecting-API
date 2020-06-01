import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = "dataset"

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=20,
    horizontal_flip=True,
    shear_range = 0.2,
    fill_mode = 'nearest',
    validation_split = 0.2 #bagi data untuk validasi sebanyak 10% dari dataset
)

#set data training dari bagian train datagen
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size =(300,200),
    batch_size = 8,
    class_mode='binary',# mode kelas binary karena 2 kelas
    subset = 'training'
)

#set data validasi dari train_datagen
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size =(300,200),
    batch_size = 8,
    class_mode='binary',
    subset = 'validation'
)

#bangun CNN 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
model = Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary() 

model.compile(loss ='binary_crossentropy', 
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(train_generator,
          steps_per_epoch = 8,
          epochs = 10,
          validation_data = validation_generator,
          validation_steps = 5,
          verbose = 2)

model.save('torso.h5') #save model