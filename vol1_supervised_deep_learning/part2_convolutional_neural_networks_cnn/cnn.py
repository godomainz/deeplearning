import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from datetime import datetime
import csv

class CNN:

    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = 'ckpt'

    # Prepossessing training set
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(64, 64), batch_size=32,
                                                    class_mode='binary')

    # Prepossessing test set
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')

    # Initializing CNN
    cnn = tf.keras.models.Sequential()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    def __init__(self):
        print("tf.__version__ "+tf.__version__)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def compile_model(self,my_model):
        model = my_model
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return model

    def make_or_restore_model(self,model):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [self.checkpoint_dir + '/' + name
                    for name in os.listdir(self.checkpoint_dir)]
        print("len(checkpoints) "+str(len(checkpoints)))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print('Restoring from', latest_checkpoint)
            return tf.keras.models.load_model(latest_checkpoint)
        print('Creating a new model')
        return self.compile_model(model)

    def first_convolution(self):
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
   

    def max_pooling(self):
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    def second_convolution(self):
        self.cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
        self.cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    def flattening(self):
        self.cnn.add(tf.keras.layers.Flatten())

    def full_connection(self):
        self.cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

    def output_layer(self):
        self.cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    
    def compile_fit_save(self):
        model = self.make_or_restore_model(self.cnn)
        # model.save('my_model')  # creates a HDF5 file 'my_model.h5'
        # cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

        callbacks = [
            # This callback saves a SavedModel every 100 batches.
            # We include the training loss in the folder name.
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_dir + '/'+ 'ckpt-loss={loss:.2f}'+'_TIMESTAMP_'+self.get_current_date_time()+'.h5',
                save_freq=100)
        ]

        # Training the CNN on the training set and evaluating it on the Test set
        model.fit(x=self.training_set, validation_data=self.test_set, steps_per_epoch=250, epochs=25, verbose=1,validation_steps=250,callbacks=callbacks)
        # model.fit_generator(self.training_set,steps_per_epoch=250, validation_data = self.test_set, epochs = 25,validation_steps=250)
        model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    def save_class_names(self):
        np.save('class_names.npy', self.training_set.class_indices)
        with open('class_names.csv', 'w') as f:
            for key in self.training_set.class_indices.keys():
                f.write("%s,%s\n"%(key,self.training_set.class_indices[key]))

    def get_current_date_time(self):
        today = datetime.today()
        date = today.strftime("%Y_%m_%d")
        print("date =", date)
        now = datetime.now()
        current_time = now.strftime("%H_%M_%S")
        print("Current Time =", current_time)
        return date+"_"+current_time

if __name__ == "__main__":
    my_cnn = CNN()
    my_cnn.save_class_names()
    print(my_cnn.get_current_date_time())
    # Step 1 - Convolution
    my_cnn.first_convolution()
    # Step 2 - Pooling
    my_cnn.max_pooling()
    # Adding second convolutional layer
    my_cnn.second_convolution()
    # Step 3 - Flattening
    my_cnn.flattening()
    # Step 4 - Full Connection
    my_cnn.full_connection()
    # Step 5 - Output Layer
    my_cnn.output_layer()
    # Compiling the CNN
    my_cnn.compile_fit_save()
    
