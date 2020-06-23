import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from cnn import CNN
from os.path import isfile

# test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.load_img('dataset/test_set/cats/cat.4500.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
indices_file = 'class_names.npy'
if isfile(indices_file):
    class_indices = np.load(indices_file, allow_pickle=True).item()

model =  tf.keras.models.load_model('my_model.h5')
result = model.predict(test_image)
my_cnn = CNN()

print('str(result) '+str(result))

print('classes '+str(class_indices))

print(class_indices.values())

for i,val in enumerate(class_indices):
    print('i '+str(i))
    print('val '+str(val))

    if result[0][0] == class_indices[str(val)]:
        prediction = str(val)
        prediction = prediction[:-1]
        prediction = prediction[0:].capitalize()

try:
    prediction
except NameError:
    print("Sorry we could not define the prediction")
else:
    print("Final prediction: "+prediction)

    