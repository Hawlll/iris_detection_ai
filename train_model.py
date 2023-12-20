import tensorflow as tf
import json
from matplotlib import pyplot as plt

"""
To train this model, you will need to create your own dataset.
For crafting your dataset, I reccommend the following modules:

    OpenCv for collecting images
    Labelme for annotating the irises in each photo
    Albumentations for augmenting the original pictures in order to create more data

After the dataset is completed, the filepaths utilzed in the subsequent script will need to be changed according to YOUR file path.
"""


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_image(x):
    byte_img = tf.io.read_file(x)
    print('filename: ',byte_img)
    img = tf.io.decode_jpeg(byte_img)
    return img
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding= "utf-8") as f:
        label = json.load(f)
    return [label['keypoints']]


    
#Get file path
train = tf.data.Dataset.list_files('augmented_data\\train\\images\\*.jpg', shuffle=False)

#Load image in jpg form
train = train.map(load_image)

#Resize image to 250X250
train = train.map(lambda x: tf.image.resize(x, (250, 250)))

#Normalize values to 0-1, helps with model performance
train = train.map(lambda x: x/255)

test = tf.data.Dataset.list_files('augmented_data\\test\\images\\*.jpg', shuffle=False)
test = test.map(load_image)
test = test.map(lambda x: tf.image.resize(x, (250, 250)))
test = test.map(lambda x: x/255)

val = tf.data.Dataset.list_files('augmented_data\\val\\images\\*.jpg', shuffle=False)
val = val.map(load_image)
val = val.map(lambda x: tf.image.resize(x, (250, 250)))
val = val.map(lambda x: x/255)

#Get file path
trainLabels = tf.data.Dataset.list_files('augmented_data\\train\\labels\\*.json', shuffle=False)

#Grab annotation coordinates. (py_function allows use of all basic python functions)
trainLabels = trainLabels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

testLabels = tf.data.Dataset.list_files('augmented_data\\test\\labels\\*.json', shuffle=False)
testLabels = testLabels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

valLabels = tf.data.Dataset.list_files('augmented_data\\val\\labels\\*.json', shuffle=False)
valLabels = valLabels.map(lambda x: tf.py_function(load_labels, [x], [tf.float16]))

#Preproccess

#correlate images and labels together
training = tf.data.Dataset.zip((train, trainLabels))

#Shuffles data
training = training.shuffle(8000)

#Batch data into 16
training = training.batch(8)

#Loading images in memory while training, helps prevent bottlenecks
training = training.prefetch(4)

testing = tf.data.Dataset.zip((test, testLabels))
testing = testing.shuffle(2000)
testing = testing.batch(8)
testing = testing.prefetch(4)

validating = tf.data.Dataset.zip((val, valLabels))
validating = validating.shuffle(2000)
validating = validating.batch(8)
validating = validating.prefetch(4)

#Build model

"""
Input: layer that defines dimensions of input data
ResNet152V2: pretrained image classifer 
Conv2D: condenses data
Dropout: layer for regularization
Reshape: reshapes input
"""

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(250,250,3)),
    tf.keras.applications.ResNet152V2(include_top=False, input_shape=(250,250,3)),
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 3, 2, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 2, 2, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Conv2D(4, 2, 2),
    tf.keras.layers.Reshape((4,))
])
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0007)
loss = tf.keras.losses.MeanSquaredError()

model.compile(optimizer, loss)


epoch_time = 100 #NOTE: This many epochs will require a great amount of time, but provides a great performing model
hist = model.fit(training, epochs=epoch_time, validation_data=validating)

model.save('eyetracker100epchos.h5')

plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color=['orange'], label='val loss')
plt.suptitle("Loss Plot - Training vs Val")
plt.legend()
plt.show()