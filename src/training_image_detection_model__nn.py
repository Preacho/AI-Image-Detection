import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, SGD
from PIL import Image

#pip install tensorflow
#pip install tensorboard
#pip install pillow


def define_data(directory, batch_size, image_size):
    
    data = tf.keras.utils.image_dataset_from_directory(
    directory, 
    batch_size = batch_size,
    image_size = image_size  
) 
    data = data.map(lambda x, y: (x / 255, y))
    return data


def train_neural_network():
    epoch = 40
    num_classes = 2
    img_width = 300
    img_height = 300
    batch_size = 64
    
    train_directory = "processed_dataset/train"
    test_directory = "processed_dataset/test"

    data_train = define_data(
        train_directory, 
        batch_size, 
        (img_height,img_width)
        )
    data_test = define_data(
        test_directory, 
        batch_size, 
        (img_height, img_width)
        )

    train_size =  int(len(data_train) * 0.75)  
    val_size = int(len(data_train) * 0.25) 
    test_size = int(len(data_test) *1.0)

    train = data_train.take(train_size)
    validate = data_train.skip(train_size).take(val_size)
    test = data_test.take(test_size)

    nn_model = models.Sequential(
        [
        layers.Conv2D(16, (3, 3), 1, activation="relu", input_shape=(img_width, img_height, 3)),
        layers.Conv2D(16, (3, 3), 1, activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(32, (3, 3), 1, activation="relu"),
        layers.Conv2D(32, (3, 3), 1, activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), 1, activation="relu"),
        layers.Conv2D(64, (3, 3), 1, activation="relu"),
        
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
        ]
    )

    nn_model.compile(
        Adam(learning_rate=0.0001),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    history = nn_model.fit(train, epochs = epoch, validation_data = validate, callbacks = [early_stop, tensorboard_callback])

    nn_model.save("image_ai_detector4.h5")

if __name__ == "__main__":
    train_neural_network()