
import os
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, utils
import numpy as np
import argparse
from google.cloud import storage
    
# Define the CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def train_model(args):
    # Input arguments
    project_id = args.project_id
    bucket_name = args.bucket_name
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    ### DATA PREPARATION SECTION ###
    
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0    
    # Convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    
    ### MODEL TRAINING AND EVALUATION SECTION ###

    if tf.config.list_physical_devices('GPU'):
        device = '/GPU:0'
    else:
        device = '/CPU:0'
    
    with tf.device(device):
        net = create_model()

    # Compile the model
    net.compile(optimizer=optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the network
    history = net.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_test, y_test))

    # Save the trained model locally
    net.save(model_path)
    
    # Return the trained model
    return net  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model on the CIFAR-10 dataset')
    
    parser.add_argument('--project_id', type=str, help='GCP Project ID')
    parser.add_argument('--bucket_name', type=str, help='GCP Bucket ID')
    parser.add_argument('--model_path', type=str, help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    train_model(args)
