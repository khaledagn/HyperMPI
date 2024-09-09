# -*- coding: utf-8 -*-
import os
import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from datetime import datetime
import sys

# Config
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# PATH
train_path = os.path.abspath('training/data_xray/train')
test_path = os.path.abspath('training/data_xray/test')
results_path = os.path.abspath('training/results')
base_path = os.path.abspath('training/data_xray')
val_path = os.path.join(base_path, 'val')

os.makedirs(results_path, exist_ok=True)

# Data Augmentation
s = 100  # Image size for resizing
datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(train_path, target_size=(s, s), batch_size=32, class_mode='binary', classes=['bacteria', 'virus'])
test_generator = test_datagen.flow_from_directory(test_path, target_size=(s, s), batch_size=32, class_mode='binary', classes=['bacteria', 'virus'])

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

# Check the number of processes
if size != 6:
    raise ValueError("This script requires 6 MPI processes")

def log_progress(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
    sys.stdout.flush()

# Function for parallel convolution
def parallel_convolution(layer, data, rank, size, comm):
    if rank == 0:
        log_progress(f"Master on {hostname}: Starting parallel convolution.")
        weights = layer.get_weights()
        if len(weights) != 2:
            raise ValueError(f"Expected 2 weights (filters and biases), got {len(weights)}")
        filters, biases = weights
        split_data = np.array_split(data, size)
        log_progress(f"Master on {hostname}: Sending filters and biases to all nodes.")
    else:
        filters = None
        biases = None
        split_data = None

    filters = comm.bcast(filters, root=0)
    biases = comm.bcast(biases, root=0)
    if rank != 0:
        log_progress(f"Node {rank} on {hostname}: Received filters and biases.")

    local_data = comm.scatter(split_data, root=0)
    if rank != 0:
        log_progress(f"Node {rank} on {hostname}: Received data chunk for processing.")

    log_progress(f"Node {rank} on {hostname}: local_data shape: {local_data.shape}, filters shape: {filters.shape}")

    local_output = tf.nn.conv2d(local_data, filters, strides=[1, 1, 1, 1], padding='SAME')
    local_output = tf.nn.bias_add(local_output, biases)
    log_progress(f"Node {rank} on {hostname}: Finished local convolution.")

    gathered_output = comm.gather(local_output, root=0)
    if rank == 0:
        log_progress(f"Master on {hostname}: Gathering results from all nodes.")

    if rank == 0:
        output = np.concatenate(gathered_output, axis=0)
        log_progress(f"Master on {hostname}: Aggregated results from all nodes.")
        return output
    else:
        return None

if rank == 0:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_results_path = os.path.join(results_path, timestamp)
    os.makedirs(current_results_path, exist_ok=True)
    log_progress(f"Results will be saved to {current_results_path}")

    model = Sequential([
        Input(shape=(s, s, 3)),
        Conv2D(16, (3, 3), activation='relu', name='conv2d_1'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_3'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    initial_data, _ = next(train_generator)
    model.predict(initial_data)

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            weights = layer.get_weights()
            if len(weights) == 0:
                raise ValueError(f"Layer {layer.name} not initialized correctly.")
            else:
                log_progress(f"Layer {layer.name} initialized with weights: {weights[0].shape}, {weights[1].shape}")

    conv_layers = [model.get_layer('conv2d_1'), model.get_layer('conv2d_2'), model.get_layer('conv2d_3')]

    terminate = False

    for layer in conv_layers:
        if terminate:
            break

        for batch_data, _ in train_generator:
            if layer == model.get_layer('conv2d_1'):
                input_data = batch_data
            else:
                input_data = model.layers[model.layers.index(layer)-1](input_data)

            output = parallel_convolution(layer, input_data, rank, size, comm)
            if output is not None:
                input_data = output

            terminate = True
            break

    overall_start_time = time.time()
    log_progress("Training the model, please wait...")

    start_time = time.time()
    history = model.fit(train_generator, epochs=40, validation_data=test_generator)
    end_time = time.time()
    training_time = end_time - start_time
    overall_end_time = time.time()
    overall_training_time = overall_end_time - overall_start_time
    log_progress(f"Training time: {training_time} seconds")
    log_progress(f"Overall training time: {overall_training_time} seconds")

    log_progress("Saving training history plots...")

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = os.path.join(current_results_path, 'accuracy_plot.png')
    plt.savefig(accuracy_plot_path)

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(current_results_path, 'loss_plot.png')
    plt.savefig(loss_plot_path)

    log_progress("Saving evaluation results...")

    model_loss, model_accuracy = model.evaluate(test_generator)
    log_progress(f'Test Loss: {model_loss}')
    log_progress(f'Test Accuracy: {model_accuracy}')

    with open(os.path.join(current_results_path, 'evaluation_results.txt'), 'w') as f:
        f.write(f'Test Loss: {model_loss}\n')
        f.write(f'Test Accuracy: {model_accuracy}\n')

    log_progress("Saving sample prediction...")

    sample_image_path = os.path.join(val_path, 'bacteria', 'person1946_bacteria_4874.jpeg')
    sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(s, s))
    sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
    sample_image = np.expand_dims(sample_image / 255.0, axis=0)
    prediction = model.predict(sample_image)
    log_progress(f'Prediction: {prediction[0]}')

    with open(os.path.join(current_results_path, 'sample_prediction.txt'), 'w') as f:
        f.write(f'Prediction: {prediction[0]}\n')

    log_progress("Saving the model...")

    model_save_path = os.path.join(current_results_path, 'model.keras')
    model.save(model_save_path)
    log_progress(f'Model saved to {model_save_path}')

else:
    while True:
        output = parallel_convolution(None, None, rank, size, comm)
        if output is None:
            break

