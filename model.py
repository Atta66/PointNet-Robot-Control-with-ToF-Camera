import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import h5py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

######################################################################################################################

DATASET_DIR = 'data/'
f = h5py.File(DATASET_DIR+ 'd0.h5','r')
NUM_FRAMES = f['data'].shape[0]


pcds = np.zeros((1*NUM_FRAMES,1024, 3))  # 4 to Num
labels = np.zeros((1*NUM_FRAMES,1024))  # 4 to num

for i in range (0,1): # 4 to 1
    f = h5py.File(DATASET_DIR+ 'd' +str(i)+'.h5','r')
    data = f['data']
    label = f['pid']
    # label = f['pid']
    pcds[i*len(data):(i+1)*len(data),:,0:3] = (data[:, :, 0:3])
    labels[i*len(data):(i+1)*len(data),:] = label[:, :]


pcd_labels = keras.utils.to_categorical(labels, num_classes=11) # 12 to 10


labely = labels.reshape(1*NUM_FRAMES,1024,1) # 192 to 48 to 248 
label_str = np.zeros((1*NUM_FRAMES, 1024, 1)) # 4 * NUM_FRAMES
cats = label_str.astype(str)

classes = ['POI_box','POI_tape','nPOI_tape','POI_extension','nPOI_extension','POI_hammer',
             'nPOI_hammer', 'POI_pliers', 'nPOI_pliers','POI_cutter', 'nPOI_cutter']

for i in range(1*NUM_FRAMES): # 192 to 48
    for j in range(1024):
        cats[i][j] = classes[int(labely[i][j])]
        
cats = cats.reshape(1*NUM_FRAMES, 1024) # 192 to 48

######################################################################################################################

VAL_SPLIT = 0.0
NUM_SAMPLE_POINTS = 1024 # why not 2046
BATCH_SIZE = 32 # well
EPOCHS = 30
INITIAL_LR = 1e-3

######################################################################################################################

for index in tqdm(range(len(pcds))):
    current_point_cloud = pcds[index]
    current_label_cloud = pcd_labels[index]
    current_labels = cats[index]
    num_points = len(current_point_cloud)
    
    # Randomly sampling respective indices.
    sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
    # Sampling points corresponding to sampled indices.
    sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
    # Sampling corresponding one-hot encoded labels.
    sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
    # Sampling corresponding labels for visualization.
    sampled_labels = np.array([current_labels[i] for i in sampled_indices])
    # Normalizing sampled point cloud.
    norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
    norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    pcds[index] = norm_point_cloud
    pcd_labels[index] = sampled_label_cloud
    cats[index] = sampled_labels

######################################################################################################################

def load_data(point_cloud_batch, label_cloud_batch):
    point_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    label_cloud_batch.set_shape([NUM_SAMPLE_POINTS, len(classes)]) # removed + 1
    return point_cloud_batch, label_cloud_batch


def augment(point_cloud_batch, label_cloud_batch):
    noise = tf.random.uniform(
        tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float64
    )
    point_cloud_batch += noise[:, :, :3]
    return point_cloud_batch, label_cloud_batch


def generate_dataset(point_clouds, label_clouds, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = (
        dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training
        else dataset
    )
    return dataset


split_index = int(len(pcds) * (1 - VAL_SPLIT))
train_point_clouds = pcds[:split_index]
train_label_cloud = pcd_labels[:split_index]
total_training_examples = len(train_point_clouds)

val_point_clouds = pcds[split_index:]
val_label_cloud = pcd_labels[split_index:]

# print("Num train point clouds:", len(train_point_clouds))
# print("Num train point cloud labels:", len(train_label_cloud))
# print("Num val point clouds:", len(val_point_clouds))
# print("Num val point cloud labels:", len(val_label_cloud))

train_dataset = generate_dataset(train_point_clouds, train_label_cloud)
val_dataset = generate_dataset(val_point_clouds, val_label_cloud, is_training=False)

# print("Train Dataset:", train_dataset)
# print("Validation Dataset:", val_dataset)


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


######################################################################################################################

def visualize_data(point_cloud, labels):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    for index, label in enumerate(classes):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5)
        except IndexError:
            pass
    ax.legend()
    plt.show()


# visualize_data(pcds[20], cats[20])
# visualize_data(pcds[27], cats[27])


######################################################################################################################


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config


######################################################################################################################


def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


######################################################################################################################


def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)


######################################################################################################################


x, y = next(iter(train_dataset))

num_points = x.shape[1]
num_classes = y.shape[-1]

segmentation_model = get_shape_segmentation_model(num_points, num_classes)
segmentation_model.summary()

training_step_size = total_training_examples // BATCH_SIZE
total_training_steps = training_step_size * EPOCHS

lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[training_step_size * 15, training_step_size * 15],
    values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
)

steps = tf.range(total_training_steps, dtype=tf.int32)
lrs = [lr_schedule(step) for step in steps]

######################################################################################################################

def run_experiment():

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    checkpoint_filepath = "checkpoint/cp.ckpt"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    # history = segmentation_model.fit(
    #     train_dataset,
    #     validation_data=val_dataset,
    #     epochs=epochs,
    #     callbacks=[checkpoint_callback],
    # )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model


segmentation_model = run_experiment()


segmentation_model.evaluate(train_dataset)

######################################################################################################################