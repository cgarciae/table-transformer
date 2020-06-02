import tensorflow as tf
import tensorflow_addons as tfa
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from pathlib import Path
import typing as tp
import numpy as np
from sklearn.model_selection import train_test_split
import shutil


def split(df, params):
    df_train, df_test = train_test_split(df, test_size=params.train_split)

    return df_train, df_test


def balance(df, params):

    df["original_steering"] = df["steering"]

    # bins = np.linspace(-0.9, 0.9, params.n_buckets)
    # df["bucket"] = np.digitize(df["steering"], bins)
    # df, _ = RandomOverSampler().fit_resample(df, df["bucket"])

    df["flipped"] = 0

    df_flipped = df.copy()
    df_flipped["flipped"] = 1

    df = pd.concat([df, df_flipped])

    return df


def as_single_image(df, column, params):

    df = df.copy()
    df["image_path"] = df[column]
    df.drop(columns=["center", "left", "right"], inplace=True)

    return df


def preprocess(df, params, mode):

    if mode == "train":
        df = balance(df, params)

        df_left = as_single_image(df, "left", params)
        df_right = as_single_image(df, "right", params)
        df_center = as_single_image(df, "center", params)

        df_left["steering"] += params.steering_correction
        df_right["steering"] -= params.steering_correction

        df = pd.concat([df_center, df_left, df_right], ignore_index=True)

        df.loc[df["flipped"] == 1, "steering"] = df["steering"][df["flipped"] == 1] * -1

    else:
        df = as_single_image(df, "center", params)
        df["flipped"] = 0

    df = df.sample(frac=1)

    return df


def get_dataset(df, params, mode):
    dataset = tf.data.Dataset.from_tensor_slices(
        {col: df[col].to_numpy() for col in df}
    )

    if mode == "train":
        dataset = dataset.repeat()

    dataset = dataset.shuffle(100)

    dataset = dataset.map(lambda d: (dict(x0=d["x0"], x1=d["x1"]), d["y"]))
    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def get_model(params) -> tf.keras.Model:

    x0 = tf.keras.Input(shape=(1,), name="x0")
    x1 = tf.keras.Input(shape=(1,), name="x1")

    inputs = [x0, x1]

    # x0 embeddings

    # x0 = tf.keras.layers.Dense(10, activation="relu")(x0)
    # x0 = x0[:, None, :]

    # x1 = tf.keras.layers.Dense(10, activation="relu")(x1)
    # x1 = x1[:, None, :]

    x = tf.concat([x0, x1], axis=1)

    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)

    # x = AddPositionalEmbeddings()(x)
    # x = SelfAttentionBlock(10, head_size=10, num_heads=8)(x)
    # x = SelfAttentionBlock(10, head_size=10, num_heads=8)(x)
    # x = SelfAttentionBlock(10, head_size=10, num_heads=8)(x)
    # x = AttentionPooling(10, n_queries=1, head_size=10, num_heads=8)(x)

    # x = x[:, 0]
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(1, activation="sigmoid", name="y")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name="tabular_attention")

    return model


class AddPositionalEmbeddings(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embeddings: tp.Optional[tf.Variable] = None

    def build(self, input_shape):

        input_shape = list(input_shape)

        self.embeddings = self.add_weight(
            name="key_kernel", shape=[1] + input_shape[1:]
        )

        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.embeddings


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
        head_size: int = 16,
        num_heads: int = 3,
        dropout: float = 0.0,
        activation: tp.Union["str", tp.Callable] = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.mha = tfa.layers.MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, dropout=dropout
        )
        self.dense = tf.keras.layers.Dense(output_size, activation=activation)

    def call(self, inputs):

        x = self.mha([inputs, inputs])
        x = self.dense(x)

        return x


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: int,
        n_queries: int,
        head_size: int = 16,
        num_heads: int = 3,
        dropout: float = 0.0,
        activation: tp.Union["str", tp.Callable] = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_queries = n_queries
        self.mha = tfa.layers.MultiHeadAttention(
            head_size=head_size, num_heads=num_heads, dropout=dropout
        )
        self.dense = tf.keras.layers.Dense(output_size, activation=activation)
        self.query: tp.Optional[tf.Variable] = None

    def build(self, input_shape):

        num_features = input_shape[-1]

        self.query = self.add_weight(
            name="key_kernel", shape=[1, self.n_queries, num_features]
        )

        super().build(input_shape)

    def call(self, inputs):

        query = tf.tile(
            self.query, [tf.shape(inputs)[0]] + [1] * (len(inputs.shape) - 1)
        )

        x = self.mha([query, inputs])
        x = self.dense(x)

        return x
