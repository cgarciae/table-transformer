from dataclasses import dataclass
from pathlib import Path
import shutil
import typing as tp

import dicto
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import skorch
import torch
from torch import embedding
import torch.utils


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


def get_dataset(df: pd.DataFrame, params: dicto.Dicto, mode: str):
    return TableDataset(df, params.label_col)


def get_model(
    params: dicto.Dicto, n_categories: tp.Dict[str, int], numerical: tp.List[str]
):
    return TableTransformer(
        categorical={col: n_categories[col] for col in n_categories},
        numerical={col: params.continous_categories for col in numerical},
        embeddings_size=params.embeddings_size * params.n_heads,
        n_heads=params.n_heads,
        num_layers=params.num_layers,
    )


@dataclass
class TableDataset(torch.utils.data.Dataset):
    df: pd.DataFrame
    label_col: str

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return (
            {
                col: torch.as_tensor(self.df.iloc[i][col])
                for col in self.df
                if col != self.label_col
            },
            torch.as_tensor(self.df.iloc[i][self.label_col]),
        )


class TableTransformer(torch.nn.Module):
    def __init__(
        self,
        categorical: tp.Dict[str, int],
        numerical: tp.Dict[str, int],
        embeddings_size: int,
        n_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.categorical = torch.nn.ModuleDict(
            {
                key: torch.nn.Sequential(
                    torch.nn.Embedding(n_categories, embeddings_size),
                    ColumnBias(embeddings_size),
                )
                for key, n_categories in categorical.items()
            }
        )
        self.numerical = torch.nn.ModuleDict(
            {
                key: torch.nn.Sequential(
                    torch.nn.Embedding(n_categories, embeddings_size),
                    ColumnBias(embeddings_size),
                )
                for key, n_categories in numerical.items()
            }
        )
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=embeddings_size,
                dim_feedforward=embeddings_size // 2,
                nhead=n_heads,
            ),
            num_layers=num_layers,
        )

        self.categorical_list = list(categorical.keys())
        self.numerical_list = list(numerical.keys())

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> torch.Tensor:

        categorical = torch.stack(
            [self.categorical[col](inputs[col]) for col in self.categorical_list], dim=1
        )

        numerical = torch.stack(
            [self.numerical[col](inputs[col]) for col in self.numerical_list], dim=1
        )

        x = torch.cat([categorical, numerical], dim=1)

        return x


class ColumnBias(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(1, size))

    def extra_repr(self):
        return f"{self.bias.shape[-1]}"

    def forward(self, x):
        return self.bias + x


if __name__ == "__main__":

    x = torch.rand(10, 5)
    print(x)

    x = ColumnBias(5)(x)
    print(x)

    table_transformer = TableTransformer(categorical=dict(x=(10, 4)), numerical={})

    print(table_transformer)
