import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path


import dataget
import dicto
import pandas as pd
import typer
import numpy as np
from plotly import graph_objects as go
import pickle
from sklearn.model_selection import train_test_split
import skorch

# from .
from . import estimator
from .generic_transformer import GenericTransformer


def main(
    params_path: Path = Path("training/params.yml"),
    cache: bool = False,
    viz: bool = False,
    debug: bool = False,
    toy: bool = False,
):
    if debug:
        import debugpy

        print("Waiting debuger....")
        debugpy.listen(("localhost", 5678))
        debugpy.wait_for_client()

    params = dicto.load(params_path)

    train_cache = Path("cache/train.csv")
    test_cache = Path("cache/test.csv")
    transformer_cache = Path("cache/transformer.pkl")

    if cache and train_cache.exists():
        df_train = pd.read_csv(train_cache)
        df_test = pd.read_csv(test_cache)
        transformer = pickle.load(transformer_cache.open("rb"))
    else:
        df, df_real = dataget.kaggle(competition="cat-in-the-dat-ii").get(
            files=["train.csv", "test.csv"]
        )

        df.drop(columns=["id"], inplace=True)

        df_train, df_test = estimator.split(df, params)

        if toy:
            df_train = df_train.sample(n=1000)
            df_test = df_test.sample(n=1000)

        transformer = GenericTransformer(
            categorical=params.categorical, numerical=params.numerical,
        )

        df_train = transformer.fit_transform(df_train)
        df_test = transformer.transform(df_test)

        df_train.to_csv(train_cache, index=False)
        df_test.to_csv(test_cache, index=False)
        pickle.dump(transformer, transformer_cache.open("wb"))

    print(df_train)
    print(df_test)

    ds_train = estimator.get_dataset(df_train, params, "train")
    ds_test = estimator.get_dataset(df_test, params, "test")

    print(ds_train[:10])
    print(ds_test[:10])

    model = estimator.get_model(
        params, n_categories=transformer.n_categories, numerical=[]
    )
    print(model)
    exit()

    net = skorch.NeuralNet(model,)

    model.summary()

    print(ds_train)

    model.fit(
        ds_train,
        epochs=params.epochs,
        steps_per_epoch=params.steps_per_epoch,
        validation_data=ds_test,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Path("summaries") / Path(model.name)), profile_batch=0
            )
        ],
    )

    # Export to saved model
    save_path = f"models/{model.name}"
    model.save(save_path)

    print(f"{save_path=}")

    vizualize(df_train, df_test, model)


def vizualize(df_train, df_test, model):
    fig = go.Figure(
        [
            go.Scatter(
                x=df_train.x0,
                y=df_train.x1,
                marker=go.scatter.Marker(
                    color=df_train.y, size=8, line_width=2, line_color="DarkSlateGrey",
                ),
                mode="markers",
                name="train",
            ),
            go.Scatter(
                x=df_test.x0,
                y=df_test.x1,
                marker=go.scatter.Marker(
                    color=df_test.y, size=12, line_width=2, line_color="DarkSlateGrey",
                ),
                mode="markers",
                name="test",
            ),
        ]
    )
    fig.update_layout(template="simple_white")
    xx, yy, zz = decision_boundaries(df_train[["x0", "x1"]].to_numpy(), model)
    xx = xx[0]
    yy = yy[:, 0]
    fig.add_trace(go.Contour(x=xx, y=yy, z=zz, opacity=0.5))
    fig.show()


def decision_boundaries(X, model, n=20):

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    print(x_min, x_max)
    print(y_min, y_max)

    hx = (x_max - x_min) / n
    hy = (y_max - y_min) / n
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max + hx, hx), np.arange(y_min, y_max + hy, hy)
    )

    # Obtain labels for each point in mesh using the model.
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)

    Z = model(dict(x0=points[:, 0], x1=points[:, 1])).numpy()
    Z = (Z > 0.5).astype(np.int32)

    zz = Z.reshape(xx.shape)

    return xx, yy, zz


if __name__ == "__main__":
    typer.run(main)
