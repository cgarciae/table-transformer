from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import typing as tp
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class LabelEncoder(LabelEncoder):
    _unknown: tp.Union[None, str, int, float] = None

    def fit(self, X: pd.Series, y=None):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        if X.dtype == np.float:
            self._unknown = X.min() - 1.0
        elif X.dtype == np.int:
            self._unknown = X.min() - 1
        else:
            self._unknown = "__NA__"

        X.fillna(value=self._unknown, inplace=True)
        X = np.asarray(X)
        np.append(X, self._unknown).reshape(-1)

        return super().fit(X)

    def transform(self, X, y=None):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        X = np.asarray(X)
        X[~np.isin(X, self.classes_)] = self._unknown
        X = X.reshape(-1)

        return super().transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


class OneHotEncoder(OneHotEncoder):
    @classmethod
    def create(cls, *args, unknown="__NA__", **kwargs):
        self = cls(*args, **kwargs)
        self._unknown = unknown
        return self

    def fit(self, X, y=None):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        X = np.array(X)
        X = np.append(X, self._unknown).reshape(-1, 1)

        return super().fit(X)

    def transform(self, X, y=None):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        items = set(np.unique(X))
        classes_ = set(self.categories_[0])
        unknown = items - classes_

        X = np.vectorize(lambda x: self._unknown if x in unknown else x)(X)

        return super().transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)


@dataclass
class GenericTransformer(BaseEstimator, TransformerMixin):
    categorical: tp.List[str]
    numerical: tp.List[str]

    categorical_: tp.Dict[str, LabelEncoder] = field(default_factory=lambda: {})
    numerical_: tp.Dict[str, SimpleImputer] = field(default_factory=lambda: {})

    def fit(self, X, y=None):

        for col in self.categorical:
            self.categorical_[col] = LabelEncoder().fit(X[col])

        for col in self.numerical:
            col_min = X[col].min()
            self.numerical_[col] = SimpleImputer(
                strategy="constant", fill_value=col_min - 1.0
            ).fit(X[col])

    def transform(self, X, y=None):
        for col in self.categorical:
            X[col] = self.categorical_[col].transform(X[col])

        for col in self.numerical:
            X[col] = self.numerical_[col].transform(X[col])

        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    @property
    def n_categories(self) -> tp.Dict[str, int]:
        return {col: len(self.categorical_[col].classes_) for col in self.categorical}
