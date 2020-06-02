from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class GaussianNoise(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        pass


class LabelEncoder(LabelEncoder):
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
        X = np.append(X, self._unknown).reshape(-1)

        return super().fit(X)

    def transform(self, X, y=None):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        items = set(np.unique(X))
        classes_ = set(self.classes_)
        unknown = items - classes_

        X = np.vectorize(lambda x: self._unknown if x in unknown else x)(X)
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
