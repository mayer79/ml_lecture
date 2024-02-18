from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin


class KerasRegressor(RegressorMixin):
    """
    A wrapper class for a keras model.

    Parameters
    ----------
    regressor : object
        A keras regressor object that has already been fit to data.

    Methods
    -------
    predict(X)
        Make predictions for the given input data X.

    fit(*args, **kwargs)
        Not used.
    """

    def __init__(self, estimator):
        self._estimator = estimator
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return self._estimator.predict(X, verbose=0, batch_size=10000).flatten()


class ColumnSplitter(BaseEstimator, TransformerMixin):
    """
    Transformer that splits a pandas.Dataframe into a dict of numpy arrays.

    Parameters
    ----------
    feature_dict : dictionary
        The keys define the keys of the dict holding the dataframe pieces, and
        the values the corresponding feature column names.

    Methods
    -------
    transform(X)
        Splits input dataframe X into dict of numpy arrays defined by `feature_dict`.

    fit(*args, **kwargs)
        Not used.
    """

    def __init__(self, feature_dict):
        self._feature_dict = feature_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = dict()
        for key, value in self._feature_dict.items():
            out[key] = X[value].to_numpy()
        return out
