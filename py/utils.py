from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


class KerasRegressor(RegressorMixin, BaseEstimator):
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
        self.estimator = estimator
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        # Bypass Pipeline.predict(): its kwarg routing calls __sklearn_tags__ on
        # every step, which the raw Keras model does not implement
        Xt = self.estimator[:-1].transform(X)
        return self.estimator[-1].predict(Xt, verbose=0, batch_size=20_000).flatten()


class ColumnSplitter(TransformerMixin, BaseEstimator):
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
        self.feature_dict = feature_dict

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        out = {}
        for key, value in self.feature_dict.items():
            out[key] = X[value].to_numpy()
        return out
