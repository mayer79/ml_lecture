from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted


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
