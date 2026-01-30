import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from datetime import datetime

def train_multitarget_baseline(model, is_classifier, X_train, X_test, y_embed_train, y_embed_test, verbose=True):
    """
    Trains a multitarget model (regressor or classifier) on embedding targets and evaluates train + test performance.

    Returns
    -------
    y_pred_test: np.array
    metric_per_item: np.array with MSE per target (regression) or accuracy per target (classification)
    """

    # The MultiOutputClassifier is an artifact from before, but was left in because 
    #    it does add additional functionality and does not hurt.

    # Wrap the model in a multi-output wrapper
    if is_classifier:
        model = MultiOutputClassifier(model)
    else:
        model = MultiOutputRegressor(model)

    # Fit
    model.fit(X_train, y_embed_train)

    # Predict train and test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate
    if is_classifier:
        metric_train = np.mean(y_pred_train == y_embed_train, axis=0)
        metric_test = np.mean(y_pred_test == y_embed_test, axis=0)
        metric_name = "accuracy"
    else:
        metric_train = mean_squared_error(y_embed_train, y_pred_train, multioutput='raw_values')
        metric_test = mean_squared_error(y_embed_test, y_pred_test, multioutput='raw_values')
        metric_name = "MSE"

    if verbose:
        print("\n\n", "#"*40, f"{type(model.estimators_[0]).__name__} Multitarget {'Classifier' if is_classifier else 'Regressor'}", "#"*40)
        print(f"Train {metric_name} per embedding:", metric_train)
        print(f"Test {metric_name} per embedding:", metric_test)
        print(f"Average train {metric_name}:", metric_train.mean())
        print(f"Average test {metric_name}:", metric_test.mean())
    
    json_metrics = {
        "model": f"{type(model.estimators_[0]).__name__}" + " Multitarget Regressor",
        "finish time": datetime.now().strftime("%d:%m:%Y %H:%M:%S"),
        "train_MSE_array": metric_train.tolist(),
        "test_MSE_array": metric_test.tolist(),
        "avg_train_MSE": metric_train.mean(),
        "avg_test_MSE": metric_test.mean()
    }

    return y_pred_test, json_metrics
