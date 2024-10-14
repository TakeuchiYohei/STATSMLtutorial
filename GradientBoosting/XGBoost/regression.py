import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from mygbdt import SquaredError, GBDT
from sklearn.preprocessing import StandardScaler


def make_data():
    np.random.seed(0)
    X = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(X.ravel()) + np.random.randn(len(X)) * 0.1
    return X, y

def train_my_model(X, y):
    my_model = GBDT(SquaredError(), n_estimators=10, reg_lambda=1, gamma=1, learning_rate=1)
    my_model.fit(X, y)
    return my_model

def train_xgb_model(X, y):
    xgb_model = XGBRegressor(
            objective='reg:squarederror', n_estimators=10, reg_lambda=1, gamma=1, learning_rate=1,
            max_depth=None, tree_method='exact', min_child_weight=0,
        )
    xgb_model.fit(X, y)
    return xgb_model

def visualize_result(X, y, my_model, xgb_model):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(15, 5))

    
    axes[0].plot(X, y, 'o', markersize=2, label='Train data')
    axes[0].plot(X.ravel(), np.sin(X.ravel()), label='Best model')
    axes[0].set_title('Train data')
    axes[0].legend()

    # my_model
    x = X.ravel()
    axes[1].plot(x, my_model.predict(X), label='Trained model')
    axes[1].plot(x, np.sin(x), label='Best model')
    axes[1].set_title('My model')
    axes[1].legend()

    # xgb_model
    axes[2].plot(x, xgb_model.predict(X), label='Trained model')
    axes[2].plot(x, np.sin(x), label='Best model')
    axes[2].set_title('XGBoost model')
    axes[2].legend()

    fig.savefig('figures/regression_experiment.png', bbox_inches='tight')

if __name__ == '__main__':
    X, y = make_data()
    my_model = train_my_model(X, y)
    xgb_model = train_xgb_model(X, y)
    visualize_result(X, y, my_model, xgb_model)