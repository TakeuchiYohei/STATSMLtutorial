import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import xgboost
from xgboost import XGBClassifier
import xgboost as xgb
from mygbdt import GBDT, LogisticLoss, show_tree
import sys

n_estimators = int(sys.argv[1])

def train_my_model(X, y):
    my_model = GBDT(LogisticLoss(), n_estimators=n_estimators, reg_lambda=1, gamma=1, learning_rate=1)
    my_model.fit(X, y)
    show_tree(my_model.trees[0], f'figures/my_model_tree0_n_estimators{n_estimators}')
    return my_model

def train_xgb_model(X, y):
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=5,
        reg_lambda=1,
        gamma=1,
        learning_rate=1,
        max_depth=None,
        tree_method='exact',
        min_child_weight=0
        )
    xgb_model.fit(X, y)
    xgb.plot_tree(xgb_model.get_booster(), num_tree=0)
    plt.savefig('figures/xgb_model_tree0.png')
    return xgb_model

def visualize_result(X, y, my_model, xgb_model):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    ## my_model
    axes[0].contourf(xx, yy, (0 < my_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)).astype(int), alpha=0.4, cmap='bwr')
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    axes[0].set_title(f'My model on train data')

    ## xgb_model
    axes[1].contourf(xx, yy, (xgb_model.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape), alpha=0.4, cmap='bwr')
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    axes[1].set_title(f'XGBoost on train data')
    
    fig.savefig('figures/binary_experiment.png', bbox_inches='tight')


if __name__ == '__main__':
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=0, factor=0.2)
    
    my_model = train_my_model(X, y)
    xgb_model = train_xgb_model(X, y)
    
    visualize_result(X, y, my_model, xgb_model)