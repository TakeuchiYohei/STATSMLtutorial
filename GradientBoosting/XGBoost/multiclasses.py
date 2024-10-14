import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from mygbdt import LogisticLoss, GBDTMulticlass

def make_data():
    """
    [ 作成するデータ ]
    * 3 x 3 のマス目上に二次元正規分布を並べたもの。
    * 3クラス分類問題のためのデータで、それぞれの正規分布がどれかのクラスに対応している。
    """
    np.random.seed(seed=9)
    X = np.empty((900, 2))
    y = np.empty(900)
    for i in range(9):
        cx, cy = i // 3, i % 3
        X[100*i:100*(i+1), :] = 0.15 * np.random.randn(100, 2) + (cx, cy)
        y[100*i:100*(i+1)] = np.random.randint(3)
    return X, y

def train_my_model(X, y):
    my_model = GBDTMulticlass(LogisticLoss(), n_estimators=10, reg_lambda=1, gamma=1, learning_rate=1)
    my_model.fit(X, y)
    return my_model

def train_xgb_model(X, y):
    xgb_model = XGBClassifier(
            objective='multi:softmax', n_estimators=10, reg_lambda=1, gamma=1, learning_rate=1,
            max_depth=None, tree_method='exact', min_child_weight=0,
        )
    xgb_model.fit(X, y)
    return xgb_model

def visualize_result(X, y, my_model, xgb_model):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))

    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    ## my_model
    axes[0].contourf(xx, yy, my_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.4)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    axes[0].set_title(f'My model on train data')

    ## xgb_model
    axes[1].contourf(xx, yy, xgb_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), alpha=0.4)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    axes[1].set_title(f'XGBoost on train data')

    fig.savefig('figures/multiclass_experiment.png', bbox_inches='tight')

if __name__ == '__main__':
    X, y = make_data()
    my_model = train_my_model(X, y)
    xgb_model = train_xgb_model(X, y)
    visualize_result(X, y, my_model, xgb_model)