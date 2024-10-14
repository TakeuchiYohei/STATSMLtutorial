import numpy as np
import graphviz

class SquaredError:
    def loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def grad(self, y, y_pred):
        return -2 * (y - y_pred)
    
    def hess(self, y, y_pred):
        return np.ones_like(y_pred) * 2

class LogisticLoss:
    def loss(self, y, y_pred):
        return y * np.log(1 + np.exp(-y_pred)) + (1 - y) * np.log(1 + np.exp(y_pred))
    
    def grad(self, y, y_pred):
        return - y * np.exp(-y_pred) / (1 + np.exp(-y_pred)) + (1 - y) * np.exp(y_pred) / (1 + np.exp(y_pred))
    
    def hess(self, y, y_pred):
        return (
            y * np.exp(-y_pred) / (1 + np.exp(-y_pred))
            - y * np.exp(-2*y_pred) / (1 + np.exp(-y_pred)) ** 2
            + (1 - y) * np.exp(y_pred) / (1 + np.exp(y_pred))
            - (1 - y) * np.exp(2*y_pred) / (1 + np.exp(y_pred)) ** 2
        )

class DecisionTree:
    def __init__(self, reg_lambda, gamma):
        """
        reg_lambda: float, L2 regularization term on weights
        gamma: float, Factor to limit the number of leaves
        """
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def fit(self, X, y, grad, hess):
        """
        X: np.array, shape [n_samples, n_features], traing values
        y: np.array, shape [n_samples], target values
        grad: np.array, shape [n_samples], gradient of loss function
        hess: np.array, shape [n_samples], hessian of loss function
        """
        # Caluculate leaf values
        n_cols = X.shape[1]
        best_gain = 0
        for i in range(n_cols):
            # Obtain the optimal division (function) of a variable
            threshold, gain = self.find_best_split(X[:, i], y, grad, hess)
            if best_gain < gain:
                best_gain = gain
                best_threshold = threshold
                best_col = i

        # If no partition with 0 < Gain is found, it is assumed to be a leaf node
        if best_gain == 0:
            self.is_leaf = True
            self.score = self.calc_best_score(grad, hess)
            return self
        
        else:
            self.is_leaf = False
            self.column_idx = best_col
            self.threshold = best_threshold
            x_best = X[:, best_col]
            is_left = x_best < best_threshold
            is_right = best_threshold <= x_best
            self.left = DecisionTree(self.reg_lambda, self.gamma).fit(X[is_left], y[is_left], grad[is_left], hess[is_left])
            self.right = DecisionTree(self.reg_lambda, self.gamma).fit(X[is_right], y[is_right], grad[is_right], hess[is_right])
            return self

    def find_best_split(self, x, y, grad, hess):
        """
        x: np.array, shape [n_samples], feature values
        y: np.array, shape [n_samples], target values
        grad: np.array, shape [n_samples], gradient of loss function
        hess: np.array, shape [n_samples], hessian of loss function
        """
        # Order by x value
        sorted_idx = x.argsort()
        x = x[sorted_idx]
        y = y[sorted_idx]
        grad = grad[sorted_idx]
        hess = hess[sorted_idx]
        
        cgrad = np.cumsum(grad)
        chess = np.cumsum(hess)

        best_threshold = None
        best_gain = 0
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                continue
            gl, hl = cgrad[i-1], chess[i-1]
            gr, hr = cgrad[-1] - cgrad[i-1], chess[-1] - chess[i-1]
            gain = self.calc_gain(gl, hl, gr, hr)
            if best_gain < gain:
                best_gain = gain
                best_threshold = (x[i] + x[i-1]) / 2

        return best_threshold, best_gain


    def predict(self, X):
        """
        X: np.array, shape [n_samples, n_features], traing values
        """
        # If it is a leaf node, return the score
        if self.is_leaf:
            return np.zeros(X.shape[0]) + self.score
        # If it is not a leaf node, divide it into left and right and predict
        else:
            x = X[:, self.column_idx]
            is_left = x < self.threshold
            is_right = self.threshold <= x
            y_pred = np.empty_like(x)
            y_pred[is_left] = self.left.predict(X[is_left])
            y_pred[is_right] = self.right.predict(X[is_right])
            return y_pred

    def calc_best_score(self, gj, hj):
        """
        gj: np.array, shape [n_samples], gradient of loss function
        hj: np.array, shape [n_samples], hessian of loss function
        """

        return -1 * np.sum(gj) / (np.sum(hj) + self.reg_lambda)

    def calc_gain(self, gl, hl, gr, hr):
        """
        Caclulate gain of split
        gl: float, sum of gradient of left node
        hl: float, sum of hessian of left node
        gr: float, sum of gradient of right node
        hr: float, sum of hessian of right node
        """
        Gl, Hl, Gr, Hr = gl.sum(), hl.sum(), gr.sum(), hr.sum()
        return (
            Gl**2 / (Hl + self.reg_lambda) 
            + Gr**2 / (Hr + self.reg_lambda)
            - (Gl + Gr)**2 / (Hl + Hr + self.reg_lambda)
        ) / 2 - self.gamma


class GBDT:
    def __init__(self, objective, n_estimators=100, reg_lambda=0, gamma=0, learning_rate=1):
        """
        n_trees: int, number of trees
        reg_lambda: float, L2 regularization term on weights
        gamma: float, Factor to limit the number of leaves
        learning_rate: float, learning rate
        """
        self.objective = objective
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        X: np.array, shape [n_samples, n_features], traing values
        y: np.array, shape [n_samples], target values
        """
        self.trees = []
        y_pred = np.zeros(len(X))
        for _ in range(self.n_estimators):
            grad = self.objective.grad(y, y_pred)
            hess = self.objective.hess(y, y_pred)
            tree = DecisionTree(self.reg_lambda, self.gamma)
            tree.fit(X, y, grad, hess)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        """
        X: np.array, shape [n_samples, n_features], traing values
        """
        y_pred = np.zeros(len(X))
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
    

def show_tree(tree, filename='tree'):
    def traverse(tree, digraph, node_idx):
        if tree.is_leaf:
            digraph.node(str(node_idx), f'leaf={tree.score:.9g}')
            return node_idx
        
        # left
        digraph.node(str(node_idx), f'f{tree.column_idx}<{tree.threshold:.9g}')
        digraph.edge(str(node_idx), str(node_idx + 1), label='Yes', color='blue')
        left_idx = traverse(tree.left, digraph, node_idx + 1)
        
        # right
        digraph.edge(str(node_idx), str(left_idx + 1), label='No', color='red')
        right_idx = traverse(tree.right, digraph, left_idx + 1)
        return right_idx

    digraph = graphviz.Digraph()
    traverse(tree, digraph, 0)
    
    # PNGとして保存
    digraph.format = 'png'
    digraph.render(filename, cleanup=True)
    return digraph

class GBDTMulticlass:
    def __init__(self, objective, n_estimators=100, reg_lambda=0, gamma=0, learning_rate=1):
        """
        n_trees: int, number of trees
        reg_lambda: float, L2 regularization term on weights
        gamma: float, Factor to limit the number of leaves
        learning_rate: float, learning rate
        """
        self.objective = objective
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.learning_rate = learning_rate

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.models = []
        for k in range(n_classes):
            y_k = (y == k).astype(int)
            model = GBDT(self.objective, self.n_estimators, self.reg_lambda, self.gamma, self.learning_rate)
            model.fit(X, y_k)
            self.models.append(model)

    def predict(self, X):
        n_classes = len(self.models)
        y_pred = np.empty((len(X), n_classes))
        for k, model in enumerate(self.models):
            y_pred[:, k] = model.predict(X)
        return np.argmax(y_pred, axis=1)
            
