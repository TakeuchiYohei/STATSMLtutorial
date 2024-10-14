# XGBoost
## Equation
Object function is

$$
\text{n: the number of samples, t:the number of trees, }
y_i: \text{target values, } 
\hat{y}_i: \text{predicted values, }
f_t: \text{function we need to learn, }
\omega(f_t) \text{:regularization}
$$



$$
\text{n: the number of samples, t:the number of trees, }
y_i: \text{target values, } 
\hat{y}_i: \text{predicted values, }
f_t: \text{function we need to learn, }
\omega(f_t) \text{:regularization}
$$


For each tree,


$$
\text{obj}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \omega(f_t) + \text{constant}
$$


Taylor expansion,


$$
g_i = \frac{\partial}{\partial \hat{y}_i^{(t-1)}} l(y_i, \hat{y}_i^{(t-1)})
$$

$$
h_i = \frac{\partial^2}{\partial \hat{y}_i^{(t-1)^2}} l(y_i, \hat{y}_i^{(t-1)})
$$


$$
\text{obj}^{(t)} = \sum_{i=1}^{n} \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \omega(f_t) + \text{constant}
$$

$$
\omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

$$
\text{obj}^{(t)} \approx \sum_{j=1}^{T} \left[ \left( \sum_{i \in I_j} g_i \right) w_j + \frac{1}{2} \left( \sum_{i \in I_j} h_i + \lambda \right) w_j^2 \right] + \gamma T
$$


$$
G_j = \sum_{i \in I_j} g_i \quad \text{and} \quad H_j = \sum_{i \in I_j} h_i:
$$

$$
\text{obj}^{(t)} = \sum_{j=1}^{T} \left[ G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 \right] + \gamma T
$$

Therefore,

$$
w_j^* = - \frac{G_j}{H_j + \lambda}
$$

$$
\text{obj}^* = - \frac{1}{2} \sum_{j=1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
$$

## Memo
- Gradient and hessian are derived from the loss of y and $\hat{y}_i^{(t-1)}$

## Script
- twoclasses.py: 2-class classification task
- regressor.py: regression task
- multiclasses.py: multi-classes classification task

## References
English
- Paper (https://arxiv.org/abs/1603.02754)
- Official document (https://xgboost.readthedocs.io/en/stable/tutorials/model.html)

Japanese
- https://zenn.dev/skwbc/articles/implement_gradient_boosting
- https://qiita.com/triwave33/items/aad60f25485a4595b5c8