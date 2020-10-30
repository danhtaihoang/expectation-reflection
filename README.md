Expectation Reflection (ER) is a multiplicative optimization method that trains the interaction weights from features to target according to the ratio of target observations to their corresponding model expectations. This approach completely separates model updates from minimization of a cost function measuring goodness of fit, so that it can take the cost function as an effective stopping criterion of the iteration. This method has advantage in dealing with the problems of small sample sizes (but many features). Using only one hyper-parameter and being able to demonstrate the system mechanism are additional benefits of this method.

In the current version, ER `classification` can work as a binary or multinomial classifier. The extension to `regression` will be appeared shortly.

## Installation
##### From PyPI

```bash
pip install expectation-reflection
```

##### From Repository

```bash
git clone https://github.com/danhtaihoang/expectation-reflection.git
```

## Usage
The implementation of ER is very similar to that of other classifiers in `sklearn`, bassically it consists of the following steps.

* Import the `expectation_reflection` package into your python script:
```python
from expectation_reflection import classification as ER
```

* Select a model:
```python
model = ER.model(max_iter=100,regu=0.01,random_state=1)
```

* Import your `dataset.txt` into python script.
```python
Xy = np.loadtxt('dataset.txt')
```

* Select the features and target from the dataset. If the target is the last column then
```python
X, y = Xy[:,:-1], Xy[:,-1]

```
 
* Import `train_test_split` from `sklearn` to split data into training and test sets:
```python
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state = 1)
```

* Train the model with `(X_train, y_train)` set: 
```python
model.fit(X_train, y_train)
```

* Predict the output class `y_pred` and its probability `p_pred` of a new dataset `X_test`:
```python
y_pred = model.predict(X_test)
print('predicted output:', y_pred)

p_pred = model.predict_proba(X_test)
print('predicted probability:', p_pred)
```

* Intercept and interaction weights:
```python
print('intercept:', model.intercept_)
print('interaction weights:', model.coef_)
```

### Hyper-Parameter Optimization 
ER has only one hyper-parameter, `regu`, which can be optimized by using `GridSearchCV` from `sklearn`:
```python
from sklearn.model_selection import GridSearchCV

model = ER.model(max_iter=100, random_state = 1)

regu = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.]

hyper_parameters = dict(regu=regu)

clf = GridSearchCV(model, hyper_parameters, cv=4, n_jobs=-1, iid='deprecated')

best_model = clf.fit(X_train, y_train)
```

* Best hyper-parameters:
```python
print('best_hyper_parameters:',best_model.best_params_)
```

* Predict the output `y_pred` and its probability `p_pred`:
```python
y_pred = best_model.best_estimator_.predict(X_test)
print('predicted output:', y_pred)

p_pred = best_model.best_estimator_.predict_proba(X_test)
print('predicted probability:', p_pred)
```

### Performance Evaluation
We can measure the model performance by using `metrics` from `sklearn`:

```python
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,\
roc_auc_score,roc_curve,auc

acc = accuracy_score(y_test,y_pred)
print('accuracy:', acc)

precision = precision_score(y_test,y_pred)
print('precision:', precision)

recall = recall_score(y_test,y_pred)
print('recall:', recall)
    
f1score = f1_score(y_test,y_pred)
print('f1score:', f1score)

roc_auc = roc_auc_score(y_test,p_pred) ## note: it is p_pred, not y_pred
print('roc auc:', roc_auc)
```

ROC AUC can be also calculated as 
```python
fp,tp,thresholds = roc_curve(y_test, p_pred, drop_intermediate=False)
roc_auc = auc(fp,tp)
print('roc auc:', roc_auc)
```
## Citation

Please cite the following papers if you use this package in your work:

* [Danh-Tai Hoang, Juyong Song, Vipul Periwal, and Junghyo Jo, Network inference in stochastic systems from neurons to currencies: Improved performance at small sample size, Physical Review E, 99, 023311 (2019)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.023311)

* [Danh-Tai Hoang, Junghyo Jo, and Vipul Periwal, Data-driven inference of hidden nodes in networks, Physical Review E, 99, 042114 (2019)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.99.042114)
