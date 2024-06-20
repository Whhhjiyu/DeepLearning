import tempfile
import os

# 创建一个不包含非ASCII字符的临时目录
temp_dir = tempfile.mkdtemp(prefix='joblib_temp_')
# 将临时目录路径分配给JOBLIB_TEMP_FOLDER环境变量
os.environ['JOBLIB_TEMP_FOLDER'] = temp_dir

print(f"Temporary directory set to: {temp_dir}")

# 导入其他必要的库和模块
from lazypredict.Supervised import LazyRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr

from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', 50)
plt.style.use('bmh')

# Ensure all file paths are correct Unicode strings
for dirname, _, filenames in os.walk(r'G:/DeepLearningTmp'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv(r'G:/DeepLearningTmp/train.csv', index_col='id')
test = pd.read_csv(r'G:/DeepLearningTmp/test.csv', index_col='id')
train.info()
test.info()

cols = train.drop('FloodProbability', axis=1).columns.tolist()
for col in cols:
    fig, ax = plt.subplots(figsize=(6, 2))
    max_val = round(train[col].max()) + 1
    train[col].hist(density=True, bins=np.arange(0, max_val, 1), ax=ax)
    plt.xticks(np.arange(0, 20, 1))
    plt.title(col)
    plt.show()

round(train.agg(['min', 'mean', 'median', 'max', 'var', 'std', 'skew']), 2).T
round(test.agg(['min', 'mean', 'median', 'max', 'var', 'std', 'skew']), 2).T

corr = train.drop('FloodProbability', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, linewidth=0.1)
plt.show()

corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, linewidth=0.1)
plt.show()

train.drop('FloodProbability', axis=1).plot(kind='box', vert=False)
plt.title('Boxplot of train variables')
plt.show()

test.plot(kind='box', vert=False)
plt.title('Boxplot of test variables')
plt.show()

for col in cols:
    col_iqr = iqr(train[col])
    Q1, Q3 = np.quantile(train[col], [0.25, 0.75])
    train.loc[train[col] < (Q1 - 1.5 * col_iqr), col] = np.nan
    train.loc[train[col] > (Q3 + 1.5 * col_iqr), col] = np.nan
    train.isna().sum() / train.shape[0]

print('Shape before:', train.shape)
train.dropna(how='any', inplace=True)
print("Shape after:", train.shape)

y = train['FloodProbability']
X = train.drop('FloodProbability', axis=1)
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)
test[test.columns] = scaler.transform(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=987)

regressors = [
    ('bayesian ridge', BayesianRidge()),
    ('elastic net', ElasticNetCV()),
    ('linear_reg', LinearRegression()),
    ('lasso', LassoCV()),
    ('ridge', RidgeCV())
]
evals = {}
for clf, model in regressors:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    evals[clf] = score
evals_df = pd.DataFrame({'model': evals.keys(), 'r2_score': evals.values()})
evals_df.sort_values('r2_score', inplace=True)
print(evals_df)

models = {
    'bayesian_ridge': BayesianRidge(),
    'elastic_net': ElasticNetCV(cv=5),
    'linear_reg': LinearRegression(),
    'lasso': LassoCV(cv=5),
    'ridge': RidgeCV(cv=5)
}

param_grids = {
    'bayesian_ridge': {
        'alpha_1': [1e-6, 1e-5],
        'alpha_2': [1e-6, 1e-4],
        'lambda_1': [1e-6, 1e-5],
        'lambda_2': [1e-6, 1e-5]
    },
    'elastic_net': {},
    'linear_reg': {
        'fit_intercept': [True, False]
    },
    'lasso': {},
    'ridge': {
        'alphas': [(0.1, 1.0, 10.0)]
    }
}

grid_searches = {name: GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
                 for name, (model, param_grid) in zip(models.keys(), zip(models.values(), param_grids.values()))}

best_estimators = {}

for name, gs in grid_searches.items():
    print(f"Running GridSearchCV for {name}")
    gs.fit(X, y)
    print(f"Best parameters for {name}: {gs.best_params_}")
    print(f"Best score for {name}: {gs.best_score_}")
    best_estimators[name] = gs.best_estimator_

for name, estimator in best_estimators.items():
    print(f"Best estimator for {name}: {estimator}")

voting_regressor = VotingRegressor(estimators=[
    ('bayesian_ridge', best_estimators['bayesian_ridge']),
    ('elastic_net', best_estimators['elastic_net']),
    ('linear_reg', best_estimators['linear_reg']),
    ('lasso', best_estimators['lasso']),
    ('ridge', best_estimators['ridge'])
])

voting_regressor.fit(X, y)
y_preds = voting_regressor.predict(test)

submission = pd.DataFrame({'id': test.index, 'FloodProbability': y_preds})
submission.to_csv(r'G:/DeepLearningTmp/submission.csv', index=False)
