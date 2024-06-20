import numpy as np
np.random.seed(42)
import pandas as pd
df_train = pd.read_csv('G:/DeepLearningTmp/train.csv')
df_train = df_train.drop('id', axis=1)
df_train.head()
# df_train.columns
df_train['MonsoonIntensity'].value_counts()
df_train['TopographyDrainage'].value_counts()
df_train['RiverManagement'].value_counts()
df_train['Deforestation'].value_counts()
df_train['Urbanization'].value_counts()
df_train['ClimateChange'].value_counts()
df_train['DamsQuality'].value_counts()
df_train['Siltation'].value_counts()
df_train['AgriculturalPractices'].value_counts()
df_train['Encroachments'].value_counts()
df_train['IneffectiveDisasterPreparedness'].value_counts()
df_train['DrainageSystems'].value_counts()
df_train['CoastalVulnerability'].value_counts()
df_train['Landslides'].value_counts()
df_train['Watersheds'].value_counts()
df_train['DeterioratingInfrastructure'].value_counts()
df_train['PopulationScore'].value_counts()
df_train['WetlandLoss'].value_counts()
df_train['InadequatePlanning'].value_counts()
df_train['PoliticalFactors'].value_counts()
X = df_train.drop(['FloodProbability'], axis=1)
y = df_train['FloodProbability']
from sklearn.model_selection import cross_validate
mets = ['neg_mean_squared_error', 'r2']
def print_metrics(scores):
    print("评估指标 ========================")
    for s in scores:
        print(
            "%s 平均值 => %f === %f 标准差."
            % (s,np.average(scores[s]), np.std(scores[s]))
        )
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

pipeline_dummy_mean = Pipeline([
    ('scaler', StandardScaler()),
    ('dummy_regressor', DummyRegressor(strategy="mean"))
])
scores = cross_validate(pipeline_dummy_mean, X, y, cv=10, scoring=mets)
print_metrics(scores)
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

pipeline_dummy_median = Pipeline([
    ('scaler', StandardScaler()),
    ('dummy_regressor', DummyRegressor(strategy="median"))
])

scores = cross_validate(pipeline_dummy_median, X, y, cv=10, scoring=mets)
print_metrics(scores)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd_regressor', SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        fit_intercept=True,
        max_iter=10000,
        verbose=1,
        learning_rate="optimal",
        eta0=0.01,
        power_t=0.40,
        n_iter_no_change=50
    ))
])

scores = cross_validate(pipeline, X, y, cv=10, scoring=mets)
print_metrics(scores)
df_test = pd.read_csv('G:/DeepLearningTmp/test.csv')
df_test.head()
pipeline.fit(X, y)

df_test['FloodProbability'] = pipeline.predict(df_test.drop('id', axis=1))
df_test.head()
df_test = df_test[['id', 'FloodProbability']]
df_test.head()
df_test.to_csv('G:/DeepLearningTmp/result.csv', index=False)