import joblib
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

df = fetch_openml(
    data_id=40945,
    as_frame=True,
    parser="pandas"
).frame


df = df[
    [
        "survived",
        "pclass",
        "sex",
        "age",
        "fare",
        "sibsp",
        "parch",
        "embarked"
    ]
]


df["survived"] = df["survived"].astype(int)


print("Dimensão do dataset:", df.shape)

print("\nValores ausentes:")
print(df.isnull().sum())


X = df.drop(columns="survived")

y = df["survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nDados de treinamento:", X_train.shape)
print("Dados de teste:", X_test.shape)

X_train.isna().sum().sort_values(ascending=False)
print("\nValores ausentes no conjunto de treinamento:")
print(X_train.isna().sum().sort_values(ascending=False))
print("\nValores ausentes no conjunto de teste:")
print(X_test.isna().sum().sort_values(ascending=False))

num_cols = ["age" , "fare" , "sibsp" , "parch"]
cat_cols = ["pclass" , "sex" , "embarked"]

assert set(X_train.columns) == set(num_cols + cat_cols), "As colunas não correspondem ao esperado."

print("\nColunas numéricas:", num_cols)
print("Colunas categóricas:", cat_cols)

print(set(X_train.columns) == set(num_cols + cat_cols))

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


num_pipe = Pipeline([
    ("imputer" , SimpleImputer(strategy="mean")),
    ("scaler" , StandardScaler())
])

from sklearn.preprocessing import OneHotEncoder

cat_pipe = Pipeline([
    ("imputer" , SimpleImputer(strategy="most_frequent")),
    ("onehot" , OneHotEncoder(handle_unknown="ignore" , sparse_output=False))
])

from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("num" , num_pipe , num_cols),
    ("cat" , cat_pipe , cat_cols)
])

prep.fit_transform(X_train).shape
print("\nShape do conjunto de treinamento após o pré-processamento:", prep.fit_transform(X_train).shape)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("prep" , prep),
    ("clf" , LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipe , X_train , y_train , cv=5 , scoring="accuracy")
print("\nAcurácia média do modelo de Regressão Logística:", scores.mean().round(4))
print("Desvio padrão da acurácia:", scores.std().round(4))

pipe.fit(X_train , y_train)
pipe.score(X_test , y_test)

print("\nAcurácia do modelo de Regressão Logística no conjunto de teste:", pipe.score(X_test , y_test))

names = pipe.named_steps["prep"].get_feature_names_out()
print("\nNomes das features após o pré-processamento:")
coefa = pipe.named_steps["clf"].coef_[0]
pd.Series(coefa , index=names).sort_values()
print(pd.Series(coefa , index=names).sort_values())

from sklearn.model_selection import GridSearchCV

grid = {
    "prep__num__imputer__strategy" : ["mean" , "median"],
    "prep__cat__imputer__strategy" : ["most_frequent" , "constant"],
    "clf__C" : [0.01 , 0.1 , 1 , 10 , 100],
}

gs = GridSearchCV(pipe , grid , cv=5 , scoring="accuracy" , n_jobs=-1)
gs.fit(X_train , y_train)
gs.best_params_, round(gs.best_score_ , 4)
print("\nMelhores parâmetros encontrados:", gs.best_params_)
print("Melhor acurácia média obtida:", round(gs.best_score_ , 4))

import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_fare = Pipeline([
    ("imputer" , SimpleImputer(strategy="mean")),
    ("log" , FunctionTransformer(np.log1p , feature_names_out="one-to-one")),
    ("scaler" , StandardScaler())
])

print("\nPré-processamento da coluna 'fare' com log-transformação:")
print(log_fare.fit_transform(X_train[["fare"]]).shape)

from sklearn.base import BaseEstimator, TransformerMixin

class TamanhoFamilia(BaseEstimator , TransformerMixin):
    def fit(self , X , y=None):
        return self

    def transform(self , X):
        familia = X["sibsp"] + X["parch"] + 1
        return familia.to_frame("familia")

    def get_feature_names_out(self , input_features=None):
        return np.array(["familia"])

    import joblib

joblib.dump(pipe, "titanic_pipeline.joblib")

modelo = joblib.load("titanic_pipeline.joblib")

nova = pd.DataFrame([{
    "pclass": 3,
    "sex": "male",
    "age": None,
    "sibsp": 0,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S"
}])

modelo.predict_proba(nova)[0, 1].round(3)
print("\nProbabilidade de sobrevivência para o novo passageiro:", modelo.predict_proba(nova)[0, 1].round(3))
