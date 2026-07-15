#%% 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, fbeta_score
import optuna
from optuna_integration import OptunaSearchCV
import mlflow
from processing import preprocessing
from modelo import build_model
from metrics import metrics_report

#%%
dados = pd.read_excel(r'D:\estudos\cursos_projetos_ciencia_dados\ida_extra_renovacao\5_modelagem_e_risco_de_credito\projeto\dados\yochi_base.xlsx')
dados.head()

#%%
x = dados.drop(columns='BAD')
y = dados['BAD']

#%%
xcat = list(x.select_dtypes(include='object').columns)
xnum = list(x.select_dtypes(include='number').columns)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

#%%
models_params = {'random_forest': {'model__n_estimators': optuna.distributions.IntDistribution(100, 500),
                                   'model__max_features': optuna.distributions.FloatDistribution(0.2, 0.5),
                                   'model__max_depth': optuna.distributions.IntDistribution(5, 15),
                                   'model__min_samples_split': optuna.distributions.IntDistribution(10, 15),
                                   'model__min_samples_leaf': optuna.distributions.IntDistribution(5, 20)
                                  },

                 'xgb': {'model__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.1),
                         'model__max_depth': optuna.distributions.IntDistribution(5, 15),
                         'model__subsample': optuna.distributions.FloatDistribution(0.6, 1.0),
                         'model__colsample_bytree': optuna.distributions.FloatDistribution(0.5, 0.8),
                         'model__gamma': optuna.distributions.IntDistribution(1, 5),
                         'model__reg_lambda': optuna.distributions.IntDistribution(1, 5)
                 },

                 'lgb':{'model__learning_rate': optuna.distributions.FloatDistribution(0.01, 0.1), 
                        'model__num_leaves': optuna.distributions.IntDistribution(15, 25),       	
                        'model__max_depth': optuna.distributions.IntDistribution(5,15),       	
                        'model__min_data_in_leaf': optuna.distributions.IntDistribution(20, 100),  
                        'model__bagging_fraction': optuna.distributions.FloatDistribution(0.6, 1), 
                        'model__feature_fraction': optuna.distributions.FloatDistribution(0.6, 1), 
                        'model__lambda_l2': optuna.distributions.IntDistribution(1, 5)
                       }
}

#%%
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
mlflow.set_experiment(experiment_id='1')

#%%
for model in list(models_params.keys()):
    # itera sobre o dicionario de modelos e parametros iniciando uma execucao mlflow para cada modelo
    with mlflow.start_run(run_name=model):
        pipe_model = Pipeline(steps=[('preprocessing', preprocessing(xnum=xnum, xcat=xcat)),
                                     ('model', build_model(model_name=model))])
        
        # usa o conjunto de parametros específico para o modelo atual
        param_dist = models_params.get(model)

        search = OptunaSearchCV(pipe_model, 
                                param_distributions=param_dist,
                                n_trials=100,
                                cv=KFold(n_splits=5,
                                         shuffle=True,
                                         random_state=42),
                                scoring=make_scorer(fbeta_score, beta=0.5), #fbeata_score otimiza precisao e recall, priorizando precisao(beta=0.5)
                                return_train_score=True,
                                n_jobs=-1,
                                random_state=42)

        search.fit(xtrain, ytrain)

        ytrain_pred = search.predict(xtrain)
        ytest_pred = search.predict(xtest)

        report_metrics = metrics_report(ytrain_true=ytrain, ytrain_pred=ytrain_pred, 
                                        ytest_true=ytest, ytest_pred=ytest_pred, model=search)

        mlflow.log_metrics(report_metrics)
        mlflow.log_params(search.best_params_)
