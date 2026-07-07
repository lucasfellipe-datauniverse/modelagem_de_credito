#%%
import pandas as pd
from sklearn import metrics

#%%
def metrics_report(ytrain_true, ytrain_pred, ytest_true, ytest_pred, model):
    # gera um relatorio com metricas de qualidade prontas para logar no mlflow ou salvar onde desejar

    df_results = pd.DataFrame(model.cv_results_) 
    best_index = model.best_index_

    report = {'precision_train': metrics.precision_score(ytrain_true, ytrain_pred),
              'precision_test': metrics.precision_score(ytest_true, ytest_pred),
              
              'recall_train': metrics.recall_score(ytrain_true, ytrain_pred),
              'recall_test': metrics.recall_score(ytest_true, ytest_pred),
              
              'mean_train_score': df_results.loc[best_index, 'mean_train_score'],
              'mean_test_score': df_results.loc[best_index, 'mean_test_score'],
              'std_train_score': df_results.loc[best_index, 'std_train_score'],
              'std_test_score': df_results.loc[best_index, 'std_test_score']}
    
    return report