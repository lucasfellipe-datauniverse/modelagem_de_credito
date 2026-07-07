#%%
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#%%
def build_model(model_name: str='random_forest'):
    # constroi de modelos
    models = {'random_forest': lambda: RandomForestClassifier(random_state=42),
              'xgb': lambda: XGBClassifier(random_state=42)}
    
    if model_name not in (list(models.keys())):
        raise ValueError(f'modelo {model_name} nao suportado, modelos suportados: {list(models.keys())}')

    return models[model_name]()
