#%%
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder

#%%
def preprocessing(xnum: list, xcat:list):
    # constroi pipeline de processamento de variaveis independentes

    pipe_num = Pipeline(steps=[('imputer_num', SimpleImputer(strategy='mean'))])

    pipe_cat = Pipeline(steps=[('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
                               ('mean_encoder', TargetEncoder(smoothing=20))])
    
    pipilene = ColumnTransformer(transformers=[('transformer_cat', pipe_cat, xcat),
                                                 ('transformer_num', pipe_num, xnum)])
    
    return pipilene
