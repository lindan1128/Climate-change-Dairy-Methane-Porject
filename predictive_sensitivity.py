"""Grouped prediction, test-set SHAP and exact permutation comparisons."""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


def fit_predictive(d, predictors, outcome, split, parameter_grid, seed):
    """Split membership assigns each AMC to training or testing."""
    import shap
    if split.amc.duplicated().any() or set(split.amc)!=set(d.amc):
        raise ValueError('Split membership must identify every AMC once')
    data=d.merge(split[['amc','set']],on='amc',validate='one_to_one')
    train=data['set'].eq('train').to_numpy();test=data['set'].eq('test').to_numpy()
    if train.sum()!=1586 or test.sum()!=295:
        raise ValueError('Expected 1,586 training and 295 test AMCs')
    groups=data.state.astype(str)+'/'+data.intermediate.astype(str)
    if set(groups[train])&set(groups[test]):raise ValueError('Intermediate-region leakage')
    if (data[outcome]<=0).any():raise ValueError('Outcome must be positive before taking logs')
    x=data[predictors];y=np.log(data[outcome].to_numpy())
    if 'n_estimators' in parameter_grid and parameter_grid['n_estimators']!=[300]:
        raise ValueError('Each ensemble requires 300 trees')
    cv=GroupKFold(n_splits=5)
    model=GridSearchCV(ExtraTreesRegressor(n_estimators=300,random_state=seed),parameter_grid,
                      scoring='neg_root_mean_squared_error',cv=cv,refit=True)
    model.fit(x.loc[train],y[train],groups=groups[train])
    background=x.loc[train].sample(n=100,random_state=seed)
    explainer=shap.TreeExplainer(model.best_estimator_,data=background,feature_perturbation='interventional')
    values=np.asarray(explainer.shap_values(x.loc[test]))
    importance=np.abs(values).mean(0)/y[train].std(ddof=1)
    predicted=model.predict(x.loc[test])
    metrics=dict(rmse=np.sqrt(mean_squared_error(y[test],predicted)),r2=r2_score(y[test],predicted))
    return model,pd.DataFrame({'predictor':predictors,'importance':importance}),metrics


def exact_spearman(x,y):
    x=rankdata(x);y=rankdata(y)
    if len(x)!=len(y) or len(x)<3 or np.std(x)==0 or np.std(y)==0:
        raise ValueError('Correlation requires matched nonconstant vectors')
    x=(x-x.mean())/np.linalg.norm(x-x.mean());y=(y-y.mean())/np.linalg.norm(y-y.mean())
    observed=x@y;extreme=total=0
    for permutation in itertools.permutations(y):
        extreme+=abs(x@permutation)>=abs(observed)-1e-12;total+=1
    return float(observed),extreme/total


def compare_effects(importance,effects):
    d=importance.merge(effects,on='predictor',validate='one_to_one')
    d['sem_magnitude']=d.direct.abs()+d.indirect.abs()
    rho,p=exact_spearman(d.sem_magnitude,d.importance)
    slope,intercept=np.polyfit(d.sem_magnitude,d.importance,1)
    return d,dict(spearman=rho,exact_p=p,descriptive_slope=slope,descriptive_intercept=intercept)
