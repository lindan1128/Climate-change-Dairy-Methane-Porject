"""Fixed-effect projection and state-cluster inference."""
import numpy as np
from scipy.stats import t


def absorb(values, train, states, years, state_year=True, local_trend=False, tolerance=1e-10):
    values = np.asarray(values, float)
    if values.ndim == 2:
        values = values[..., None]
    n, nt, k = values.shape
    groups = np.unique(states, return_inverse=True)[1] if state_year else np.zeros(n, dtype=int)
    m = np.asarray(train, float); time = np.asarray(years, float) - np.mean(years)
    nobs = m.sum(1)
    if (nobs < 2).any():
        raise ValueError('Insufficient AMC training observations')
    a = np.zeros((n, k)); b = a.copy(); g = np.zeros((groups.max()+1, nt, k))
    s1, s2 = (m*time).sum(1), (m*time*time).sum(1)
    for iteration in range(10000):
        previous = a[:, None] + b[:, None]*time[None, :, None] + g[groups]
        u = values-g[groups]; v0=(u*m[...,None]).sum(1)
        if local_trend:
            v1=(u*(m*time)[...,None]).sum(1); determinant=nobs*s2-s1*s1
            a=(s2[:,None]*v0-s1[:,None]*v1)/determinant[:,None]
            b=(nobs[:,None]*v1-s1[:,None]*v0)/determinant[:,None]
        else:
            a=v0/nobs[:,None]
        residual=values-a[:,None]-b[:,None]*time[None,:,None]
        for group in range(len(g)):
            take=groups==group; denominator=m[take].sum(0)
            if (denominator==0).any():
                raise ValueError('A year or state-year has no training observations')
            g[group]=(residual[take]*m[take,:,None]).sum(0)/denominator[:,None]
        fitted=a[:,None]+b[:,None]*time[None,:,None]+g[groups]
        if np.max(np.abs(fitted-previous)) < tolerance:
            return values-fitted
    raise RuntimeError('Fixed effects did not converge')


def fit_panel(y, x, train, states, years, state_year=True, local_trend=False):
    residual=absorb(np.concatenate([y[...,None],x],axis=2),train,states,years,state_year,local_trend)
    xx=residual[...,1:][train]; yy=residual[...,0][train]
    if np.linalg.matrix_rank(xx)!=xx.shape[1]:
        raise ValueError('Rank-deficient climate design')
    inverse=np.linalg.inv(xx.T@xx); beta=inverse@(xx.T@yy)
    error=residual[...,0]-residual[...,1:]@beta
    levels=np.unique(states); G=len(levels)
    scores=np.array([residual[...,1:][train&(states[:,None]==s)].T@error[train&(states[:,None]==s)] for s in levels])
    influence=scores@inverse
    covariance=influence.T@influence*G/(G-1)
    return dict(beta=beta,covariance=covariance,influence=influence,residual=error,groups=G)


def coefficient_table(fit, names):
    import pandas as pd
    se=np.sqrt(np.diag(fit['covariance']));beta=fit['beta'];df=fit['groups']-1
    return pd.DataFrame(dict(variable=names,beta=beta,se=se,lower=beta-t.ppf(.975,df)*se,
                             upper=beta+t.ppf(.975,df)*se,p=2*t.sf(np.abs(beta/se),df)))


def load_panel(frame):
    ids=np.sort(frame.amc.unique());years=np.sort(frame.year.unique())
    if frame.duplicated(['amc','year']).any():raise ValueError('Duplicate AMC-year')
    def matrix(column):
        return frame.pivot(index='amc',columns='year',values=column).reindex(index=ids,columns=years).to_numpy(float)
    if (frame.groupby('amc').state.nunique()!=1).any():raise ValueError('AMC crosses state labels')
    states=frame.groupby('amc').state.first().reindex(ids).to_numpy()
    return ids,years,states,matrix
