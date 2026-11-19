"""Annual associations, temporal validation and paired trend-removal contrasts."""
import argparse
import json
import numpy as np
import pandas as pd
from common import CLIMATE, config, read_table, output_dir
from panel import load_panel, absorb, fit_panel, coefficient_table


def design(climate, mean, sd, quadratic):
    z=(climate-mean)/sd
    return np.concatenate([z,z*z],axis=-1) if quadratic else z


def temporal_folds(valid, states, years, rng):
    blocks=np.searchsorted([1984,1995,2006,2017],years,side='left')
    offsets=rng.integers(0,5,len(states))
    assignment=(offsets[:,None]+blocks[None,:])%5
    folds=[]
    for fold in range(5):
        train=valid&(assignment!=fold)
        for state in np.unique(states):
            for j in range(len(years)):
                eligible=np.flatnonzero(valid[:,j]&(states==state))
                need=min(2,len(eligible))-train[eligible,j].sum()
                if need>0:
                    train[eligible[~train[eligible,j]][:int(need)],j]=True
        folds.append((train,valid&~train))
    return folds


def select_model(y, climate, valid, states, years, rng, controls):
    mean=climate[valid].mean(0);sd=climate[valid].std(0)
    folds=temporal_folds(valid,states,years,rng)
    rows=[]
    if not controls:raise ValueError('At least one control structure is required')
    for state_year,trend in controls:
        for fold,(train,test) in enumerate(folds):
            residual=absorb(y,train,states,years,state_year,trend)[...,0]
            rows.append(dict(state_year=state_year,local_trend=trend,fold=fold,shape='baseline',
                             n=int(test.sum()),mse=float(np.mean(residual[test]**2))))
    scores=pd.DataFrame(rows)
    choice=min(controls,
               key=lambda pair:np.average(scores.loc[(scores.state_year==pair[0])&(scores.local_trend==pair[1]),'mse'],
                  weights=scores.loc[(scores.state_year==pair[0])&(scores.local_trend==pair[1]),'n']))
    errors={}
    for quadratic in [False,True]:
        errors[quadratic]=[]
        for fold,(train,test) in enumerate(folds):
            fit=fit_panel(y,design(climate,mean,sd,quadratic),train,states,years,*choice)
            mse=np.mean(fit['residual'][test]**2);errors[quadratic].append(mse)
            rows.append(dict(state_year=choice[0],local_trend=choice[1],fold=fold,
                             shape='quadratic' if quadratic else 'linear',n=int(test.sum()),mse=mse))
    gains=np.array(errors[False])-np.array(errors[True])
    quadratic=bool(gains.mean()>1.96*gains.std(ddof=1)/np.sqrt(5))
    return choice,quadratic,pd.DataFrame(rows),mean,sd


def counterfactual(climate, years, slopes, fraction):
    return climate-fraction*slopes[:,None,:]*(years-1976)[None,:,None]


def contrasts(fit, observed, alternative, milk, methane, years, mean, sd, quadratic, signs, elasticities=(0,.25,.5,.75)):
    delta=design(alternative,mean,sd,quadratic)-design(observed,mean,sd,quadratic)
    recent=(years>=2020)&(years<=2024);delta=delta[:,recent];milk=milk[:,recent];methane=methane[:,recent]
    if not np.isfinite(milk).all() or not np.isfinite(methane).all() or (milk<0).any() or (methane<0).any():
        raise ValueError('Invalid reference milk or methane')
    if milk.sum()<=0 or methane.sum()<=0:raise ValueError('Zero reference totals')
    draws=fit['beta']+signs@fit['influence']*np.sqrt(fit['groups']/(fit['groups']-1))
    coefficients=np.vstack([fit['beta'],draws]);rows=[]
    # Process one draw at a time to avoid storing AMC × year × draw arrays.
    estimates=np.array([[np.sum(milk*np.exp(delta@b))/milk.sum()]+[
        np.sum(methane*np.exp((k-1)*(delta@b)))/methane.sum() for k in elasticities] for b in coefficients])
    labels=['milk_ratio']+['methane_ratio_k_'+str(k) for k in elasticities]
    for j,label in enumerate(labels):
        low,high=np.quantile(estimates[1:,j],[.025,.975])
        rows.append(dict(outcome=label,estimate=estimates[0,j],lower=low,upper=high))
    return rows


def analyse(panel, rng, fractions):
    ids,years,states,matrix=load_panel(panel)
    climate=np.stack([matrix(c) for c in CLIMATE],axis=-1)
    raw=matrix('yield');valid=raw>0;y=np.log(np.where(valid,raw,1))
    if not np.isfinite(climate).all():raise ValueError('Incomplete climate panel')
    mean=climate[valid].mean(0);sd=climate[valid].std(0)
    linear=fit_panel(y,design(climate,mean,sd,False),valid,states,years)
    association=coefficient_table(linear,CLIMATE)
    # Compare AMC effects with national/state-year effects, with/without local trends.
    controls=[(False,False),(True,False),(False,True),(True,True)]
    choice,quadratic,cv,mean,sd=select_model(y,climate,valid,states,years,rng,controls)
    fit=fit_panel(y,design(climate,mean,sd,quadratic),valid,states,years,*choice)
    slope=np.einsum('ntv,t->nv',climate,years-years.mean())/np.sum((years-years.mean())**2)
    # Temperature and vapour remove positive trends; individual precipitation
    # and wind scenarios remove trends of either sign.
    individual_slopes=slope.copy();individual_slopes[:,:2]=np.maximum(slope[:,:2],0)
    milk=matrix('milk');methane=matrix('methane')
    signs=rng.choice([-1.,1.],size=(1999,len(np.unique(states))))
    rows=[]
    for variable,name in enumerate(CLIMATE):
        selected=np.zeros_like(slope);selected[:,variable]=individual_slopes[:,variable]
        for fraction in fractions:
            cf=counterfactual(climate,years,selected,fraction)
            for row in contrasts(fit,climate,cf,milk,methane,years,mean,sd,quadratic,signs):
                rows.append(dict(variable=name,fraction=fraction,**row))
    joint_slopes=individual_slopes.copy();joint_slopes[:,2:]=np.minimum(slope[:,2:],0)
    joint_rows=[]
    for fraction in fractions:
        cf=counterfactual(climate,years,joint_slopes,fraction)
        joint_rows.extend(dict(fraction=fraction,**r) for r in
                          contrasts(fit,climate,cf,milk,methane,years,mean,sd,quadratic,signs,(0,)))
    single=single_predictor_sensitivity(y,climate,milk,methane,states,years,valid,individual_slopes,fractions,rng)
    temporal=joint_sensitivity(y,climate,milk,methane,states,years,valid,joint_slopes,fractions,rng)
    return (association,cv,pd.DataFrame(rows),
            dict(state_year=choice[0],local_trend=choice[1],quadratic=quadratic),
            pd.DataFrame(joint_rows),single,temporal)


def joint_sensitivity(y, climate, milk, methane, states, years, valid, pressure_slopes, fractions, rng):
    """Use direction-filtered climate slopes for joint trend removal."""
    mean=climate[valid].mean(0);sd=climate[valid].std(0)
    signs=rng.choice([-1.,1.],size=(1999,len(np.unique(states))))
    rows=[]
    for local_trend in [False,True]:
        fit=fit_panel(y,design(climate,mean,sd,True),valid,states,years,True,local_trend)
        for fraction in fractions:
            cf=counterfactual(climate,years,pressure_slopes,fraction)
            rows.extend(dict(local_trend=local_trend,fraction=fraction,**r) for r in
                        contrasts(fit,climate,cf,milk,methane,years,mean,sd,True,signs))
    return pd.DataFrame(rows)


def single_predictor_sensitivity(y, climate, milk, methane, states, years, valid, pressure_slopes, fractions, rng):
    signs=rng.choice([-1.,1.],size=(1999,len(np.unique(states))));rows=[]
    for j,name in enumerate(CLIMATE):
        for specification,index in [('four_climates',list(range(4))),('single_climate',[j])]:
            c=climate[:,:,index];mean=c[valid].mean(0);sd=c[valid].std(0)
            fit=fit_panel(y,design(c,mean,sd,True),valid,states,years)
            slopes=np.zeros_like(pressure_slopes);slopes[:,j]=pressure_slopes[:,j]
            for fraction in fractions:
                cf=counterfactual(climate,years,slopes,fraction)[:,:,index]
                rows.extend(dict(variable=name,specification=specification,fraction=fraction,**r) for r in
                            contrasts(fit,c,cf,milk,methane,years,mean,sd,True,signs,(0,)))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config');parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fractions',type=float,nargs='+',required=True)
    args=parser.parse_args()
    if not all(0<=a<=1 for a in args.fractions):parser.error('Fractions must be within [0, 1]')
    cfg=config(args.config);p=read_table(cfg['annual_panel']);p=p[p.year.between(1974,2024)].copy();p['yield']=p.milk/p.cows
    out=output_dir(cfg,'historical_panel');results=analyse(p,np.random.default_rng(args.seed),args.fractions)
    for name,table in zip(['annual_coefficients','temporal_cv','individual_counterfactuals'],results[:3]):table.to_csv(out/(name+'.csv'),index=False)
    (out/'selected_model.json').write_text(json.dumps(results[3],indent=2))
    for name,table in zip(['joint_counterfactuals','single_predictor_sensitivity','temporal_emission_factor_sensitivity'],results[4:]):
        table.to_csv(out/(name+'.csv'),index=False)

if __name__=='__main__':main()
