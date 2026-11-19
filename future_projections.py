"""Fixed-supply, productivity and population-demand scenarios, equations 6-9."""
import numpy as np
import pandas as pd
from common import CLIMATE
from panel import fit_panel,load_panel

MODELS=['ACCESS-CM2','CESM2','CNRM-CM6-1','CanESM5','EC-Earth3']
SCENARIOS=['ssp126','ssp370','ssp585']
YEARS=np.arange(2030,2061)


def unscale_coefficients(coefficients,scales):
    """scales are the original-variable standard deviations used by the SEM."""
    d=coefficients.copy()
    d['coefficient']=d.standardized* d.target.map(scales)/d.source.map(scales)
    if not np.isfinite(d.coefficient).all():raise ValueError('Missing SEM scales')
    return {(r.target,r.source):r.coefficient for r in d.itertuples()}


def project(climate_delta,pasture_reference,milk_reference,cows_reference,methane_reference,beta,
            productivity=1.,demand=1.):
    """Climate arrays end with four variables; pasture is in percentage points."""
    delta=np.asarray(climate_delta,float)
    if delta.shape[-1]!=4:raise ValueError('Expected four climate variables')
    def coefficient(target,source):return beta.get((target,source),0.)
    p_change=sum(delta[...,j]*coefficient('P',c) for j,c in enumerate(CLIMATE))
    pasture=np.clip(pasture_reference+p_change,0,100)
    p_change=pasture-pasture_reference
    cy=sum(delta[...,j]*coefficient('Y',c) for j,c in enumerate(CLIMATE))+coefficient('Y','P')*p_change
    ce=sum(delta[...,j]*coefficient('E',c) for j,c in enumerate(CLIMATE))+coefficient('E','P')*p_change
    b=coefficient('Y','N')
    if abs(1+b)<1e-12:raise ValueError('Singular fixed-supply solution')
    if np.any(np.asarray(productivity)<=0) or np.any(np.asarray(demand)<=0):raise ValueError('Positive productivity and demand required')
    log_a=np.log(productivity);log_d=np.log(demand)
    log_y=(cy+log_a+b*log_d)/(1+b)
    log_n=log_d-log_y
    log_e=ce+coefficient('E','N')*log_n+coefficient('E','Y')*log_y
    y0=np.asarray(milk_reference)/cows_reference
    y=y0*np.exp(log_y);n=np.asarray(cows_reference)*np.exp(log_n);e=np.asarray(methane_reference)*np.exp(log_e)
    np.testing.assert_allclose(y*n,np.broadcast_to(np.asarray(milk_reference)*demand,y.shape),rtol=1e-10)
    np.testing.assert_allclose(log_y,cy+log_a+b*log_n,rtol=1e-10,atol=1e-12)
    return dict(yield_value=y,cows=n,methane=e,pasture=pasture)


def productivity_rate(panel,start,quadratic):
    d=panel[panel.year.between(start,2024)].copy();ids,years,states,matrix=load_panel(d)
    raw=matrix('yield')
    if not np.isfinite(raw).all() or (raw<=0).any():raise ValueError('Productivity trend requires complete positive yield')
    climate=np.stack([matrix(c) for c in CLIMATE],axis=-1);mean=climate.mean(axis=(0,1));sd=climate.std(axis=(0,1))
    z=(climate-mean)/sd;X=np.concatenate([z,z*z],axis=-1) if quadratic else z
    y=np.log(raw);fit=fit_panel(y,X,np.ones_like(y,dtype=bool),states,years,state_year=False)
    adjusted=(y-X@fit['beta']).mean(axis=0)
    return float(np.polyfit(years,adjusted,1)[0])


def productivity_checks(panel):
    return pd.DataFrame([dict(start=start,end=2024,quadratic=quadratic,g=productivity_rate(panel,start,quadratic))
                         for start in [2007,2010,2015] for quadratic in [False,True]])


def productivity_factor(years,g,fraction):
    reference=np.arange(2020,2025)
    return np.exp(fraction*g*(np.asarray(years)-2020))/np.exp(fraction*g*(reference-2020)).mean()


def population_factor(population):
    """National demand multiplier, reconstructed from the 27 IBGE state units.

    Population is divided by its 2020–2024 national mean, as specified in
    Supplementary Materials. Input may alternatively contain the national series.
    """
    required=np.r_[2020:2025,2030:2061]
    d=population.copy()
    if not np.isfinite(d.population).all() or (d.population<=0).any():
        raise ValueError('Population must be finite and positive')
    if 'abbreviation' in d:
        if d.duplicated(['abbreviation','year']).any():
            raise ValueError('Duplicate population unit-year')
        if d.abbreviation.eq('BR').any():
            p=d[d.abbreviation.eq('BR')].set_index('year').population
        else:
            expected=set('RO AC AM RR PA AP TO MA PI CE RN PB PE AL SE BA MG ES RJ SP PR SC RS MS MT GO DF'.split())
            if set(d.abbreviation)!=expected:
                raise ValueError('All 26 states and the Federal District are required')
            table=d.pivot(index='year',columns='abbreviation',values='population').reindex(required)
            if not np.isfinite(table.to_numpy()).all():raise ValueError('Incomplete state population projection')
            p=table.sum(axis=1)
    else:
        if d.year.duplicated().any():raise ValueError('Duplicate national population year')
        p=d.set_index('year').population
    if not np.isfinite(p.reindex(required)).all():raise ValueError('Incomplete population projection')
    return p.reindex(YEARS)/p.reindex(range(2020,2025)).mean()


def summarize_models(results):
    """Return the median and full model range; retain individual model means."""
    keys=['scenario','year','outcome']
    if results.duplicated(keys+['model']).any():raise ValueError('Duplicate model mean')
    if not set(results.model).issubset(MODELS) or not set(results.scenario).issubset(SCENARIOS):raise ValueError('Unexpected model or scenario')
    if not results.groupby(keys).model.nunique().eq(5).all():raise ValueError('Five climate models required per summary')
    return results.groupby(keys).value.agg(median='median',minimum='min',maximum='max').reset_index()


def historical_intervals(values,metadata,seed):
    """values has dimensions AMC × year × outcome, in metadata order."""
    values=np.asarray(values,float)
    if len(values)!=len(metadata) or not np.isfinite(values).all():raise ValueError('Invalid historical array')
    rng=np.random.default_rng(seed);boot=[]
    groups=[]
    for state,d in metadata.reset_index(drop=True).groupby('state'):
        groups.append([q.index.to_numpy() for _,q in d.groupby('intermediate')])
    for b in range(1999):
        indexes=np.concatenate([np.concatenate([g[j] for j in rng.integers(0,len(g),len(g))]) for g in groups])
        boot.append(values[indexes].mean(axis=0))
    return values.mean(axis=0),np.quantile(boot,[.025,.975],axis=0)


def sensitivity_contrasts(climate_delta,reference,beta,years,g,population_multiplier):
    """Equal-AMC contrasts under the same productivity rate in each pair."""
    demand=np.asarray(population_multiplier,float)
    if demand.shape!=(len(years),):raise ValueError('One national demand multiplier is required per year')
    demand=demand[:,None]
    rows=[]
    for fraction in [0,.25,.5,1.]:
        a=productivity_factor(years,g,fraction)[:,None]
        baseline=project(np.zeros_like(climate_delta),**reference,beta=beta,productivity=a)
        climate=project(climate_delta,**reference,beta=beta,productivity=a)
        combined=project(climate_delta,**reference,beta=beta,productivity=a,demand=demand)
        for i,year in enumerate(years):
            rows.append(dict(year=year,growth_fraction=fraction,
                yield_shortfall=np.mean(baseline['yield_value'][i]-climate['yield_value'][i]),
                climate_methane=np.mean(climate['methane'][i]-baseline['methane'][i]),
                climate_and_demand_methane=np.mean(combined['methane'][i]-baseline['methane'][i])))
    return pd.DataFrame(rows)
