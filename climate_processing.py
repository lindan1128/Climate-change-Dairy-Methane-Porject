"""Area-weighted daily climate, annual summaries and quantile delta mapping."""
import numpy as np
from scipy import sparse
from scipy.stats import rankdata


def intersection_weights(amcs,cells,equal_area_crs):
    """cells contains the polygons of the climate grid in their original CRS."""
    import geopandas as gpd
    regions=amcs[['amc','geometry']].to_crs(equal_area_crs).reset_index(drop=True)
    grid=cells[['geometry']].to_crs(equal_area_crs).reset_index(drop=True)
    if not regions.crs.is_projected:raise ValueError('Use an equal-area projected CRS')
    regions['row']=np.arange(len(regions));grid['column']=np.arange(len(grid))
    overlap=gpd.overlay(regions,grid,how='intersection',keep_geom_type=False)
    areas=overlap.geometry.area.to_numpy()
    w=sparse.coo_matrix((areas,(overlap.row,overlap.column)),shape=(len(regions),len(grid))).tocsr()
    covered=np.asarray(w.sum(1)).ravel();coverage=covered/regions.geometry.area.to_numpy()
    if not np.all(coverage>.98):raise ValueError('AMC grid coverage must exceed 98%')
    return sparse.diags(1/covered)@w,coverage


def spatial_mean(daily,weights):
    values=np.asarray(daily,float);valid=np.isfinite(values)
    coverage=np.asarray(weights@valid.T).T
    if not np.all(coverage>.98):raise ValueError('Valid grid area must exceed 98%')
    return np.asarray(weights@np.where(valid,values,0).T).T/coverage


def annual_from_daily(temperature_kelvin,rh,precipitation_flux,wind,weights):
    t=np.asarray(temperature_kelvin)-273.15
    vapour=np.clip(rh,0,100)/100*.6108*np.exp(17.27*t/(t+237.3))
    fields=[t,vapour,precipitation_flux,wind]
    result=np.stack([spatial_mean(a,weights).mean(axis=0) for a in fields],axis=-1)
    result[:,2]*=86400*365
    return result


def quantile_delta_mapping(values,model_historical,observed_historical,additive):
    values=np.asarray(values,float);model_historical=np.asarray(model_historical,float);observed_historical=np.asarray(observed_historical,float)
    if not all(np.isfinite(x).all() for x in [values,model_historical,observed_historical]):raise ValueError('Incomplete calibration values')
    if values.ndim==1:
        values=values[:,None];model_historical=model_historical[:,None];observed_historical=observed_historical[:,None];squeeze=True
    else:squeeze=False
    result=np.empty_like(values)
    for j in range(values.shape[1]):
        q=(rankdata(values[:,j])-.5)/len(values)
        model=np.quantile(model_historical[:,j],q);observed=np.quantile(observed_historical[:,j],q)
        if additive:result[:,j]=values[:,j]+observed-model
        else:
            if (model<=0).any():raise ValueError('Multiplicative mapping requires positive historical model quantiles')
            result[:,j]=values[:,j]*observed/model
    return result[:,0] if squeeze else result
