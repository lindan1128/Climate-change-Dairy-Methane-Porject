"""Study design, annual aggregation and production-context transformations."""
import argparse
import json
import numpy as np
import pandas as pd
from common import CLIMATE, config, read_table, require, output_dir


def annual_climate(monthly):
    require(monthly, ['amc', 'year', 'month', 'tmin', 'tmax', 'vapour', 'precipitation', 'wind'])
    if monthly.duplicated(['amc', 'year', 'month']).any():
        raise ValueError('Duplicate AMC-month')
    d = monthly.copy()
    d['temperature'] = (d.tmin + d.tmax) / 2
    d['days'] = pd.to_datetime(dict(year=d.year, month=d.month, day=1)).dt.days_in_month
    if not d.groupby(['amc', 'year']).month.nunique().eq(12).all():
        raise ValueError('Annual climate requires all twelve months')
    if not np.isfinite(d[CLIMATE]).all().all():
        raise ValueError('Missing monthly climate')
    for c in ['temperature', 'vapour', 'wind']:
        d[c] *= d.days
    result = d.groupby(['amc', 'year'])[CLIMATE + ['days']].sum()
    result[['temperature', 'vapour', 'wind']] = result[['temperature', 'vapour', 'wind']].div(result.days, axis=0)
    return result.drop(columns='days').reset_index()


def context_variables(d):
    require(d, ['farm_size_2006', 'farm_size_2017', 'cows_mean', 'cooling_family_litres',
                'cooling_nonfamily_litres', 'census_cows', 'area_km2'])
    d = d.copy()
    d['farm_size'] = (11 * d.farm_size_2006 + 7 * d.farm_size_2017) / 18
    for c in ['farm_size', 'cows_mean', 'area_km2']:
        if (d[c] <= 0).any():
            raise ValueError('Nonpositive ' + c)
    if (d.census_cows <= 0).any():
        raise ValueError('Nonpositive census cow count')
    d['log_farm_size'] = np.log(d.farm_size)
    d['log_cows'] = np.log(d.cows_mean)
    d['log_area'] = np.log(d.area_km2)
    d['cooling_total'] = d.cooling_family_litres + d.cooling_nonfamily_litres
    if (d.cooling_total < 0).any():
        raise ValueError('Negative storage capacity')
    d['cooling_per_cow'] = d.cooling_total / d.census_cows
    d['log1p_cooling_per_cow'] = np.log1p(d.cooling_per_cow)
    d['log1p_cooling_total'] = np.log1p(d.cooling_total)
    return d


def market_summary(annual):
    require(annual, ['state', 'year', 'price', 'cost_index'])
    if annual.duplicated(['state', 'year']).any() or (annual[['price', 'cost_index']] <= 0).any().any():
        raise ValueError('Market inputs require unique positive annual values')
    d = annual[annual.year.between(2007, 2024)].copy()
    d['log_price'] = np.log(d.price / d.cost_index)
    rows = []
    for state, q in d.groupby('state'):
        if len(q) != 18:
            raise ValueError('Incomplete market series for ' + str(state))
        rows.append(dict(state=state, price_log_mean=q.log_price.mean(),
                         price_log_trend=np.polyfit(q.year, q.log_price, 1)[0]))
    return pd.DataFrame(rows)


def check_design(panel, context):
    require(panel, ['amc', 'state', 'year', 'milk', 'cows'] + CLIMATE)
    p = panel[panel.year.between(1974, 2024)].copy()
    if p.duplicated(['amc', 'year']).any() or p.amc.nunique() != 3115 or len(p) != 3115 * 51:
        raise ValueError('Expected 3,115 AMCs × 51 unique years')
    if (p.cows <= 0).any() or not np.isfinite(p[['milk', 'cows'] + CLIMATE]).all().all():
        raise ValueError('Invalid annual inputs')
    p['yield'] = p.milk / p.cows
    if (p['yield'] > 0).sum() != 158862:
        raise ValueError('Positive-yield sample does not match 158,862 observations')
    require(context, ['amc', 'state', 'intermediate'])
    if context.amc.duplicated().any() or len(context) != 1881:
        raise ValueError('Expected 1,881 unique context AMCs')
    if context.state.nunique() != 7 or context[['state', 'intermediate']].drop_duplicates().shape[0] != 59:
        raise ValueError('Expected seven states and 59 intermediate regions')
    if not set(context.amc).issubset(set(p.amc)):
        raise ValueError('Context sample is outside the historical cohort')
    return p


def technology_index(mechanical, insemination, cooling_per_cow):
    x=np.column_stack([mechanical,insemination,cooling_per_cow]).astype(float)
    denominator=x.mean(0)+3*x.std(0,ddof=1)
    if not np.isfinite(x).all() or (denominator<=0).any():
        raise ValueError('Invalid technology components')
    score=np.minimum(x/denominator,1).mean(1)
    return (score-score.mean())/score.std(ddof=1)


def historical_summaries(panel):
    d=panel[panel.year.between(1974,2024)].copy()
    d['yield']=d.milk/d.cows
    means=d.groupby('amc')[CLIMATE+['yield']].mean().reset_index()
    annual=d.groupby('year')[CLIMATE+['yield']].agg(['mean','std'])
    return means,annual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args(); cfg = config(args.config)
    p, d = read_table(cfg['annual_panel']), read_table(cfg['context'])
    p = check_design(p, d)
    out = output_dir(cfg, 'data')
    p.to_csv(out / 'annual_panel.csv', index=False)
    means,annual=historical_summaries(p)
    means.to_csv(out/'amc_means.csv',index=False)
    annual.to_csv(out/'annual_descriptive.csv')
    context_variables(d).to_csv(out / 'context.csv', index=False)
    (out / 'sample_counts.json').write_text(json.dumps({'historical_amcs':3115, 'positive_yield_rows':158862,
                                                       'context_amcs':1881, 'states':7, 'intermediate_regions':59}, indent=2))

if __name__ == '__main__':
    main()
