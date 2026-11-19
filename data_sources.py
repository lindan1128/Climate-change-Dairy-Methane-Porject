"""Data sources named in the supplementary methods."""
from pathlib import Path

SOURCES = {
    'dairy': 'IBGE Pesquisa da Pecuária Municipal: milk production and milked cows',
    'methane': 'SEEG: dairy enteric fermentation and manure management',
    'historical_climate': 'TerraClimate v1.1: temperature, vapour pressure, precipitation and wind',
    'macroregions': 'IBGE regional classification',
    'biome_pasture': 'MapBiomas: biome, pasture area and pasture vigour',
    'elevation': 'WorldClim 2.1: 30-arc-second elevation',
    'census': 'IBGE agricultural censuses 2006 and 2017',
    'market': 'CILeite: milk prices and ICPLeite production-cost index',
    'future_climate': 'NASA NEX-GDDP-CMIP6',
    'population': 'IBGE population projections, revision 2024',
}


def check_sources(paths):
    """Check supplied locations without downloading or moving data."""
    unknown = set(paths) - set(SOURCES)
    if unknown:
        raise ValueError('Unknown source keys: ' + ', '.join(sorted(unknown)))
    return {name: Path(path).expanduser().exists() for name, path in paths.items()}
