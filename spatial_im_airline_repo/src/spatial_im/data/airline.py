from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class AirlineTables:
    airports: pd.DataFrame
    routes: pd.DataFrame


def load_airline_tables(airports_csv: str | Path, routes_csv: str | Path) -> AirlineTables:
    airports = pd.read_csv(airports_csv)
    routes = pd.read_csv(routes_csv)

    required_airports = {'airport_id', 'name', 'iata', 'lat', 'lon'}
    required_routes = {'source_airport_id', 'target_airport_id'}
    if not required_airports.issubset(set(airports.columns)):
        missing = required_airports - set(airports.columns)
        raise ValueError(f'Missing airport columns: {missing}')
    if not required_routes.issubset(set(routes.columns)):
        missing = required_routes - set(routes.columns)
        raise ValueError(f'Missing route columns: {missing}')

    airports = airports.drop_duplicates('airport_id').reset_index(drop=True)
    routes = routes.drop_duplicates(['source_airport_id', 'target_airport_id']).reset_index(drop=True)
    return AirlineTables(airports=airports, routes=routes)
