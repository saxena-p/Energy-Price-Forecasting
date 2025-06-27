import pandas as pd
import requests
from urllib.parse import quote


def get_historical_fr_data(query_start_date: str, query_end_date: str) -> pd.DataFrame:
    """
    Collects historical frequency response data from the NESO API.
    Returns a dataframe of price and volume.

    """
    query = f'''SELECT * FROM "596f29ac-0387-4ba4-a6d3-95c243140707"
            WHERE "serviceType" = 'Response' 
            AND  "deliveryStart" >= '{query_start_date}'
            AND "deliveryStart" <= '{query_end_date}'
            '''
    # URL encode query
    url = f"https://api.neso.energy/api/3/action/datastore_search_sql?sql={quote(query)}"

    # Fetch the data from API
    response = requests.get(url)
    data = response.json()

    # Convert the data to a DataFrame
    fr_auctions = pd.DataFrame(data['result']['records'])

    service_stacked_df = fr_auctions.copy()
    # Converting from string -> utc -> 'London' -> Native timezone 
    # 2025-04-09T22:00:00 -> 2025-04-09 22:00:00 -> 2025-04-09 22:00:00+01:00 -> 2025-04-09 23:00:00
    service_stacked_df.deliveryStart = pd.to_datetime(service_stacked_df.deliveryStart, utc = True).dt.tz_convert('Europe/London').dt.tz_localize(None, nonexistent='shift_forward')
    service_stacked_df.deliveryEnd = pd.to_datetime(service_stacked_df.deliveryEnd, utc = True).dt.tz_convert('Europe/London').dt.tz_localize(None, nonexistent='shift_forward')
    service_stacked_df.index = service_stacked_df.deliveryStart

    # Extract only the 'DCL' data
    dcl_data = service_stacked_df[service_stacked_df['auctionProduct'] == 'DCL']

    # Convert price and volume columns to numeric
    dcl_data['clearingPrice'] = pd.to_numeric(dcl_data['clearingPrice'], errors='coerce')
    dcl_data['clearedVolume'] = pd.to_numeric(dcl_data['clearedVolume'], errors='coerce')

    return dcl_data

def get_historical_fr_data_price_volume(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Collects historical frequency response data from the NESO API.
    Returns a dataframe of price and volume.
    
    Parameters:
    start_date (str): The start date for the data collection in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data collection in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: A DataFrame containing the historical frequency response data.
    """

    df = get_historical_fr_data(start_date, end_date)
    dcl_data = df[['clearingPrice', 'clearedVolume']].copy()
    return dcl_data  # 


def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers together.
    
    Parameters:
    a (float): The first number.
    b (float): The second number.
    
    Returns:
    float: The sum of the two numbers.
    """
    return a + b