import pandas as pd

def rolling_average_multi_timeperiod(num_days = 7, num_timestamps_per_day = 6, df: pd.DataFrame = None):
    """
    Calculate rolling averages over multiple days for multiple time periods per day.
    
    Parameters:
    num_days (int): The number of days to consider for the rolling average.
    num_timestamps_per_day (int): The number of timestamps per day.
    df (pd.DataFrame): The DataFrame containing the data for which rolling averages are to be calculated.
    
    Returns:
    List with num_timestamps_per_day elements containing averages for each time step.
    """

    avg_list = []

    for period in range(1, num_timestamps_per_day+1):
        # For each time step, get data for the past num_days days
        indices = [-i * num_timestamps_per_day - period for i in range( num_days)]
        data = df.iloc[indices]
        avg_list.append(data.mean())
    
    return avg_list