import pandas as pd

def rolling_average_multi_timeperiod(num_days = 7, num_timestamps_per_day = 6, df: pd.DataFrame = None):
    """
    Calculate rolling averages over multiple days for multiple time periods per day.
    
    Parameters:
    num_days (int): The number of days to consider for the rolling average.
    num_timestamps_per_day (int): The number of timestamps per day.
    df (pd.DataFrame): The DataFrame containing the data for which rolling averages are to be calculated.
    
    Returns:
    df (pd.DataFrame): DataFrame with num_timestamps_per_day rows containing averages for each time step.
    """

    avg_list = []

    for period in range(0, num_timestamps_per_day):
        current_sum = 0
        for day in range(1, num_days+1):
            current_sum += df.iloc[-day* num_timestamps_per_day + period]
        
        avg_list.append(current_sum / num_days)
    
    
    return avg_list