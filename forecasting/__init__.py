# init file for the forecasting module

__version__ = "0.1.0"
__author__ = "Prashant Saxena"

from .get_data import get_historical_fr_data, add_numbers, get_historical_fr_data_price_volume, plot_series
from .model import rolling_average_multi_timeperiod, windowed_dataset_multi_input, get_LSTM_model, lstm_forecast_multi_input

__all__ = ['get_historical_fr_data', 'add_numbers', 'get_historical_fr_data_price_volume', 'plot_series',
           'rolling_average_multi_timeperiod', 'windowed_dataset_multi_input', 'get_LSTM_model', 'lstm_forecast_multi_input']