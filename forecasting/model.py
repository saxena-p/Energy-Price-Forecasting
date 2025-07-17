import pandas as pd
import tensorflow as tf

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


def windowed_dataset_multi_input(series, X_lag_full, window_size, forecast_horizon, batch_size, shuffle_buffer):
    """
    Generates a TensorFlow dataset of (X_seq, X_lag), y for a multi-input NN

    Args:
      series (1D array): univariate time series
      X_lag_full (2D array): aligned lag/static features
      window_size (int): number of time steps for each sequence window
      forecast_horizon (int): Number of future time steps to predict
      batch_size (int): batch size for training
      shuffle_buffer (int): buffer size for shuffling

    Returns:
      tf.data.Dataset yielding ((X_seq, X_lag), y)
    """

    # Expand dims for sequence input (shape: (n, 1))
    series = tf.expand_dims(series, axis=-1)

    # Concatenate along axis 1: final shape = (n, 1 + num_lag_features)
    full_input = tf.concat([series, X_lag_full], axis=1)

    total_window = window_size + forecast_horizon

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices(full_input)

    # Window and batch
    ds = ds.window(total_window, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(total_window))

    # Split into (X_seq, X_lag), y
    def split_window(window):
        series_window = window[:, 0:1]
        lag_window = window[:, 1:]

        X_seq = series_window[:window_size]     # (window_size, 1)
        X_lag = lag_window[-1]                  # static features from last input step
        y = tf.reshape(series_window[window_size:], (forecast_horizon,))    # (forecast_horizon, )                  

        return (X_seq, X_lag), y

    ds = ds.map(split_window)
    ds = ds.shuffle(shuffle_buffer).batch(batch_size).cache().prefetch(1)

    return ds

def get_LSTM_model(window_size: int = 24, 
                   learning_rate: float = 1e-6, 
                   momentum: float = 0.9, 
                   num_lag_features: int = 6) -> tf.keras.Model:
    # Create the deep learning model
    

    # First the sequence Input layer
    seq_input = tf.keras.Input(shape=(window_size, 1), name = 'sequence input')

    # Convolutional layer to extract features from the sequence
    x_seq = tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                        strides=1, padding="causal",
                        activation="relu")(seq_input)

    # Stacked LSTM layers to capture temporal dependencies
    x_seq = tf.keras.layers.LSTM(64, return_sequences=True)(x_seq)
    x_seq = tf.keras.layers.LSTM(64)(x_seq)

    # Lag + static features Input layer
    lag_input = tf.keras.Input(shape = (num_lag_features,), name = 'lag input')
    x_lag = tf.keras.layers.Dense(64, activation='relu')(lag_input)

    # Combine both of them
    x_combined = tf.keras.layers.Concatenate()([x_seq, x_lag])
    x = tf.keras.layers.Dense(64, activation='relu')(x_combined)
    output = tf.keras.layers.Dense(6)(x)

    # Create the model
    model = tf.keras.Model(inputs=[seq_input, lag_input], outputs=output)

    # Initialize the optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-6, momentum=0.9)

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

    return model

# The function to get the predictions
def lstm_forecast_multi_input(model, series, X_lag_full, window_size, forecast_horizon, batch_size):
    """
    Generates predictions using a multi-input model (sequence + lag/static).

    Args:
        model (tf.keras.Model): Trained model
        series (1D array): Time series data (shape: [n,])
        X_lag_full (2D array): Lag/static features aligned with the series (shape: [n, num_features])
        window_size (int): Number of time steps for sequence input
        forecast_horizon (int): number of future steps to predict
        batch_size (int): Batch size for inference

    Returns:
        forecast (np.ndarray): Model predictions
    """


    # Add feature dimension to series for RNN input
    series = tf.expand_dims(series, axis=-1)  # (n, 1)

    # Combine sequence and lag input
    full_input = tf.concat([series, X_lag_full], axis=1)  # (n, 1 + num_lag_features)

    total_window = window_size + forecast_horizon

    # Create dataset of input windows
    ds = tf.data.Dataset.from_tensor_slices(full_input)
    ds = ds.window(total_window, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(total_window))

    # Split into inputs for model: (X_seq, X_lag)
    def extract_inputs(window):
        series_window = window[:window_size, 0:1]   # (window_size, 1)
        lag_features = window[-1, 1:]    # (num_lags,)
        return series_window, lag_features

    ds = ds.map(extract_inputs)  # Extract inputs for model
    ds = ds.batch(batch_size).prefetch(1)


    # Run predictions
    # Step 6: Extract all x_seq and x_lag from the dataset into tensors
    x_seq_list = []
    x_lag_list = []

    for x_seq, x_lag in ds:
        x_seq_list.append(x_seq)
        x_lag_list.append(x_lag)

    # Concatenate batches into full input arrays
    x_seq_full = tf.concat(x_seq_list, axis=0)
    x_lag_full = tf.concat(x_lag_list, axis=0)

    # Step 7: Predict using separate inputs
    forecast = model.predict([x_seq_full, x_lag_full], verbose=0)

    return forecast