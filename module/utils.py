import pandas as pd
import numpy as np
import yfinance as yf
import os

from sklearn.preprocessing import MinMaxScaler


# Create new CSV file with the stock data
def get_csv(coin_type):
    """
    Retrieves historical stock data for a given coin type and saves it as a CSV file.

    Args:
        coin_type (str): The type of coin for which to retrieve the data.

    Returns:
        pandas.DataFrame: The historical stock data as a pandas DataFrame.
    """
    ticket = yf.Ticker(coin_type)
    hist = ticket.history(period="5y")
    hist.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
    hist.index = pd.to_datetime(hist.index).date
    hist = hist.reset_index().rename(columns={"index": "Date"})
    hist[["Open", "High", "Low", "Close"]] = hist[
        ["Open", "High", "Low", "Close"]
    ].round(5)
    hist.to_csv(
        f"{coin_type}.csv",
        columns=["Date", "Open", "High", "Low", "Close"],
        index=False,
    )
    return hist


def get_all_csv():
    """
    Retrieves CSV files for multiple coin types.

    This function iterates over a list of coin types and calls the `get_csv` function
    to retrieve the CSV file for each coin type.

    Args:
        None

    Returns:
        None
    """
    coin_types = ["BTC-USD", "ADA-USD", "ETH-USD"]
    for coin_type in coin_types:
        get_csv(coin_type)


# get new candles data
def getNewTicketData(name, period):
    """
    Retrieves new stock data for the given stock name and period, and appends it to a CSV file.

    Args:
        name (str): The name of the stock.
        period (str): The time period for which to retrieve the stock data.

    Returns:
        None: If the first date of the retrieved data is the same as the last date in the CSV file.

    """
    # get stock data from yfinance
    ticket = yf.Ticker(name)
    hist = ticket.history(period)

    # modify data
    hist.drop(columns=["Volume", "Dividends", "Stock Splits"], inplace=True)
    hist.index = pd.to_datetime(hist.index).date
    hist = hist.reset_index().rename(columns={"index": "Date"})
    hist[["Open", "High", "Low", "Close"]] = hist[
        ["Open", "High", "Low", "Close"]
    ].round(5)

    csv_file = name + ".csv"
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        # Read the last row of the CSV
        last_row = pd.read_csv(csv_file).tail(1)
        last_date_csv = pd.to_datetime(last_row["Date"].values[0]).date()

        # Get the first date from the hist DataFrame
        first_date_hist = hist["Date"].iloc[0]

        # Compare the dates and return if they are the same
        if first_date_hist == last_date_csv:
            return

    # Append to CSV if the dates are not the same
    hist.to_csv(name + ".csv", mode="a", index=False, header=False)


def updateAllTicket(period):
    getNewTicketData("BTC-USD", period)
    getNewTicketData("ETH-USD", period)
    getNewTicketData("ADA-USD", period)


### transform data to be used in the model
def preprocess_data(name):
    """
    Preprocesses the stock data for training a predictive model.

    Args:
        name (str): The name of the stock.

    Returns:
        list: A list containing the preprocessed data and other related information.
            - x_train_data (numpy.ndarray): The input training data.
            - y_train_data (numpy.ndarray): The target training data.
            - X_test (numpy.ndarray): The test data.
            - valid_data (pandas.DataFrame): The validation data.
            - scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for scaling the data.
            - original_df (pandas.DataFrame): The original dataframe.

    """
    # Load and preprocess data
    df = pd.read_csv(f"{name}.csv", parse_dates=["Date"], index_col="Date")
    original_df = df.copy()
    df.drop(columns=["Open", "High", "Low"], inplace=True)  # Keep Date and Close column

    # Split the dataset into training and validation sets
    data_length = len(df)
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length

    # Scale the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Generate training data
    x_train_data = [scaled_data[i - 60 : i, 0] for i in range(60, train_data_length)]
    y_train_data = scaled_data[60:train_data_length, 0]

    # Prepare the test dataset
    inputs_data = scaled_data[train_data_length - 60 :]
    X_test = [inputs_data[i : i + 60, 0] for i in range(len(inputs_data) - 59)]

    # Prepare the validation dataset
    valid_data = df.iloc[train_data_length:].copy()
    next_day = valid_data.index[-1] + pd.DateOffset(days=1)
    valid_data.loc[next_day] = np.nan  # Append a row for the next day with NaN values

    # Convert lists to numpy arrays
    x_train_data, y_train_data, X_test = map(
        np.array, [x_train_data, y_train_data, X_test]
    )
    original_df = original_df.iloc[train_data_length + 1 :]
    return [x_train_data, y_train_data, X_test, valid_data, scaler, original_df]


# handle data for LSTM and RNN
def handle_data(name):
    """
    Preprocesses the data for stock prediction.

    Args:
        name (str): The name of the stock.

    Returns:
        list: A list containing the preprocessed data:
            - x_train_data (numpy.ndarray): The reshaped training data.
            - y_train_data (numpy.ndarray): The training labels.
            - X_test (numpy.ndarray): The reshaped test data.
            - valid_data (numpy.ndarray): The validation data.
            - scaler (object): The scaler used for normalization.
            - original_df (pandas.DataFrame): The original dataframe.

    """
    [x_train_data, y_train_data, X_test, valid_data, scaler, original_df] = preprocess_data(name)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return [x_train_data, y_train_data, X_test, valid_data, scaler, original_df]


# handle data for XGBoost
def handle_data_xgboost(name):
    return preprocess_data(name)


def preprocess_data_roc(name):
    """
    Preprocesses the data for rate of change (ROC) analysis.

    Args:
        name (str): The name of the CSV file containing the data.

    Returns:
        list: A list containing the preprocessed data:
            - x_train_data (numpy.ndarray): The input training data.
            - y_train_data (numpy.ndarray): The target training data.
            - X_test (numpy.ndarray): The input test data.
            - valid_data (pandas.DataFrame): The validation data.
            - scaler (sklearn.preprocessing.MinMaxScaler): The scaler used for normalization.
            - original_df (pandas.DataFrame): The original data before preprocessing.
    """
    df = pd.read_csv(name + ".csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df.set_index("Date", inplace=True)
    original_df = df.copy()
    df.drop(columns=["Open", "High", "Low"], inplace=True)
    df_copy = df
    df = df.pct_change(periods=1).dropna()

    dataset = df.values

    data_length = dataset.shape[0]
    validation_data_length = int(data_length * 0.1)
    train_data_length = data_length - validation_data_length
    valid_data = dataset[train_data_length:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train_data, y_train_data = [], []
    for i in range(60, len(scaled_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    X_test = []
    inputs_data = df[-len(valid_data) - 60 :].values.reshape(-1, 1)
    inputs_data = scaler.transform(inputs_data)

    X_test = [inputs_data[i - 60 : i, 0] for i in range(60, len(inputs_data) + 1)]
    X_test = np.array(X_test)

    valid_data = df_copy[train_data_length + 1 :].copy()
    next_day = valid_data.index[-1] + pd.DateOffset(days=1)
    valid_data.loc[next_day] = np.nan  # Append a row for the next day with NaN values

    original_df = original_df.iloc[train_data_length + 1 :]
    return [x_train_data, y_train_data, X_test, valid_data, scaler, original_df]


def handle_data_roc(name):
    [x_train_data, y_train_data, X_test, valid_data, scaler, original_df] = preprocess_data_roc(name)
    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return [x_train_data, y_train_data, X_test, valid_data, scaler, original_df]


def handle_data_roc_xgboost(name):
    return preprocess_data_roc(name)
