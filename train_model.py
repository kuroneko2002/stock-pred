from utils import (
    updateAllTicket,
    handle_data,
    handle_data_roc,
    handle_data_xgboost,
    handle_data_roc_xgboost,
)
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, Input
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor


def train_LSTM(x_train_data, y_train_data, name, units=50, epochs=1, batch_size=1):
    """
    Trains an LSTM model on the provided training data and saves the model.

    Parameters:
    - x_train_data: Features for training.
    - y_train_data: Target variable for training.
    - name: Suffix for the model filename.
    - units: Number of units for the LSTM layers.
    - epochs: Number of epochs to train for.
    - batch_size: Batch size for training.

    Returns:
    None
    """
    try:
        lstm_model = Sequential()
        lstm_model.add(
            Input(shape=(x_train_data.shape[1], 1))
        )  # Use Input layer to define input shape
        lstm_model.add(LSTM(units=units, return_sequences=True))
        lstm_model.add(LSTM(units=units))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss="mean_squared_error", optimizer="adam")

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor="loss", patience=10)

        lstm_model.fit(
            x_train_data,
            y_train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=[early_stopping],
        )
        lstm_model.save(f"models/lstm-{name}.h5")
    except Exception as e:
        print(f"Error training LSTM model: {e}")


def train_RNN(x_train_data, y_train_data, name, units=32, epochs=100, batch_size=150):
    """
    Trains a Simple RNN model on the provided training data and saves the model.

    Parameters:
    - x_train_data: Features for training.
    - y_train_data: Target variable for training.
    - name: Suffix for the model filename.
    - units: Number of units for the RNN layers.
    - epochs: Number of epochs to train for.
    - batch_size: Batch size for training.

    Returns:
    None
    """
    try:
        my_rnn_model = Sequential()
        my_rnn_model.add(
            Input(shape=(x_train_data.shape[1], 1))
        )  # Use Input layer to define input shape
        my_rnn_model.add(SimpleRNN(units, return_sequences=True))
        my_rnn_model.add(SimpleRNN(units))
        my_rnn_model.add(Dense(1))  # The time step of the output

        my_rnn_model.compile(optimizer="rmsprop", loss="mean_squared_error")

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor="loss", patience=10)

        my_rnn_model.fit(
            x_train_data,
            y_train_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping],
        )
        my_rnn_model.save(f"models/rnn{name}.h5")
    except Exception as e:
        print(f"Error training RNN model: {e}")


def train_XGBoost(
    x_train_data,
    y_train_data,
    name,
    objective="reg:squarederror",
    random_state=42,
    booster="gbtree",
):
    """
    Trains an XGBoost model on the provided training data and saves the model.

    Parameters:
    - x_train_data: Features for training.
    - y_train_data: Target variable for training.
    - name: Suffix for the model filename.
    - objective: Objective used for training the model.
    - random_state: Seed used by the random number generator.
    - booster: Specify which booster to use: gbtree, gblinear or dart.

    Returns:
    None
    """
    try:
        xgb = XGBRegressor(
            objective=objective, random_state=random_state, booster=booster
        )
        xgb.fit(x_train_data, y_train_data)
        xgb.save_model(f"models/xgb{name}.json")
    except Exception as e:
        print(f"Error training XGBoost model: {e}")


def train_models():
    updateAllTicket(period="1d")
    cryptos = ["BTC-USD", "ETH-USD", "ADA-USD"]
    data_types = ["", "roc"]
    model_functions = [train_LSTM, train_RNN, train_XGBoost]

    for crypto in cryptos:
        for data_type in data_types:
            if data_type == "":
                x_train_data, y_train_data, X_test, valid_data, scaler = handle_data(
                    crypto
                )
            else:
                x_train_data, y_train_data, X_test, valid_data, scaler = (
                    handle_data_roc(crypto)
                )

            for model_func in model_functions[:-1]:  # LSTM and RNN
                if data_type == "":
                    model_func(x_train_data, y_train_data, f"{crypto}")
                else:
                    model_func(x_train_data, y_train_data, f"{crypto}-{data_type}")

            if data_type == "":
                x_train_data, y_train_data, X_test, valid_data, scaler = (
                    handle_data_xgboost(crypto)
                )
            else:
                x_train_data, y_train_data, X_test, valid_data, scaler = (
                    handle_data_roc_xgboost(crypto)
                )

            if data_type == "":
                train_XGBoost(x_train_data, y_train_data, f"{crypto}")
            else:
                train_XGBoost(x_train_data, y_train_data, f"{crypto}-{data_type}")


if __name__ == "__main__":
    train_models()
