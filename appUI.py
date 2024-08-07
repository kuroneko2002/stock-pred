import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import numpy as np
from module.utils import (
    handle_data,
    handle_data_roc,
    handle_data_xgboost,
    handle_data_roc_xgboost,
)

app = dash.Dash(__name__)
server = app.server

# Load models
models = {
    "btc": {
        "lstm": load_model("models/lstm-BTC-USD.h5"),
        "rnn": load_model("models/rnnBTC-USD.h5"),
        "xgb": xgb.XGBRegressor(),
    },
    "btc_roc": {
        "lstm": load_model("models/lstm-BTC-USD-roc.h5"),
        "rnn": load_model("models/rnnBTC-USD-roc.h5"),
        "xgb": xgb.XGBRegressor(),
    },
    "eth": {
        "lstm": load_model("models/lstm-ETH-USD.h5"),
        "rnn": load_model("models/rnnETH-USD.h5"),
        "xgb": xgb.XGBRegressor(),
    },
    "eth_roc": {
        "lstm": load_model("models/lstm-ETH-USD-roc.h5"),
        "rnn": load_model("models/rnnETH-USD-roc.h5"),
        "xgb": xgb.XGBRegressor(),
    },
    "ada": {
        "lstm": load_model("models/lstm-ADA-USD.h5"),
        "rnn": load_model("models/rnnADA-USD.h5"),
        "xgb": xgb.XGBRegressor(),
    },
    "ada_roc": {
        "lstm": load_model("models/lstm-ADA-USD-roc.h5"),
        "rnn": load_model("models/rnnADA-USD-roc.h5"),
        "xgb": xgb.XGBRegressor(),
    },
}

# Load XGB models
models["btc"]["xgb"].load_model("models/xgbBTC-USD.json")
models["btc_roc"]["xgb"].load_model("models/xgbBTC-USD-roc.json")
models["eth"]["xgb"].load_model("models/xgbETH-USD.json")
models["eth_roc"]["xgb"].load_model("models/xgbETH-USD-roc.json")
models["ada"]["xgb"].load_model("models/xgbADA-USD.json")
models["ada_roc"]["xgb"].load_model("models/xgbADA-USD-roc.json")


# Process data
def process_data(currency, name):
    """
    Process the data for the given currency and model name.

    Args:
        currency (str): The currency to process the data for.
        name (str): The name of the model.

    Returns:
        tuple: A tuple containing the processed data and the original data.
    """
    [_, _, X_test, valid_data, scaler, df] = handle_data(currency)
    [_, _, X_xgb_test, _, _, _] = handle_data_xgboost(currency)

    lstm_pred = models[name]["lstm"].predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)

    rnn_pred = models[name]["rnn"].predict(X_test)
    rnn_pred = scaler.inverse_transform(rnn_pred)

    xgb_pred = models[name]["xgb"].predict(X_xgb_test)
    xgb_pred = np.reshape(xgb_pred, (xgb_pred.shape[0], 1))
    xgb_pred = scaler.inverse_transform(xgb_pred)

    valid_data["Predictions-lstm"] = lstm_pred
    valid_data["Predictions-rnn"] = rnn_pred
    valid_data["Predictions-xgb"] = xgb_pred

    return valid_data, df


def process_data_roc(currency, name):
    """
    Process the data for rate of change (ROC) prediction.

    Args:
        currency (str): The currency to process the data for.
        name (str): The name of the model to use for prediction.

    Returns:
        tuple: A tuple containing the processed data for ROC prediction and the original data.

    """
    [_, _, X_test, valid_data, scaler, df] = handle_data_roc(currency)
    [_, _, X_xgb_test, _, _, _] = handle_data_roc_xgboost(currency)

    lstm_pred = models[name]["lstm"].predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred)

    A = valid_data.shift(1)["Close"]
    lstm_pred = lstm_pred.reshape(-1)
    B = lstm_pred * A
    valid_data["Predictions-lstm"] = valid_data.shift(1)["Close"] + B

    rnn_pred = models[name]["rnn"].predict(X_test)
    rnn_pred = scaler.inverse_transform(rnn_pred)

    A = valid_data.shift(1)["Close"]
    rnn_pred = rnn_pred.reshape(-1)
    B = rnn_pred * A
    valid_data["Predictions-rnn"] = valid_data.shift(1)["Close"] + B

    xgb_pred = models[name]["xgb"].predict(X_xgb_test)
    xgb_pred = np.reshape(xgb_pred, (xgb_pred.shape[0], 1))
    xgb_pred = scaler.inverse_transform(xgb_pred)

    A = valid_data.shift(1)["Close"]
    xgb_pred = xgb_pred.reshape(-1)
    B = xgb_pred * A
    valid_data["Predictions-xgb"] = valid_data.shift(1)["Close"] + B

    return valid_data, df


# Prepare data for plotting
data = {
    "btc": process_data("BTC-USD", "btc"),
    "btc_roc": process_data_roc("BTC-USD", "btc_roc"),
    "eth": process_data("ETH-USD", "eth"),
    "eth_roc": process_data_roc("ETH-USD", "eth_roc"),
    "ada": process_data("ADA-USD", "ada"),
    "ada_roc": process_data_roc("ADA-USD", "ada_roc"),
}

# Layout
app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="BTC-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="btc-dropdown",
                                    options=[
                                        {"label": "Closed", "value": "Closed"},
                                        {"label": "RoC", "value": "Roc"},
                                    ],
                                    value="Closed",
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="btc-lstm"),
                                html.H1(
                                    "RNN Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="btc-rnn"),
                                html.H1(
                                    "XGB Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="btc-xgb"),
                            ],
                            className="container",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="ETH-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="eth-dropdown",
                                    options=[
                                        {"label": "Closed", "value": "Closed"},
                                        {"label": "RoC", "value": "Roc"},
                                    ],
                                    value="Closed",
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="eth-lstm"),
                                html.H1(
                                    "RNN Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="eth-rnn"),
                                html.H1(
                                    "XGB Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="eth-xgb"),
                            ],
                            className="container",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="ADA-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "LSTM Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="ada-dropdown",
                                    options=[
                                        {"label": "Closed", "value": "Closed"},
                                        {"label": "RoC", "value": "Roc"},
                                    ],
                                    value="Closed",
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="ada-lstm"),
                                html.H1(
                                    "RNN Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="ada-rnn"),
                                html.H1(
                                    "XGB Predicted closing price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="ada-xgb"),
                            ],
                            className="container",
                        ),
                    ],
                ),
            ],
        ),
    ]
)


def get_figure(dropdown_value, currency, model_type, title):
    """
    Generate a figure object for plotting candlestick and predicted close data.

    Parameters:
    - dropdown_value (str): The selected dropdown value ("Closed" or "ROC").
    - currency (str): The currency for which the data is fetched.
    - model_type (str): The type of model used for predictions.
    - title (str): The title of the figure.

    Returns:
    - figure (dict): A dictionary containing the data and layout for the figure.
    """

    valid_data, df = data[currency]
    valid_data_roc, _ = data[currency + "_roc"]
    selected_data = valid_data if dropdown_value == "Closed" else valid_data_roc

    actual_close = go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Actual",
    )
    predicted_close = go.Scatter(
        x=selected_data.index,
        y=selected_data[f"Predictions-{model_type}"],
        name="Predicted",
        mode="text",
        text="-",
        textfont=dict(size=25, color="blue"),
        legendgroup="predicted",
        showlegend=False
    )
    predicted_close_marker = go.Scatter(
        x=[None],  # No data points to display
        y=[None],
        mode="markers",  # Only markers
        marker=dict(color="blue", symbol="diamond"),
        name="Predicted",
        legendgroup="predicted",  # Same group as the text trace
        showlegend=True  # Show this trace in the legend
    )

    figure = {
        "data": [actual_close, predicted_close, predicted_close_marker],
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=title,
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {
                                "count": 1,
                                "label": "1M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {
                                "count": 6,
                                "label": "6M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
            },
            yaxis={"title": "Price"},
        ),
    }
    return figure


@app.callback(Output("btc-lstm", "figure"), Input("btc-dropdown", "value"))
def update_btc_lstm(dropdown_value):
    return get_figure(dropdown_value, "btc", "lstm", "BTC LSTM Predicted Price")


@app.callback(Output("btc-rnn", "figure"), Input("btc-dropdown", "value"))
def update_btc_rnn(dropdown_value):
    return get_figure(dropdown_value, "btc", "rnn", "BTC RNN Predicted Price")


@app.callback(Output("btc-xgb", "figure"), Input("btc-dropdown", "value"))
def update_btc_xgb(dropdown_value):
    return get_figure(dropdown_value, "btc", "xgb", "BTC XGB Predicted Price")


@app.callback(Output("eth-lstm", "figure"), Input("eth-dropdown", "value"))
def update_eth_lstm(dropdown_value):
    return get_figure(dropdown_value, "eth", "lstm", "ETH LSTM Predicted Price")


@app.callback(Output("eth-rnn", "figure"), Input("eth-dropdown", "value"))
def update_eth_rnn(dropdown_value):
    return get_figure(dropdown_value, "eth", "rnn", "ETH RNN Predicted Price")


@app.callback(Output("eth-xgb", "figure"), Input("eth-dropdown", "value"))
def update_eth_xgb(dropdown_value):
    return get_figure(dropdown_value, "eth", "xgb", "ETH XGB Predicted Price")


@app.callback(Output("ada-lstm", "figure"), Input("ada-dropdown", "value"))
def update_ada_lstm(dropdown_value):
    return get_figure(dropdown_value, "ada", "lstm", "ADA LSTM Predicted Price")


@app.callback(Output("ada-rnn", "figure"), Input("ada-dropdown", "value"))
def update_ada_rnn(dropdown_value):
    return get_figure(dropdown_value, "ada", "rnn", "ADA RNN Predicted Price")


@app.callback(Output("ada-xgb", "figure"), Input("ada-dropdown", "value"))
def update_ada_xgb(dropdown_value):
    return get_figure(dropdown_value, "ada", "xgb", "ADA XGB Predicted Price")


def run_app():
    app.run_server(debug=True)


if __name__ == "__main__":
    run_app()
