from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import streamlit as st
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error


@st.cache
def get_data(stock: str, lookback: int) -> Tuple[
    np.array, np.array, np.array, np.array, np.array, np.array, float, float, int, pd.DataFrame]:
    df = pdr.get_data_yahoo(stock)
    df = df[['Close']]
    TEST_SIZE = 0.05
    VALIDATION_SIZE = 0.05
    TRAIN_SIZE = 1 - TEST_SIZE - VALIDATION_SIZE

    train_ending_index = int(len(df) * TRAIN_SIZE)
    validation_ending_index = train_ending_index + int(len(df) * VALIDATION_SIZE)

    train_df = df[:train_ending_index]
    validation_df = df[train_ending_index:validation_ending_index]
    test_df = df[validation_ending_index:]

    train_max = train_df['Close'].max()
    train_min = train_df['Close'].min()

    train_df_scaled = (train_df['Close'] - train_min) / (train_max - train_min)
    validation_df_scaled = (validation_df['Close'] - train_min) / (train_max - train_min)
    test_df_scaled = (test_df['Close'] - train_min) / (train_max - train_min)

    def create_dataset(dataset, look_back):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            X.append(a)
            Y.append(dataset[i + look_back])
        X = np.asarray(X).astype(np.float32)
        Y = np.asarray(Y).astype(np.float32)
        return np.array(X).reshape(X.shape[0], X.shape[1], 1), np.array(Y)

    x_train, y_train = create_dataset(train_df_scaled, look_back=lookback)
    x_validation, y_validation = create_dataset(validation_df_scaled, look_back=lookback)
    x_test, y_test = create_dataset(test_df_scaled, look_back=lookback)

    return x_train, y_train, x_validation, y_validation, x_test, y_test, train_min, train_max, validation_ending_index, df


def train_model(
        layer_sizes: Tuple[int, int],
        dropout: float,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
        x_train: np.array,
        y_train: np.array,
        x_validation: np.array,
        y_validation: np.array
):
    model = Sequential()

    # Adding the LSTM layer and some Dropout regularisation
    input_shape = x_train.shape[1]
    for i, layer_size in enumerate(layer_sizes):
        model.add(LSTM(layer_size, return_sequences=i < (len(layer_sizes) - 1), input_shape=(input_shape, 1)))
        model.add(Dropout(dropout))
        input_shape = layer_size

    # Adding the output layer
    # For Full connection layer we use dense
    # As the output is 1D so we use unit=1
    model.add(Dense(1))

    # Compiling the RNN-LSTM
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # Fitting the RNN to the Training set
    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                        validation_data=(x_validation, y_validation),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=True)

    # Show the model summary
    print(model.summary())
    return model, history


def evaluate_model(model, data, labels, train_min, train_max, prefix):
    predictions = model.predict(data)
    unscaled_predictions = predictions * (train_max - train_min) + train_min
    unscaled_labels = labels * (train_max - train_min) + train_min
    st.write(f'{prefix} Mean Absolute Error (MAE):', mean_absolute_error(unscaled_labels, unscaled_predictions[:, 0]))
    st.write(f'{prefix} Root Mean Squared Error (RMSE):',
             np.sqrt(mean_squared_error(unscaled_labels, unscaled_predictions[:, 0])))
    st.write(f'{prefix} Mean Absolute Percentage Error (MAPE):',
             mean_absolute_percentage_error(unscaled_labels, unscaled_predictions[:, 0]))


def plot_training_curves(history):
    # Plot the model loss
    fig = plt.figure(figsize=(12, 6))

    fig.add_subplot(111)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')

    st.pyplot(fig)


def plot_stock_with_predictions(stock: str, lookback: int, train_min: float, train_max: float, model,
                                validation_ending_index: int, x_train, y_train, x_test, y_test):
    x = df[:validation_ending_index + 1].index.values

    # prepare empty array with 'lb' size
    emp = np.empty(lookback, dtype='object')

    def get_scaled_predictions(model, data, train_min, train_max):
        predictions = model.predict(data)
        return predictions * (train_max - train_min) + train_min

    train_predict = get_scaled_predictions(model, x_train, train_min, train_max)
    test_predict = get_scaled_predictions(model, x_test, train_min, train_max)

    # combine the Train-Test results
    allData = np.concatenate(
        (emp, y_train * (train_max - train_min) + train_min, emp, y_test * (train_max - train_min) + train_min))
    allPred = np.concatenate((emp, train_predict[:, 0].reshape(train_predict.shape[0], ),
                              emp, test_predict[:, 0].reshape(test_predict.shape[0], )))

    # plot the actual and prediction results
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    length = min(x.size, allData.size, allPred.size)
    ax.plot(x[:length], allData[:length], label="Actual Price")
    ax.plot(x[:length], allPred[:length], 'r', label="Predicted Price")
    ax.legend()

    # title
    ax.set_title("STOCK NAME - " + stock, fontsize=24, fontweight='bold')

    # axis title
    ax.set_xlabel('TIME', fontsize=15)
    ax.set_ylabel('CLOSE PRICE', fontsize=15)

    st.pyplot(fig)


st.title('Stock Price Prediction with Keras')

with st.sidebar:
    selected_stock = st.selectbox(
        "Select a technology stock to analyze:",
        ("GOOG", "MSFT", "META", "AMZN", "AAPL", "NFLX")
    )
    selected_lookback = st.slider(
        "Select number of days to lookback:",
        min_value=2,
        max_value=100,
        value=20
    )

    selected_neurons_per_layer = st.slider(
        "Select number of neurons per layer:",
        min_value=8,
        max_value=200,
        value=100,
    )

    selected_num_layers = st.slider(
        "Select number of hidden layers:",
        min_value=1,
        max_value=8,
        value=2
    )

    selected_dropout = st.slider(
        "Select level of dropout:",
        min_value=0.0,
        max_value=1.0,
        value=0.2
    )

    selected_num_epochs = st.slider(
        "Select number of epochs to train for:",
        min_value=1,
        max_value=100,
        value=50
    )

    selected_batch_size = st.slider(
        "Select batch size to use for training:",
        min_value=2,
        max_value=256,
        value=32
    )

if st.button(
        "Train model (may take some time to run)"
):
    with st.spinner("Training model..."):
        x_train, y_train, x_validation, y_validation, x_test, y_test, train_min, train_max, validation_ending_index, df = get_data(
            stock=selected_stock,
            lookback=selected_lookback)
        model, history = train_model(
            layer_sizes=[selected_neurons_per_layer for _ in range(selected_num_layers)],
            dropout=selected_dropout,
            learning_rate=1e-3,
            num_epochs=selected_num_epochs,
            batch_size=selected_batch_size,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation
        )
    st.success("Model trained!")

    evaluate_model(model, x_train, y_train, train_min, train_max, 'Train')
    evaluate_model(model, x_validation, y_validation, train_min, train_max, 'Validation')
    evaluate_model(model, x_test, y_test, train_min, train_max, 'Test')
    plot_training_curves(history=history)
    plot_stock_with_predictions(
        stock=selected_stock,
        lookback=selected_lookback,
        train_min=train_min,
        train_max=train_max,
        model=model,
        validation_ending_index=validation_ending_index,
        x_train=x_train,
        y_train=y_train,
        x_test=x_validation,
        y_test=y_validation,
    )
