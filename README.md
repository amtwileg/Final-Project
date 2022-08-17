# Final Project
The goal of this project is to train a time series model to accurately predict stock prices for any generic stock.

## Database
We chose to use MongoDB as the database for this project. Each individual company's stock prices are stored in a separate MongoDB document. This was chosen because we do not need the relational aspect of SQL, so we embrace the simplicity of MongoDB.

See the [database notebook](db.ipynb) for more detail.

## Exploratory Data Analysis (EDA)
The data needed to be preprocessed in order to train a model. The following steps were performed:
1. Only the close data (and the date) is being used
2. Split data into train (90%) vs. validation (5%) vs. test (5%)
3. Features were normalized so that the training data has a maximum of 1 and a minimum of 0.
4. Data was reshaped from a single column to N columns, where N is the number of days we want to "lookback" for the time series data set. This value has been set at 20.

See the [exploratory data analysis notebook](EDA.ipynb) for more detail.

## Machine Learning Model
We chose to use an LSTM because this model has been shown to be quite good at modeling order-dependent interactions (e.g. time-series data).

The limitations of this model choice is that LSTM's typically perform well with a lot of data, and we don't have quite a lot of data available here. This model will perform better if we can acquire a larger volume of data.

The benefit of this model choice is that we don't have to spend as much time feature engineering as we would with other types of models. We simply need to normalize the data and transform it into a properly shaped dataframe and can then begin training / hyperparameter tuning until we achieve the desired performance.

The model has been trained for 50 epochs so far, using the Adam optimizer, and a learning rate of `1e-3`. According to the training plot, it may be beneficial to continue training the model, or try tuning some of the hyperparameters, such as the number of layers, etc.

The current accuracy of the model is a Mean Absolute Percentage Error (MAPE) of 2% for train and 3% for validation / test. This means that, using the past 20 days of close stock prices, I am able to predict the next day's closing stock price within 3% of the true value. A caveat is that tomorrow's stock price is typically quite close to the current day's stock price, which biases this metric optimistically.

See the [model training notebook](train.ipynb) for more detail.

## Dashboard
The app can be viewed [here](https://amtwileg-final-project-dashboard-a8c10q.streamlitapp.com/).

The dashboard allows the user to select a company's stock ticker from a list, and then customize hyperparameters and train a neural network to predict the stock price for the selected stock. It outputs a pair of visualizations and a few pertinent metrics after the model has been trained to better understand how the model is performing.

You can also deploy the app yourself, following the setup instructions below.

## Google Slides Presentation
The presentation can be found [here](https://docs.google.com/presentation/d/1MYcK-HIOGhLhr7A_eM8Kly_ar1-E3yFk4Y9WqkUbBaI/edit#slide=id.g147ae13f04a_0_2868).

## How to setup / run the app & notebooks locally
1. Clone repository: `git clone $PATH_TO_REPO`
2. Change working directory: `cd $REPO`
3. Install requirements: `pip install -r requirements.txt`
4. Run the code: Feel free to run any of the notebooks in order to play around with the results!
5. Run the app: `streamlit run dashboard.py` (or if you are running a virtual env, `venv/bin/streamlit run dashboard.py`)
