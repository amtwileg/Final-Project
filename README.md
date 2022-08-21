# Final Project
The goal of this project is to train a time series model to accurately predict stock prices for any generic stock.

* **Selected topic**: Time series stock prediction
* **Reason why I selected this topic**: I am personally interested in stock prediction, as I spend my free time investing in stocks.
* **Description of source of data**: I am using publicly available stock ticker data, sourced from yahoo via the `pandas_datareader` library.
* **Questions I hope to answer with the data**:
  * Can stock price data be predicted using stock price history data and machine learning models?
  * Are certain stocks easier to predict than others?

## Communication protocols
The data used in this project is stock ticker data (time series). The raw data comes in the following format:

![example data](dataExampleScreenshot.png)

Data for any stock ticker can be pulled using this one-liner function, and then stored in a SQL (or mongo) database accordingly.
We will be focusing our analysis on using only the Date and Close columns.

## Technology
### Data cleaning and analysis
* `pandas_datareader` will be used to retrieve the data
* `pandas` / `numpy` will be used to perform exploratory analysis and format it into a structure conducive for time series machine learning algorithms.

### Database storage
* This database can easily be kept in memory and retrieved as needed.
* Alternatively, if the data proves too large to fit into memory, we can store the data in a sqlite or psql database. Another alternative is mongo, since we do not need the relational aspect of sql and it will be simpler.
* See [main](main.ipynb) for an example of a sample database in action with this data.

### Machine Learning
* `sklearn` / `pytorch` / `keras` will be used to build the machine learning time series models. We may also use `statsmodels` to fit an STL model.
* See [main](main.ipynb) for an example of a sample (dummy) machine learning model in action with this data.

### Dashboard
* I plan on using `streamlit` to build a dashboard.

## Data Exploration
* The time series data did not come in a format that is conducive for machine learning.
* I decided to only use the "Closed" column of the data, and normalized / reshaped it so a machine learning model could be used.
* See the [EDA notebook](EDA.ipynb) for details about the exploration
