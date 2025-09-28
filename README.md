# Covid-19-Cases-Prediction
# COVID-19 Cases Forecasting and Visualization

This project analyzes and forecasts global COVID-19 confirmed cases and deaths using historical data. It utilizes the **Prophet** library for time series forecasting and generates various visualizations to explore the data, including geographical maps and time series plots.

---

## 📂 Project Structure

The repository contains the following key files:

| File Name | Description |
| :--- | :--- |
| `covid19_cases_forecasting.py` | The main Python script for data loading, preparation, visualization, and forecasting. |
| `CONVENIENT_global_confirmed_cases.csv` | Daily confirmed COVID-19 cases by country/region. |
| `CONVENIENT_global_deaths.csv` | Daily confirmed COVID-19 deaths by country/region. |
| `continents2.csv` | A mapping file to link countries to continents for geographical visualization. |
| `newplot.png` | Output image for the geographical visualization (Choropleth map). |
| `Figure_1.png` | Output image, likely showing the time series of confirmed cases or deaths. |
| `Figure_2.png` | Output image, possibly showing the forecast components (trend, weekly seasonality). |
| `Figure_3.png` | Output image, possibly showing the final forecast plot. |

---

## 🚀 Getting Started

### Prerequisites

You need **Python 3.x** installed. The project relies on several libraries, including **Prophet** (which has specific installation requirements), **pandas**, **numpy**, **matplotlib**, and **plotly**.

You can install the necessary Python packages using `pip`:

```bash
pip install pandas numpy matplotlib plotly prophet scikit-learn
Note: Installation of Prophet may require additional dependencies depending on your operating system.

Running the Script
Place the provided data files (.csv) and the Python script (.py) in the same directory.

Run the main script from your terminal:

Bash

python covid19_cases_forecasting.py
The script will:

Load and prepare the data.

Perform geographical and time series visualizations (which will be displayed and/or saved).

Run the Prophet model to forecast future cases/deaths.

Print the R2 Score of the forecasting model.

💡 Methodology and Key Features
Data Preparation
The raw data files are loaded, and country names are cleaned to ensure proper aggregation.

The data is structured for time series analysis, converting dates to the ds (date) and values to y (value) format required by Prophet.

Visualization
The script generates several visualizations:

Geographical Map: A choropleth map (using Plotly) showing the total number of cases per country.

Time Series Plots: Plots of the daily cases/deaths, including a moving average (MA5) to smooth the data.

Forecasting
The Prophet library is used for time series decomposition and forecasting.

The model is configured with weekly_seasonality=True, daily_seasonality=False, and yearly_seasonality=False.

A forecast for 30 future days is generated.

The model's performance is evaluated using the R2 score on the training data.

📈 Results
The key outputs of the script include:

R2 Score: 0.7063023981856846.

Forecast Plot (Figure_3.png): A visualization of the historical data and the 30-day forecast.

Component Plots (Figure_2.png): Separate plots for the trend and weekly seasonality components discovered by Prophet.

Geographical Plot (newplot.png): A map displaying the aggregated total cases per country.

