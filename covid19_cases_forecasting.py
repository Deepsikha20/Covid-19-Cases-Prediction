import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import r2_score

plt.style.use("ggplot")

# ------------------ Data Loading ------------------
df_cases = pd.read_csv("CONVENIENT_global_confirmed_cases.csv")
df_deaths = pd.read_csv("CONVENIENT_global_deaths.csv")
df_continents = pd.read_csv("continents2.csv")
df_continents["name"] = df_continents["name"].str.upper()

# ------------------ Data Preparation ------------------
world = pd.DataFrame({"Country":[], "Cases":[]})
world["Country"] = df_cases.iloc[:,1:].columns
cases = []
for country in world["Country"]:
    cases.append(pd.to_numeric(df_cases[country][1:]).sum())
world["Cases"] = cases

country_list = list(world["Country"].values)
for idx, country in enumerate(country_list):
    for i, char in enumerate(country):
        if char == ".":
            country_list[idx] = country[:i]
            break
        elif char == "(":
            country_list[idx] = country[:i-1]
            break
world["Country"] = country_list
world = world.groupby("Country")["Cases"].sum().reset_index()

# ------------------ Geographical Visualization ------------------
world["Cases Range"] = pd.cut(world["Cases"], [-150000,50000,200000,800000,1500000,15000000],
                              labels=["U50K","50Kto200K","200Kto800K","800Kto1.5M","1.5M+"])
alpha = []
for name in world["Country"].str.upper().values:
    if name == "BRUNEI":
        name = "BRUNEI DARUSSALAM"
    elif name == "US":
        name = "UNITED STATES"
    alpha_val = df_continents[df_continents["name"]==name]["alpha-3"].values
    alpha.append(alpha_val[0] if len(alpha_val)>0 else np.nan)
world["Alpha3"] = alpha

fig = px.choropleth(world.dropna(),
                   locations="Alpha3",
                   color="Cases Range",
                   projection="mercator",
                   color_discrete_sequence=["white","khaki","yellow","orange","red"])
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

# ------------------ Daily Cases and Deaths Visualization ------------------
cases_count = [sum(pd.to_numeric(df_cases.iloc[i,1:].values)) for i in range(1, len(df_cases))]
df = pd.DataFrame()
df["Date"] = df_cases["Country/Region"][1:]
df["Cases"] = cases_count
df = df.set_index("Date")

deaths_count = [sum(pd.to_numeric(df_deaths.iloc[i,1:].values)) for i in range(1, len(df_deaths))]
df["Deaths"] = deaths_count

df["Cases"].plot(title="Daily Covid19 Cases in World", marker=".", figsize=(10,5), label="daily cases")
df["Cases"].rolling(window=5).mean().plot(figsize=(10,5), label="MA5")
plt.ylabel("Cases")
plt.legend()
plt.show()

df["Deaths"].plot(title="Daily Covid19 Deaths in World", marker=".", figsize=(10,5), label="daily deaths")
df["Deaths"].rolling(window=5).mean().plot(figsize=(10,5), label="MA5")
plt.ylabel("Deaths")
plt.legend()
plt.show()

# ------------------ Time Series Forecasting with Prophet ------------------
class Fbprophet(object):
    def fit(self, data):
        self.data = data
        self.model = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        self.model.fit(self.data)
    
    def forecast(self, periods, freq):
        self.future = self.model.make_future_dataframe(periods=periods, freq=freq)
        self.df_forecast = self.model.predict(self.future)
    
    def plot(self, xlabel="Years", ylabel="Values"):
        self.model.plot(self.df_forecast, xlabel=xlabel, ylabel=ylabel, figsize=(9,4))
        self.model.plot_components(self.df_forecast, figsize=(9,6))
    
    def R2(self):
        return r2_score(self.data.y, self.df_forecast.yhat[:len(self.data)])

df_fb = pd.DataFrame()
df_fb["ds"] = pd.to_datetime(df.index)
df_fb["y"] = df.iloc[:,0].values

model = Fbprophet()
model.fit(df_fb)
model.forecast(30, "D")
print("Model R2 Score:", model.R2())

forecast = model.df_forecast[["ds","yhat_lower","yhat_upper","yhat"]].tail(30).reset_index().set_index("ds").drop("index", axis=1)
forecast["yhat"].plot(marker=".", figsize=(10,5))
plt.fill_between(x=forecast.index, y1=forecast["yhat_lower"], y2=forecast["yhat_upper"], color="gray")
plt.legend(["forecast", "Bound"], loc="upper left")
plt.title("Forecasting of Next 30 Days Cases")
plt.show()
