import pandas as pd

df = pd.read_csv("taiwan_route_dataset.csv")

print(df["route_name"].head())

