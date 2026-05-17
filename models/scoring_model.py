# scoring_model.py
import pandas as pd
import numpy as np

path = r'/content/drive/MyDrive/ML_project/content/sample_data/taiwan_transport_dataset.csv'
data_score = pd.read_csv(path)
print(data_score.info())
print(data_score.describe())

# ------ user input ------
origin = input("Origin:")
destination = input("Destination:")
people = int(input('Please enter the numbers of people(1~9):'))
luggage = int(input('Please enter the number of luggage:'))
budget = int(input('Please enter the budget(TWD per person):'))
time_pref = int(input('Please enter the time preference(1~10):'))  # 1(not important)~10(important)
cost_pref = int(input('Please enter the cost preference(1~10):'))  # 1(not important)~10(important)
conv_pref = int(input('Please enter the convenience preference(1~10):'))  # 1(not important)~10(important)

# ------ traffic way for input location ------
def find_traffic_way(origin, destination):
    traffic_way = data_score[(data_score['origin'] == origin) & (data_score['destination'] == destination)]
    return traffic_way

# ------ decision wieght ------
def get_weights(time_pref, cost_pref, conv_pref):
    total = time_pref + cost_pref + conv_pref

    w_time = time_pref / total
    w_cost = cost_pref / total
    w_conv = conv_pref / total

    return w_time, w_cost, w_conv

# ------ calculate score ------
def compute_score(row):
    w_time, w_cost, w_conv = get_weights(
        row["time_pref"],
        row["cost_pref"],
        row["conv_pref"]
    )
    # Note: since time and cost are better when smaller, we transform them using (1 − x) so that larger values represent better performance.
    return (
        w_time * (1 - row["time_n"]) +
        w_cost * (1 - row["cost_n"]) +
        w_conv * row["conv_n"]
    )

# ------ adjust convenience ------
def adjust_convenience(row, people, luggage):
    base = row["convenience"]
    mode = row["mode"]

    # luggage effect(more luggage, less convenience)
    if mode == "Bus":
        penalty = luggage * 0.5 # penalty = number of luggaga * wieght (influence strength)
    elif mode == "TRA":
        penalty = luggage * 0.3
    elif mode == "HSR":
        penalty = luggage * 0.25
    else:  # Car
        penalty = luggage * 0.1

    # people effect
    if mode == "Bus":
        penalty += people * 0.25

    elif mode == "TRA":
        penalty += people * 0.15

    elif mode == "HSR":
        penalty += people * 0.1

    else:  # Car (more people, more advantage)
        penalty -= people * 0.05

    return max(1, base - penalty) # adjust convenience > 0

# ------ normalization function ------
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# ------ main system ------
traffic_way = find_traffic_way(origin, destination)

# no result
if traffic_way.empty:

    print('\nNo transportation data found.')

else:
    traffic_way = traffic_way.copy()
    traffic_way["effective_cost"] = traffic_way["cost"]

    # Car: cost / people
    traffic_way.loc[
        traffic_way["mode"] == "Car",
        "effective_cost"
    ] = traffic_way["cost"] / people


    # budget limit
    traffic_way = traffic_way[traffic_way["effective_cost"] <= budget]

    # add user context
    traffic_way['people'] = people
    traffic_way['luggage'] = luggage
    traffic_way['budget'] = budget
    traffic_way['time_pref'] = time_pref
    traffic_way['cost_pref'] = cost_pref
    traffic_way['conv_pref'] = conv_pref

    # adjust convenience
    traffic_way['conv_adj'] = traffic_way.apply(
        lambda row: adjust_convenience(row, people, luggage),
        axis=1
    )

    # local normalization
    traffic_way['time_n'] = normalize(traffic_way['time_min'])
    traffic_way['cost_n'] = normalize(traffic_way['cost'])
    traffic_way['conv_n'] = normalize(traffic_way['conv_adj'])

    # score calculation
    traffic_way['score'] = traffic_way.apply(
        compute_score,
        axis=1
    )

    # ranking
    traffic_way = traffic_way.sort_values(
        by="score",
        ascending=False
    )

    print("\n==============================")
    print("Transportation Recommendation")
    print("==============================\n")

    for rank, (_, row) in enumerate(traffic_way.head(3).iterrows(), start=1):
        print(f"Rank {rank}")
        print(f"Mode      : {row['mode']}")
        print(f"Service   : {row['service']}")
        print(f"Time(min) : {row['time_min']}")
        print(f"Cost(TWD per person) : {row['effective_cost']:.0f}")
        print(f"Score     : {row['score']:.3f}")

    best = traffic_way.iloc[0]

    print("\n==============================")
    print("Best Recommendation")
    print("==============================")

    print(f"Mode    : {best['mode']}")
    print(f"Service : {best['service']}")
    print(f"Score   : {best['score']:.3f}")
