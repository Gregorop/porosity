from neurobiba import Weights
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint

df = pd.read_csv("porosity_1408.csv",index_col=False)
df.drop(["id","red","total_pixels","porosity_pixels"],axis=1,inplace=True)

df["porosity"] = df["% porosity"]

def speed_normalizer(speed):
    n = (speed-1500)/2000
    return max(0,min(n,1))

def power_normalizer(power):
    n = (power-150)/200
    return max(0,min(n,1))

def calculate_density(value):
    return int((value["power"] * (5/value["speed"]))/ (2*3.14*((5**-6)**2)))

def density_normalizer(density):
    n = (density-8330549)/37024662
    return max(0,min(n,1))

df["power_density"] = df.apply(calculate_density,axis=1)

df["speed"] = df["speed"].apply(speed_normalizer)
df["power"] = df["power"].apply(power_normalizer)
df["power_density"] = df["power_density"].apply(density_normalizer)


weights = Weights([3,5,1])

for n in range(1000):
    for i in range(len(df["porosity"])):
        speed, power, density = df["speed"][i], df["power"][i], df["power_density"][i]
        output = df["porosity"][i]
        weights.train([speed, power,density], [output]) # train

print('train прошелся')

speeds = []
powers = []
densitys = []
s = 1500
while s<=3500:
    p = 150
    while p<=350:
        speeds.append(randint(1500,3500))
        powers.append(randint(150,350))
        densitys.append(int((p * (5/s))/ (2*3.14*((5**-6)**2))))
        p+=2
    s+=10

print('while прошелся')

future = {"speed": speeds,
            "power":powers,
             "power_density":densitys}

future = pd.DataFrame.from_dict(future)

future["speed"] = future["speed"].apply(speed_normalizer)
future["power"] = future["power"].apply(power_normalizer)
future["power_density"] = future["power_density"].apply(density_normalizer)

predict = list()

for i in range(len(future)):
    predict.append(weights.feed_forward([future.iloc[i]["speed"],
                                         future.iloc[i]["power"],
                                         future.iloc[i]["power_density"]])[0])

future.insert(3,"porosity",predict)

def speed_denorm(value):
    return value * 2000 + 1500

def power_denorm(value):
    return value * 200 + 150

future["speed"] = future["speed"].apply(speed_denorm)
future["power"] = future["power"].apply(power_denorm)
future["porosity"] = future["porosity"]*100

sns.pairplot(future[future["porosity"]<2],hue="porosity",
                    x_vars=["speed"],
                    y_vars=["power"])

plt.show()