from neurobiba import Weights
import pandas as pd

df = pd.read_csv("porosity_1408.csv",index_col=False)
df.drop(["id","red","total_pixels","porosity_pixels"],axis=1,inplace=True)

df["porosity"] = df["% porosity"]*100

def speed_normalizer(speed):
    n = (speed-1500)/2000
    return max(0,min(n,1))

def power_normalizer(power):
    n = (power-150)/200
    return max(0,min(n,1))

df["speed"] = df["speed"].apply(speed_normalizer)
df["power"] = df["power"].apply(power_normalizer)


weights = Weights([2,10,10,1])

for i in range(len(df["porosity"])):
    speed, power = df["speed"][i], df["power"][i]
    output = df["porosity"][i]
    weights.train([speed, power], [output]) # train

result = weights.feed_forward([0.5, 0.3])[0] # result is close to 0
print(result)