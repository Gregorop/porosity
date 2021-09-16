import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint,random

df = pd.read_csv("porosity_1408.csv",index_col=False)
#print(df.groupby("porosity_pixels")["red"].value_counts())
df.drop(["id","red","total_pixels","porosity_pixels"],axis=1,inplace=True)
df["% porosity"] = df["% porosity"]*100

def set_result(value):
    if value > 2:
        return 0
    return 1

#df["result"] = df["% porosity"].apply(set_result)
#df.drop(["% porosity"],axis=1,inplace=True)
#df[df["% porosity"]<2]["% porosity"].plot(kind="box")
#df = pd.get_dummies(df,prefix=["surface"])
def set_surface(value):
    if value == "bottom":
        return 1
    return 0

df["surface_bottom"] = df["surface"].apply(set_surface)
df.drop(["surface"],axis=1,inplace=True)
#sns.pairplot(df[df["% porosity"]<2],hue="surface")
#plt.show()

def calculate_density(value):
    return (value["power"] * (5/value["speed"])) / (2*3.14*( (0.1) **2))

df["power_density"] = df.apply(calculate_density,axis=1)
df["power_density"].plot(kind="box")

sns.pairplot(df,hue="surface_bottom")

plt.show()


"""
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop("% porosity",axis=1)
y = df["% porosity"]
 
X_train, X_test, Y_train, Y_test  = train_test_split(X, y, test_size = 0.05)


max_value_speed = 5000
max_value_power = 400

power_fill = []
for i in range(10000):
    power_fill.append(randint(50,max_value_power))

speed_fill = []
for i in range(10000):
    speed_fill.append(randint(50,max_value_speed))

future_speed = speed_fill
future_power = power_fill
future_front = [1]*len(future_speed)

future = {"speed":future_speed,
            "power":future_power,
                "surface_bottom":future_front}

future = pd.DataFrame.from_dict(future)
future["power_density"] = future.apply(calculate_density,axis=1)

normalaizer = StandardScaler()
X = normalaizer.fit_transform(X)
#X_train = normalaizer.fit_transform(X_train)
norm_future = normalaizer.transform(future)
#X_test  = normalaizer.transform(X_test)

brain = KNeighborsRegressor(n_neighbors=5)
brain.fit(X,y)

y_predict = brain.predict(norm_future)
future.insert(4,"predict",y_predict)
#print(future.head())

def roun(value):
    if value == 0:
        return "0"
    elif value < 0.5:
        return "0-0.5"
    elif value < 1:
        return "0.5-1"
    elif value < 1.5:
        return "1-1.5" 
    elif value < 2:
        return "1.5-2"
    else:
        return "more"

#future["predict"] = future["predict"].apply(roun)
print(future["power_density"])
sns.pairplot(future[future["predict"]<2],hue="predict",
                    x_vars=["speed"],
                    y_vars=["power"])

                    #palette={"0":"black","0-0.5":"orange","0.5-1":"green",
                    #        "1-1.5":"blue","1.5-2":"purple","more":"red"})
#sns.lmplot(data=future[future["predict"]<2],
#             x="speed", y="power", hue="predict")

#sns.jointplot(data=future[future["predict"]<2],
#                    x="speed", y="power", hue="predict")


plt.show()
#print(accuracy_score(Y_test,y_predict))
"""