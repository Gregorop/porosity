import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import randint,random
from math import pi

modes = [{"speed":0,"power":0},
        {"speed":2000,"power":150},
        {"speed":2000,"power":200},
        {"speed":2000,"power":250},
        {"speed":2000,"power":300},
        {"speed":2000,"power":350},
        {"speed":1500,"power":200},
        {"speed":2000,"power":250},
        {"speed":1250,"power":250},
        {"speed":1000,"power":200},
        {"speed":1250,"power":200},
        {"speed":1000,"power":250},
        {"speed":1500,"power":250},
        {"speed":1750,"power":200},
        {"speed":2000,"power":200},
        {"speed":1750,"power":250}]

porosity = [{"speed":0,"power":0,"porosity":0},
            {"speed":2000,"power":150,"porosity":19.6},
            {"speed":2000,"power":200,"porosity":2.89},
            {"speed":2000,"power":250,"porosity":0.23},
            {"speed":2000,"power":300,"porosity":0.81},
            {"speed":2000,"power":350,"porosity":0.52},
            {"speed":1500,"power":200,"porosity":0.19},
            {"speed":2000,"power":250,"porosity":0},
            {"speed":1250,"power":250,"porosity":0},
            {"speed":1000,"power":200,"porosity":0},
            {"speed":1250,"power":200,"porosity":0},
            {"speed":1000,"power":250,"porosity":0},
            {"speed":1500,"power":250,"porosity":0},
            {"speed":1750,"power":200,"porosity":0},
            {"speed":2000,"power":200,"porosity":2.89},
            {"speed":1750,"power":250,"porosity":0}]

addition_porosty = [{"power":200,"speed":2500,"porosity":10.1,"top":0,"bottom45":0,"side":1,"rough":0},
                    {"power":200,"speed":3000,"porosity":14.3,"top":0,"bottom45":0,"side":1,"rough":0},
                    {"power":200,"speed":3500,"porosity":21.9,"top":0,"bottom45":0,"side":1,"rough":0}]

df = pd.read_csv("al_rough_2508.csv",index_col=False)
sur = pd.get_dummies(df["surface"])

df["top"] = sur["top"]
df["bottom45"] = sur["Bottom 45"]
df["side"] = sur["side"]
#df.drop("surface",inplace=True,axis=1)

def set_rough(value):
    return float(value.replace(",","."))

def set_speed(value):
    return modes[value]["speed"]

def set_power(value):
    return modes[value]["power"]

def set_porosity(value):
    for pair in porosity:
        if value["speed"] == pair["speed"] and value["power"] == pair["power"]:
            if value["side"]:
                return pair["porosity"]
            return 0

def calculate_power_density(value):
    speed_m_per_s = value["speed"]/1000
    area = pi*((0.0001/2)**2) #0.0001 это ДИАМЕТР пучка, берем половину?
    impact_time = 0.0001/speed_m_per_s
    joule = value["power"]*impact_time
    final = joule/area
    return final
  
df["rough"] = df["rough"].apply(set_rough)
df["speed"] = df["mode"].apply(set_speed)
df["power"] = df["mode"].apply(set_power)
df["porosity"] = df.apply(set_porosity,axis=1)

df["power_density"] = df.apply(calculate_power_density,axis=1)

#print("{:e}".format(df["power_density"].max()))

df.drop("mode",inplace=True,axis=1)

#df[df["porosity"]>0].plot(kind="scatter",x="rough",y="porosity")

#sns.pairplot(,x_vars=["power_density"],y_vars=["shit"])

#df["power_density"].plot(kind="box")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score

df.drop("surface",inplace=True,axis=1)

X = df[df["porosity"]!=0].drop("porosity",axis=1)
y = df[df["porosity"]!=0]["porosity"]
 
X_train, X_test, Y_train, Y_test  = train_test_split(X, y, test_size = 0.05)

normalaizer = StandardScaler()

X = df.drop("porosity",axis=1)
X = normalaizer.fit_transform(X)
X_train = normalaizer.fit_transform(X_train)

X_test  = normalaizer.transform(X_test)

brain = KNeighborsRegressor(n_neighbors=3)
brain.fit(X_train,Y_train)

porosity_predict = brain.predict(X)

df.insert(8,"porosity_predict",porosity_predict)

def final_porosity(value):
    if value["porosity"]:
        return value["porosity"]
    else:
        return value["porosity_predict"]

#в pandas теперь так
new_df = pd.DataFrame(addition_porosty)
df = pd.concat([df, new_df], ignore_index=True)

df["power_density"] = df.apply(calculate_power_density,axis=1)
df["final_porosity"] = df.apply(final_porosity,axis=1)
df.drop("porosity_predict",inplace=True,axis=1)

X = df[df["rough"]!=0].drop("rough",axis=1)
y = df[df["rough"]!=0]["rough"]
 
X_train, X_test, Y_train, Y_test  = train_test_split(X, y, test_size = 0.05)

normalaizer = StandardScaler()

X = df.drop("rough",axis=1)
X = normalaizer.fit_transform(X)
X_train = normalaizer.fit_transform(X_train)

X_test  = normalaizer.transform(X_test)

brain = KNeighborsRegressor(n_neighbors=3)
brain.fit(X_train,Y_train)

rough_predict = brain.predict(X)

df.insert(9,"rough_predict",rough_predict)

def final_rough(value):
    if value["rough"]:
        return value["rough"]
    else:
        return value["rough_predict"]

df["final_rough"] = df.apply(final_rough,axis=1)
df.drop("rough_predict",inplace=True,axis=1)

def normalize_rough_porosity(df):
    result = df.copy()
    for feature_name in ["final_porosity","final_rough"]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

norm_df = normalize_rough_porosity(df)

def estimate_poor(value):
    return (value["final_rough"]+value["final_porosity"])

df["poor_surface"] = norm_df.apply(estimate_poor,axis=1)

def set_surface(value):
    if value["side"]:
        return "side"
    elif value["top"]:
        return "top"
    else:
        return "surface45"

df["surface"] = df.apply(set_surface,axis=1)

#sns.pairplot(df[df["surface"]=="surface45"], hue="poor_surface",
#                    x_vars=["power"],
#                    y_vars=["speed"])

#plt.show()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-##########################
# d частиц 30 мкм == 0.03мм
#r 15 мкм == 0.15мм

#laser r 50мкм = 0.05 мм
#laser d = 100мкм == 0.1 мм

particle_d, particle_r = 30*10**-6, 15*10**-6
laser_d, laser_r = 100*10**-6, 50*10**-6

"""
т.е в лазерный радиус помещается 1 целая частица
и на 2/3 целой частицы заезжает лазер (20мкм от 30мкм)
**pic 1 синим**

эту зону обозначим КРАЙ NEAR EDGE
"""

#теплоемкость как функция **pic3**
al_C_heat_energy = {27:903,127:950,227:990,327:1036,427:1090,527:1153,627:1288,727:1176}

class Fuield:
    def __init__(self,volume,T,heat_energy=al_C_heat_energy[27]):
        self.density  = 2700 #из kg/м³
        self.volume = volume
        self.T = T
        self.mass = self.volume * self.density
        self.heat_energy = heat_energy
        self.energy = 0

    def increase_energy(self,joule):
        self.energy += joule
    
    def increase_t(self,time=1):
        global n_particles
        while self.energy > 0:
            self.energy -= (self.heat_energy*self.mass)
            #print(self.energy)
            self.T += 1
            self.check_heat_energy()
        if self != far_edge:
            self.calculate_heat_flux(far_edge,time)
            
    def decrease_t(self):
        while self.energy < 0:
            self.energy += (self.heat_energy*self.mass)
            #print(self.energy)
            self.T -= 1
            self.check_heat_energy()

    def calculate_heat_flux(self,target,time):
        next_particle = target
        particle = self
        diff_T = max(0,self.T - next_particle.T)
        distance = 30
        pass_energy = (190 * time * distance * 10**-6) * int(diff_T)
        #print(pass_energy)
        next_particle.increase_energy(pass_energy)
        next_particle.increase_t()
        particle.energy -= pass_energy
        particle.decrease_t()

    def calculate_heat_flux_throu_azot(self,time):
        azotT = 0
        diff_T = max(0,self.T - azotT)
        azot_distance = 10
        azot_termal_conduct = 0.02
        pass_energy = (azot_termal_conduct * time * azot_distance * 10**-6) * int(diff_T)
        #print(pass_energy)
        return pass_energy

    def check_heat_energy(self):
        for t in al_C_heat_energy.keys():
            if self.T > t:
                continue
            else:
                self.T = al_C_heat_energy[t]
                return 1

    def obnulay(self):
        self.T = 29
        self.heat_energy = al_C_heat_energy[27]

    def get_t(self):
        return self.T

    def get_phase(self):
        return min(1,self.T/660)

#NEAR EDGE начинает плавиться от полученной ЭНЕРГИИ (по теплоемкости)
#КРАЙ NEAR EDGE занимает FULL обьем 30мкм сферы - половину обьема сферы r=10

volume_30um = (4/3) * pi * particle_r**3
volume_10um = (4/3) * pi * (10*10**-6)**3
near_edge_volume = volume_30um-volume_10um
far_edge_volume = volume_30um - near_edge_volume

#тут почнется цикл перебираем каждый режим

near_edge = Fuield(near_edge_volume, 27) #цельсиев
far_edge = Fuield(far_edge_volume, 27)

#ЭНЕРГИЮ надо типо пересчитать по гаусу (какая часть энергии приходит в крайюю частицу)
#**pic 2**
"""
берем 3 ситуации (2 когда частица крайняя в углу) и 1 когда луч двигается -> больше энергии
потом плавно она под центром гауса и потом опять мало
**pic под 3**

достаточно сложить эти 3 ситуации и не париться с пересчетом в гаус по этой оси
по горизонтали надо посчитать 2/5 гауса сбоку 
"""
#как бы тут его пересчитать, я просто буду брать 2/5 мощности да и все!
reflex = 0.7
part_of_gause = 0.3
df["edge_power"] = df["power"]*part_of_gause*(1-reflex)

def set_time(value):
    speed = value
    return 3*particle_d/speed

def set_joule(value):
    return value["edge_power"]*value["time"]

#добавляем time в таблицу

df["time"] = df["speed"].apply(set_time)
# вариант РАБОЧИЙ
# df["edge_joule"] = df.apply(set_joule,axis=1)

s_light = 2*3.14*(30*10**-6)**2
df["edge_joule"] = (df["power_density"]*s_light) / 1000

df["near_t"],df["far_t"],df["near_phase"],df["far_phase"] = 29,29,0.0,0.0

new_df = pd.DataFrame()

for pair_n in range(len(df)):
    print(pair_n)
    data = df.iloc[pair_n]

    full_energy = data["edge_joule"]
    programm_delta = 10
    step = full_energy/programm_delta
    while full_energy > 0:
        #передаем в ближнйи край энергии 1/10 от дозы
        near_edge.increase_energy(step)
        near_edge.increase_t(data["time"]/programm_delta)
        #она греется + #пересчитываем передачу в дальнюю зону
    
        full_energy -= step
    
    #обнуляем на следующий режим
    data["near_t"] = near_edge.get_t()
    data["far_t"] = far_edge.get_t()

    near_edge.obnulay()
    far_edge.obnulay()

    data = pd.DataFrame(data)
    df = pd.concat([df, new_df], ignore_index=True)
    #new_df = new_df.append(data,ignore_index=True)


#df = new_df
#нагреваем частицы
#передаем часть температуры
#проверяем плавку

"""
**pic4** 
теплопроводность выражается в Вт/(м2*К).
если взять стену из кирпича, с коэффициентом теплопроводности 0,67 Вт/(м2*К),
толщиной 1 метр и площадью 1 м^2, то при разнице температур в 1 градус,
через стену будет проходить 0,67 ватта тепловой энергии.

ватт = 1 Джоуль за 1 секунду
количество секунд у нас есть (из скорости)

в нашем случае теплопроводность 237 Вт/(м·К) 
                                      

у нас без площади, просто на расстоянии 1м. 
надо пересчитать зеленую часть на расстоянии 10 мкм
это часть обозначим ДАЛЬНИЙ КРАЙ FAR EDGE 

таким образом мы можем оценить температуры и степень расплавленности частиц (я надеюсь)

"""

df["near_t"] = df["near_t"]/2
df["far_t"] = df["far_t"]/2

data["near_phase"] = near_edge.get_phase()
data["far_phase"] = far_edge.get_phase()

sns.pairplot(df[df["surface"]=="surface45"],hue="final_rough",
                    x_vars=["power"],
                    y_vars=["speed"])

sns.pairplot(df[df["surface"]=="surface45"],hue="final_porosity",
                    x_vars=["power"],
                    y_vars=["speed"])



sns.pairplot(df[df["speed"]<=2000], hue="surface",
                    x_vars=["power_density","power","speed"],
                    y_vars=["near_t","far_t","poor_surface","final_porosity","final_rough"])

plt.show()


