import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv("merc.csv") #transmission kolonu sayısal olmadığı için df değişkeninde gözükmüyor eğer istenirse encoderlarla sayısal değere dönüştürülüp kullanılabilir
df_Describe = df.describe()
df_NullValues = df.isnull().sum()

plt.figure(figsize=(7,5))
sbn.distplot(df["price"])

sbn.countplot(df["year"])

print(df.corr())#hangi kolonun hangi kolonu ne kadar etkilediğini gösterir
print("----------------------------------------------------------------------------------------------------")
print(df.corr()["price"].sort_values()) #Kolonların price kolonuna olan etkilerini gösterir negetifleri fiyatı düşürüyor pozitifler arttırıyor

sbn.scatterplot(x="mileage",y="price",data=df) #mil (km) arttıkça fiyat düşüyor, amaç veriyi incelemek


df_SortedPrices = df.sort_values("price",ascending=False).head(20) 

print(int(len(df) * 0.01)) # en pahalı arbalardan yüzde 1 ini df den atıyoruz çünkü yüksek fiyatlı araba sayısı çok az ve veri setini bozabilir ve %1 lik veri kaybı sıkıntı edilmez 

df_DropedValues = df.sort_values("price", ascending=False).iloc[131:] # büyükten küçüğe sıralatıp 131.terimden(yani %1 lik kısmını attık)başlayarak df yi güncelledik

df_DropedValuesDescribe = df_DropedValues.describe()

plt.figure(figsize=(7,5))    
sbn.distplot(df_DropedValues["price"])
print("--------------------------------------------------------------------------------------------------------")
print(df.groupby("year").mean()["price"])
print("----------------------------------------------")
print(df_DropedValues.groupby("year").mean()["price"])

print(df_DropedValues[df_DropedValues.year != 1970].groupby("year").mean()["price"])#çok uçuk özelliklere sahip olan yılı tespit ediliyor
df_new = df_DropedValues[df_DropedValues.year != 1970]#yukarıda yapılan değişiklilik yeni df olarak kullanılıyor
print(df_new.groupby("year").mean()["price"])

df_new = df_new.drop("transmission",axis=1)
df_new = df_new.drop("model",axis=1)
df_new = df_new.drop("fuelType",axis=1)

y = df_new["price"].values
x = df_new.drop("price",axis=1)



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)  


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))

#son durum değerlendirme
model.compile(optimizer="adam",loss="mse")
model.fit(x= x_train, y = y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)
kayipVerisi = pd.DataFrame(model.history.history)
print(kayipVerisi.head())
kayipVerisi.plot()
from sklearn.metrics import mean_squared_error, mean_absolute_error
tahminDizisi = model.predict(x_test)
print(mean_absolute_error(y_test,tahminDizisi))
plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")

#code test
yeniArabaSeries = df_new.iloc[2]
expectedPrice = yeniArabaSeries['price']
yeniArabaSeries = yeniArabaSeries.drop('price')
yeniArabaSeries = yeniArabaSeries.values.reshape(-1,5)
yeniArabaSeries = scaler.transform(yeniArabaSeries)
tahmin = model.predict(yeniArabaSeries)

print(tahmin)
print("----------------------------------------------------------------------------")
print(df.corr()["price"].sort_values())
print("----------------------------------------------------------------------------")
print(df.corr())







































