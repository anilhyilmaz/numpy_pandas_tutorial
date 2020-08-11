import numpy as np
import pandas as pd
import seaborn as sns
"""a = np.array([2,4,6,8])
b = np.array([1,3,5,7],)
print(a*b)
numpy_zeros = np.zeros(10,dtype=int)
numpy_ones = np.ones((3,5),dtype=int)
print(numpy_zeros)
print(numpy_ones)
fours_np = np.full((3,5),4)
print(fours_np)
print(np.random.normal(10,4,(3,4)))
my_numpy = np.random.randint(5,10,size=4)
print(my_numpy)
print(my_numpy.shape)
print(my_numpy.ndim)
print(np.arange(1,10).reshape(3,3))
x = np.array([[2,3,4],[5,6,7]])
y = np.array([4,6,8])
#print(np.concatenate([x,y]))
#print(np.concatenate([x,x],axis=1))
a = np.array([1,2,3,99,99,3,2,1])
r,t,y = np.split(a,[3,5])
print(r,t,y)
m = np.arange(16).reshape(4,4)
print(np.vsplit(m,[2]))
print(np.hsplit(m,[2]))
m = np.random.normal(20,5,(3,4))
print(m)
print(np.sort(m,axis=1))
m = np.random.randint(10,size=(3,5))
print(m)
print(m[:,4])
a = np.random.randint(10,size=(3,4))
print(a)
alt_a = a[0:3,0:2]
print(alt_a)
alt_a[0,0] = 9999
alt_a[1,1] = 888
print(alt_a)
print(a)
m = np.random.randint(10,size=(3,4))
alt_b = m[0:3,0:2].copy()
v = np.array([1,2,3,4,5,6,7,8,9])
print(v>3)
print(v[v>5]) #v[v>3] fancy
#5*x0 + x1 = 12
#x0 + 3*x1 = 10
a = np.array([[5,1],[1,3]])
b = np.array([12,10])
print(a,b)
print(np.linalg.solve(a,b))
my_series = pd.Series([10,20,30,40,50])
#print(my_series.axes)
print(my_series.values)
print(my_series.head(3))
print(my_series[1:3])
my_series_1 = pd.Series([10,20,30,40,50],index=[1,3,5,7,9])
print(my_series_1)
sozluk = pd.Series({"reg":10,"log":11,"cart":12})
print(sozluk)
print(pd.concat([sozluk,sozluk]))
a = np.array([1,2,33,444,555])
seri = pd.Series(a)
print(seri)
b = pd.Series([1,2,3,4,5,6,7],index=["reg","log","cart","rf","aa","bb","cc"])
print(b)
print("reg" in b)
print(b["reg":"log"])
l = [1,2,56,75,32]
#pandas dataframe oluşturmak
print(pd.DataFrame(l,columns=["degisken ismi"]))
m = np.arange(1,10).reshape((3,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df.head())
print(df.columns)
#dataframe içinde sozluk ve np.random.randint
s1 = np.random.randint(10,size=5)
s2 = np.random.randint(10,size=5)
s3 = np.random.randint(10,size=5)
sozluk = {"var1":s1,"var2":s2,"var3":s3}
print(sozluk)
df = pd.DataFrame(sozluk)
print(df)
#print(df[0:3])
#silme işlemleri
print(df.drop(1,axis=0)) #drop işlemleri df üzerinde tam anlamıyla tümünde degişiklik yapmamaktadır bunun için yeni bir dataframe içine copy edilmelidir.
#df orjinalinde degişiklik isteniyorsa inframe parametresi true yapılmalıdır!
print(df.drop(0,axis=0,inplace=True))
print(df)
df["var4"] = df["var1"] / df["var2"]
print(df)
#Gözlem ve degişken seçimi : &loc , &iloc
m = np.random.randint(1,30,size=(10,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df)
#&loc : tanımlandıgı şekli ile seçim yapmak için kullanılır. index olarak 0,1,2,3 alır.
print(df.loc[0:3])
#&iloc : alışık oldugumuz indexleme mantıgıyla seçim yapar. 0,1,2yi alır index olarak.
print(df.iloc[0:3])
#Koşullu eleman işlemleri
m = np.random.randint(1,30,size=(10,3))
df = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df)
#print(df[0:2][["var1","var3"]])
print(df[df.var1 > 15]["var1"])
print(df[(df.var1 > 15) & (df.var3 < 10)]) 
#birleştirme(join) işlemleri
m = np.random.randint(1,30,size=(5,3))
df1 = pd.DataFrame(m,columns=["var1","var2","var3"])
print(df1)
df2 = df1 + 99
print(df2)
print(pd.concat([df1, df2],ignore_index=False))
print(pd.concat([df1, df2],ignore_index=True))
#ileri seviye birleştirme işlemleri(birebir birleştirme)
df1 = pd.DataFrame({"çalisanlar":["ali","veli","ayse","fatma"],
                    "grup":["muhasebe","muhendislik","muhendislik","ik"]})
df2 = pd.DataFrame({"çalisanlar":["ayse","ali","veli","fatma"],
                    "ilk_giris":[2010,2009,2014,2019]})
print(df1)
print(df2)
print(pd.merge(df1,df2)) #merge ile iki listede çalışanlar grubu ortak oldugu için ona göre birleştirdi.
print(pd.merge(df1,df2,on="çalisanlar")) #on etiketi ile neye göre birleştirecegini seçmiş olduk.
df3 = pd.merge(df1,df2)
print("DF3:",df3)
df4 = pd.DataFrame({"grup":["muhasebe","muhendislik","ik"],
                    "mudur":["Caner","Mustafa","Berkcan"]})
print(df4)
print(pd.merge(df3,df4)) #df3 4 degerden df4 3 degerden oluşmasına ragmen grup nesnesine göre sıralamıştır.
#çoktan çoka sıralama
df5 = pd.DataFrame({"grup":["muhasebe","muhasebe","muhendislik","muhendislik","ik","ik"],
                    "yetenekler":["matematik","excel","kodlama","linux","excel","yonetim"]})
print(pd.merge(df1,df5)) #birleştirme işlemleri ile birden fazla yetenegi olan kişiler çoklamıştır.
##Toplulaştırma ve Gruplama(Aggregation and Grouping)
df = sns.load_dataset("planets") #sns kütüphanesinden planets eklenmiştir
print(df.head())
print(df.shape)
print(df.mean())
print(df["mass"].mean())
print(df["mass"].count())
print(df.describe()) #yukarıda tek tek baktıgımız bilgilere tek satırda bakmamıza yarıyor!
print(df.describe().T) #tersini alır transpoz
print(df.dropna().describe())
#Gruplama işlemleri
df = pd.DataFrame({"gruplar":["A","B","C","A","B","C"],
                   "veri":[10,11,52,23,43,55]},columns=["gruplar","veri"])
print(df)
print(df.groupby("gruplar").mean()) #groupby ile gruplar nesnesini yakalar, mean ile ayrı ayrı a,b,c gruplarının ortalamasını alır.
#mesela iki tane A grubu var degerleri 10 ve 23 toplayıp ikiye böldü.

df_planets = sns.load_dataset("planets")
print(df_planets)
print(df_planets.groupby("method")["orbital_period"].mean())  #method grubuna gidip orbital_period degişkinine göre ortalamasını bulmuştur.
print(df_planets.groupby("method")["orbital_period"].describe())
#İleri toplulaştırma işlemleri (Aggregate,filter,transfrom,apply)
df = pd.DataFrame({"gruplar":["A","B","C","A","B","C"],
                    "degisken1":[10,23,33,22,11,99],
                     "degisken2":[100,253,333,262,111,969]},
                      columns=["gruplar","degisken1","degisken2"])
print(df)
#aggregate
print(df.groupby("gruplar").aggregate(["min",np.median,"max"])) #grupları hedef alarak min np.median ve max ı kendisi ayrı olarak hesaplar.
#örnek olarak A grubu için min median ve max hesaplar.
print(df.groupby("gruplar").aggregate({"degisken1":"min","degisken2":"max"})) #gruplar nesnesini ele alarak degisken1 için min degisken2 için
# max hesapla.
#####Filtreleme
df = pd.DataFrame({"gruplar":["A","B","C","A","B","C"],
                    "degisken1":[10,23,33,22,11,99],
                     "degisken2":[100,253,333,262,111,969]},
                      columns=["gruplar","degisken1","degisken2"])
def filter_func(x):
    return x["degisken1"].std() > 9 #std ---> standart sapma
print(df)
print(df.groupby("gruplar").std())
print(df.groupby("gruplar").filter(filter_func))
#Transform
df = pd.DataFrame({"gruplar":["A","B","C","A","B","C"],
                    "degisken1":[10,23,33,22,11,99],
                     "degisken2":[100,253,333,262,111,969]},
                      columns=["gruplar","degisken1","degisken2"])
print(df)
df_a = df.iloc[:,1:3] #DataFrame içinde bulunan A,B,C degişkenlerine matematiksel işlem yapılamadıgı için iloc kullanarak bu hatadan kurtarıldı.
#iloc ile tüm satırları aldı ve 1 ve 3 e kadar sütun seçti A,B,C seçilmemiştir böylece.
print(df_a)
print(df_a.transform(lambda x: x-x.mean())) #lamba ayrı bir fonksiyon oluşturmadan parantez içinde fonksiyon oluşturmak için kullanılır.
#transform fonk belirli bir dönüştürme fonksiyonları için nesneler için kullanılır.
#Apply kullanımı
df = pd.DataFrame({"degisken1":[10,23,33,22,11,99],
                     "degisken2":[100,253,333,262,111,969]},
                      columns=["degisken1","degisken2"])
print(df.apply(np.sum))
#pivot tablolar
titanic = sns.load_dataset("titanic")
print(titanic.head())
print(titanic.groupby(["sex","class"])[["survived"]].aggregate("mean").unstack())
#pivot ile oluşturulması
print(titanic.pivot_table("survived",columns="class", index="sex"))
age = pd.cut(titanic["age"],[0,18,90]).head()
print(pd.cut(titanic["age"],[0,18,90]).head())
#titanic.pivot_table("survived",["sex",age],"class")"""
#txt excel dosya okuma
#print(pd.read_csv("ornekcsv.csv",sep=";"))
#print(pd.read_csv("duz_metin.txt"))
#print(pd.read_excel("ornekx.xlsx"))









