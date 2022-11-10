import pandas as pd
import numpy as np
import mlxtend
import openpyxl
import datetime
from mlxtend.frequent_patterns import apriori,association_rules

#################################################
# ayarlamalar
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.max_columns",None)
# İŞ PROBLEMİ
"""
Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.
Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.
"""
# VERİ SETİ HİKAYESİ
"""
Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
Alınan her hizmetin tarih ve saat bilgisini içermektedir.
UserId      : muşteri nuamrası

serviceId   : Her kategoriye ait anonimleştirilmiş servislerdir. 
            (Örnek : Temizlik kategorisi altında koltuk yıkama servisi) 
            Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler
            altında farklı servisleri ifade eder. 
            (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken
            CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
            
categoryId  : Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi) 

createdate  :Hizmetin satın alındığı tarih
"""
#################################################

#################################################
# GÖREV 1 ADIM 1
#todo: armut_data.csv dosyasını okutunuz.
#################################################
df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()
df.head()
df.info()
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.shape
df.nunique()
df.isnull().sum()
#################################################
# GÖREV 1 ADIM 2
#todo: ServisID her bir CategoryID özelinde farklı
# bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri
# temsil edecek yeni bir değişken oluşturunuz.
#################################################
df["stockcode"] = df.apply(lambda row: "%s_%s" % (row["ServiceId"],row["CategoryId"]),axis=1)
df.head()
df.nunique()
#################################################
# GÖREV 1 ADIM 3
#todo: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır,
# herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.)
# tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir
# müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4
# hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri
# başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile
# tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay
# içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz
# date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.
#################################################
df["year"] = df["CreateDate"].dt.year
df["month"] = df["CreateDate"].dt.month
df["transactions"] = df.apply(lambda row: "%s_%s_%s" % (row["UserId"],row["year"],row["month"]),axis=1)
#################################################
# GÖREV 2 ADIM 1
#todo: sepet, hizmet pivot table’i oluşturunuz.
#################################################
pivot_df = df.groupby(["transactions","stockcode"])["stockcode"].\
    count().\
    unstack().\
    fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0)

#################################################
# GÖREV 2 ADIM 2
#todo: Birliktelik kurallarını oluşturunuz.
#################################################
# apriori' ye uygun hale gelen verisetini kullanarak support degerlerini belirlemek
pivot_df_apriori = apriori(pivot_df,min_support=0.01,use_colnames=True)

rules = association_rules(pivot_df_apriori,metric="support",min_threshold=0.01).sort_values("lift",ascending=False)

#################################################

# GÖREV 2 ADIM 3
#todo: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
#################################################
def arl_recomender(df,product_id,rec_lim=1):
    recomendation_list = []
    product_id = product_id
    for index,antecedent in enumerate(rules["antecedents"]):
         for i in list(antecedent):
            if i == product_id:
                recomendation_list.append(list(rules.iloc[index]["consequents"])[0])
    return  recomendation_list[:rec_lim]

arl_recomender(rules,"15_1")



































