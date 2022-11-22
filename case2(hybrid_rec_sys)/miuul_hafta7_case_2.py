######################
# iş problemi
"""
ID'si verilen kullanıcı için item-based ve user-based recommender
yöntemlerini kullanarak 10 film önerisi yapınız.
"""
# veri seti hikayesi
"""
Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. 
İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
27.278 filmde 2.000.0263 derecelendirme içermektedir. 
Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir.
kullanıcılar rastgele seçilmiştir.
Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.
"""

# veri setleri
"""
# movie.csv : movieId | title | genres

# rating.csv: userid | movieId | rating | timestamp
"""
######################
import sys
import os
import math
import pandas as pd
import numpy as np
pd.set_option("display.expand_frame_repr",False)
pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################
############################################# USER BASED RECOMENDATION #################################################
########################################################################################################################

######################  Veri Hazırlama  ###################
# GÖREV 1 ADIM 1:
#TODO: movie, rating veri setlerini okutunuz.
######################
movie_ = pd.read_csv("datasets/movie_lens_dataset/movie.csv")
movie = movie_.copy()

rating_ = pd.read_csv("datasets/movie_lens_dataset/rating.csv")
rating = rating_.copy()

movie.shape
movie.isnull().sum()
movie.info()

rating.shape
rating.isnull().sum()
rating.info()

sys.getsizeof(movie[["movieId"]])
sys.getsizeof(movie["movieId"])
sys.getsizeof(movie["movieId"].astype(np.float32))
sys.getsizeof(movie)
sys.getsizeof(rating)

# --------- boyutu yarıya indirmek ------------
# --------- rating
# zaman bilgisi bize lazım olamayacak
rating = rating[["userId","movieId","rating"]]
rating_matrix = rating.astype(np.float32)
sys.getsizeof(rating_matrix)
sys.getsizeof(rating)
# --------- movie
movie.columns
movie.head()
# title ları ayrı bir yere aktar , genres zaten lazım olmayacak
movie_title = movie[["title","movieId"]]
movie = movie[["movieId"]]
movie_matrix = movie.astype(np.float32)
sys.getsizeof(movie_matrix)

movie_matrix.head()
rating_matrix.head()
######################
# GÖREV 1 ADIM 2:
#TODO: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
######################
#df = pd.merge(rating,movie[["movieId"]],on="movieId")
df = movie.merge(rating, how="left", on="movieId")
df.nunique()
######################
# GÖREV 1 ADIM 3:
#TODO: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
######################

comment_count = rating_matrix.groupby("movieId").agg({"movieId":"count"})
comment_count.rename(columns={"movieId":"count"},inplace=True)
comment_count = comment_count.reset_index()

rare_movies = comment_count[comment_count["count"]< 1000]["movieId"]

common_movies = rating_matrix[~rating_matrix["movieId"].isin(rare_movies)]

rating_matrix.nunique()
len(rare_movies)
common_movies.nunique()

######################
# GÖREV 1 ADIM 4:
#TODO: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz
######################
type(common_movies)
sys.getsizeof(common_movies)

common_movies_matrix = common_movies.astype(np.float32)

sys.getsizeof(common_movies_matrix)

user_movie_df = common_movies_matrix.pivot_table(index="userId" , columns="movieId", values="rating")

######################
# GÖREV 1 ADIM 5:
#TODO: Yapılan tüm işlemleri fonksiyonlaştırınız.
######################
def create_user_movie_df(rating_df, movie_df, comment_threshold=1000):
    import pandas as pd
    # userId | movieid | rating | genres | title  bilgilerini içeren df oluşturuldu
    df = pd.merge(rating_df, movie_df[["movieId", "genres", "title"]], on="movieId")
    # her bir film için max ve min yorum sayısı
    print(df.groupby("title").agg({"movieId": "count"}).values.max())
    print(df.groupby("title").agg({"movieId": "count"}).values.min())
    # her bir filme kaç yorum yapıldıgını bulduk
    comment_count = df.groupby("title").agg({"movieId": "count"}).reset_index()
    # sutun isismleri düzenlendi
    comment_count.columns = ["title", "num_of_comment"]
    # filmler belirlenen eşik degerden daha az yorum aldıysa  nadir filmler olarak atandı
    rare_movies = comment_count[comment_count["num_of_comment"] < comment_threshold].index
    # nadir filmler listesinde adı olmayan filmler yaygın filmlere atandı
    common_movies = df[~df["title"].isin(rare_movies)]
    # satırlarda kullanıcılar sutunlarda filmleri deger olarak ratingleri  içeren user_movie df oluşturuldu
    user_movie = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie


######################  Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi #####################

# GÖREV 2 ADIM 1:
#TODO: Rastgele bir kullanıcı id’si seçiniz..
######################
random_user = user_movie_df.sample(1).index.values[0]


######################
# GÖREV 2 ADIM 2:
#TODO: Seçilen kullanıcıya ait gözlem birimleriminden oluşan random_user_df adında yeni bir dataframe oluşturunuZ.
######################
type(user_movie_df)
user_movie_df.index
# userid ler indexlerde yer alıyor ve indexin adı userId
random_user_df = user_movie_df[user_movie_df.index == random_user]
######################
# GÖREV 2 ADIM 3:
#TODO: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız
######################
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_not_watched = random_user_df.columns[random_user_df.isna().any()].tolist()
len(movies_watched)
######################  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi ################
# GÖREV 3 ADIM 1:
#TODO: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturunuz.
######################
# satırlarda tüm kullanıcılar sutunlarda ise sadece secilmiş kullanıcıın izledigi tüm filmler
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape
######################
# GÖREV 3 ADIM 2:
#TODO: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini
# taşıyan user_movie_count adında yeni bir dataframe oluşturunuz
######################
user_movie_count = movies_watched_df.notnull().sum(axis=0)# her bir filmi izleyen kişi sayısı
user_movie_count = movies_watched_df.notnull().sum(axis=1)# her bir kişinin izlediği film sayısı
user_movie_count = movies_watched_df.T.notnull().sum()# her bir kişinin izlediği film sayısı

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]
######################
# GÖREV 3 ADIM 3:
#TODO: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı
# id’lerinden users_same_movies adında bir liste oluşturunuz
######################
def perc_to_num(perc):
    return (perc*len(movies_watched))/100

perc_to_num(60)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc_to_num(60)]

######################  Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi  ########################
# GÖREV 4 ADIM 1:
#TODO: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların
# id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
######################
"""
movies_watched_df de secili userımın izlediği tüm filmler ve tüüm kullanıcılar var.
şimdi ise sadece en az 123 tanesini izlemiş olan kullanıcıları bırakıcam movies_watched_df de
zaten elimizde en az 123 tanesini izleyenlerin listesi var 
(users_same_movies) burada en az 123 ortak film izleyenlerin userıd si var
"""
len(users_same_movies)
users_same_movies["userId"].values
movies_watched_df = movies_watched_df.iloc[movies_watched_df.index.isin(users_same_movies["userId"].values), :]
######################
# GÖREV 4 ADIM 2:
#TODO: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
######################
corr_df = movies_watched_df.T.corr()
######################
# GÖREV 4 ADIM 3:
#TODO:  Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek
# top_users adında yeni bir dataframe oluşturunuz.
######################
# corr_df için upper triangular matrix oluşturmak
# utm = np.triu(corr_df, k=1)[:5,:5]

top_users = corr_df.unstack().sort_values(ascending=False).drop_duplicates()
type(top_users)

top_users_df = pd.DataFrame(top_users, columns=["corr_val"])
top_users_df.index.names = ["user_1","user_2"]
top_users_df.reset_index(inplace=True)

random_user

top_users_df = top_users_df.loc[(top_users_df["user_1"] == random_user) & (top_users_df["corr_val"]>.45),["user_2","corr_val"] ]
######################
# GÖREV 4 ADIM 4:
#TODO: top_users dataframe’ine rating veri seti ile merge ediniz.
######################
rating_matrix
top_users_df
# top_user_df deki user_2 yi userId olarak degiştir ki rating verisi ile inner merge edebil.
top_users_df.rename(columns={"user_2":"userId"}, inplace =True)
top_users_rating = top_users_df.merge(rating_matrix[["userId", "movieId", "rating"]], how='inner')

######################  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması  ###############
# GÖREV 5 ADIM 1:
#TODO: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan
# weighted_rating adında yeni bir değişken oluşturunuz.
######################
top_users_rating.head()
top_users_rating["weighted_rating"] = top_users_rating["corr_val"] *top_users_rating["rating"]
######################
# GÖREV 5 ADIM 2:
#TODO: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin
# ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz
######################
recommendation_df = top_users_rating.groupby(["movieId","userId"]).agg({"weighted_rating":"mean"}).reset_index()
recommendation_df.weighted_rating.max()
######################
# GÖREV 5 ADIM 3:
#TODO:  recommendation_df içerisinde weighted rating'i 2.5'ten büyük olan filmleri
# seçiniz ve weighted rating’e göre sıralayınız.
######################
recommendation_df.head()
recommendation_df = recommendation_df[recommendation_df["weighted_rating"]>2.5].sort_values(by="weighted_rating",ascending=False)

######################
# GÖREV 5 ADIM 4:
#TODO:  movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.
######################
def get_movie_name(recommendation_df,num_of_rec):
    mov_id_list = recommendation_df.movieId.head(6).values
    return movie_title[movie_title["movieId"].isin(mov_id_list)]["title"]


get_movie_name(recommendation_df,6)


########################################################################################################################
############################################# ITEM BASED RECOMENDATION #################################################
########################################################################################################################

#############  Kullanıcının izlediği en son ve en yüksek puan verdiği filme göre item-based öneri yapınız.  ############
# GÖREV 1 ADIM 1:
#TODO: movie, rating veri setlerini okutunuz.
######################
movie_ = pd.read_csv("datasets/movie.csv")
movie = movie_.copy()

rating_ = pd.read_csv("datasets/rating.csv")
rating = rating_.copy()

sys.getsizeof(movie)
sys.getsizeof(rating)

rating["timestamp"] = pd.to_datetime(rating["timestamp"])
rating.info()

######################
# GÖREV 1 ADIM 2:
#TODO: Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
######################
random_user

rating.head()

random_filtered_df = rating[(rating["userId"]==random_user) & (rating["rating"]==5)]

random_filtered_df[random_filtered_df["timestamp"]==random_filtered_df["timestamp"].max()]["movieId"].values

best_latest_movie = random_filtered_df[random_filtered_df["timestamp"]==random_filtered_df["timestamp"].max()]["movieId"].values


######################
# GÖREV 1 ADIM 3:
#TODO: User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
######################
user_movie_df

user_movie_random_filtered_df = user_movie_df[best_latest_movie]
type(user_movie_random_filtered_df)
user_movie_random_filtered_df.columns
user_movie_random_filtered_df.index
######################
# GÖREV 1 ADIM 4:
#TODO: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız
######################
type(user_movie_df)
type(user_movie_random_filtered_df)

user_movie_random_filtered_df.columns
user_movie_df.columns

user_movie_random_filtered_df.reset_index(inplace=True)
user_movie_random_filtered_df.columns= ["userid","film"]

# 953-205 olan veri setinde puan dagılımı korelasyonu arıyorumm bu yüzden user_movie_random_filtered içinden
# sadece 953 kişiyi filtrelemem gerek bu sayede hybrid sistemi oluşturabiliyim

user_movie_random_double_filtered_df = user_movie_random_filtered_df.\
                                           loc[user_movie_random_filtered_df["userid"].isin(users_same_movies["userId"].values),:]
# 953-205 lik ile 953-1 lik film ve userid bilgilerini içeren versi setleri kullanıyorum
corr_df = movies_watched_df.corrwith(user_movie_random_double_filtered_df["film"]).sort_values(ascending=False)
corr_df.shape
# nan degerli filmleri ele
corr_df = corr_df[corr_df.notnull()]
corr_df.shape
######################
# GÖREV 1 ADIM 5:
#TODO: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
######################
corr_df.index
def get_movie_name(corr_df,num_of_rec):
    mov_id_list = corr_df.index[:num_of_rec].values
    return movie_title[movie_title["movieId"].isin(mov_id_list)]["title"]

get_movie_name(corr_df,6)
