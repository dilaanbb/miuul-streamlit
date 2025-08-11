import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import joblib

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("winequality-red.csv")
df.head()

############################# FEATURE ENGINEERING #############################
# Binary target: quality > 6.5 -> good (1), else bad (0)
df["quality_binary"] = (df["quality"] > 6.5).astype(int)
df["quality_binary"].value_counts()

df['alcohol'].describe().T
df["pH"].describe().T

df['alcohol_level'] = pd.cut(df['alcohol'],  bins=[8.39, 9.5, 11.1, 14.91],
                             labels=['low', 'medium', 'high'],
                             include_lowest=True)
df['alcohol_level'].value_counts()
df.groupby('alcohol_level',observed=False)['quality'].mean()

df['pH_category'] = pd.cut(
    df['pH'],
    bins=[2.7, 3.21, 3.4, 4.1],
    labels=['acidic', 'optimal', 'basic'],
include_lowest=True
)#Orta veya yüksek asidite = iyi bir şarap

df['pH_category'].value_counts()
df.groupby('pH_category',observed=False)['quality'].mean()
df.head()

############################# EDA #############################



############################# KATEGORİK DEĞİŞKEN,SAYISAL DEĞİŞKEN ANALİZİ VE GÖRSELLEŞTİRME #############################

def grab_col_names(dataframe, cat_th=10, car_th=30):
    """
Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Parameters
    ----------
    dataframe : dataframe
        değişken isimleri alınmak istenen data frame' dir.
    cat_th: int,float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th : int,float
        kategorik fakat kardinal olan değişkenler için sınıf eşik değeri
    Returns
    -------
    cat_cols : list
        kategorik değişken listesi
    num_cols : list
        numerik değişken listesi
     cat_but_car: list
        kategorik görünümlü kardinal değişken listesi
     Notes
     -------
     cat_cols + num_cols + cat_but_car = toplam değişken sayısı
     num_but_cat cat_cols un içerisinde
     return olan 3 lsite toplamı toplam değişken sayısına eşittir: cat_cols+num_cols+cat_but_car

    """

    # Kategorik değişkenler: object, category veya bool tipindekiler
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["bool", "category", "object"]]

    # Sayısal tipte olup eşsiz değeri belirli bir eşikten az olanlar da kategorik sayılmalı
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

    # Kategorik gibi görünen ama çok fazla sınıfa sahip değişkenler kardinal sayılır
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    # Kategorik değişken listesine sayısal ama kategorik olanları da ekle
    cat_cols = cat_cols + num_but_cat

    # Kardinal değişkenleri kategorik listeden çıkar
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Geriye kalan sayısal değişkenler (kategorik olmayan sayısallar)
    num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    # Özet bilgileri yazdır
    print(f"Observations: {dataframe.shape[0]}")  # Satır sayısı
    print(f"Variables: {dataframe.shape[1]}")  # Sütun sayısı
    print(f"cat_cols: {len(cat_cols)}")  # Kategorik değişken sayısı
    print(f"num_cols: {len(num_cols)}")  # Sayısal değişken sayısı
    print(f"cat_but_car:{len(cat_but_car)}")  # Kardinal değişken sayısı
    print(f"num_but_cat:{len(num_but_cat)}")  # Sayısal olup kategorik sayılan değişken sayısı

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("######################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("#########")

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#############################  HEDEF DEĞİŞKEN ANALİZİ #############################
# Hedef Değişkenin Kategorik Değişkenler İle Analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col,observed=False)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "quality_binary", col)  # tüm kategorik değişkenler bağımlı değişken ile analize sokuldu.


# Hedef Değişkenin Sayısal Değişkenler İle Analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target,observed=False).agg({numerical_col: "mean"}), end="\n\n")

# Tüm sayısal değişkenler için quality değişkenine göre ortalama değerleri hesapla
for col in num_cols:
    target_summary_with_num(df, "quality_binary", col)

df.head()
######################### KORELASYON ANALİZİ ###########################
df.shape

# Sayısal (numeric) değişkenleri seçiyoruz
num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]

# Seçilen değişkenler arasında korelasyon matrisini oluşturuyoruz
corr = df[num_cols].corr()

# Isı haritası ile korelasyonları görselleştiriyoruz
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

# Tüm korelasyonları mutlak değere çevirerek işaretlerden bağımsız hale getiriyoruz
cor_matrix = df[num_cols].corr().abs()

# Üst üçgen matris: korelasyon matrisinin üst kısmını alıp alt tarafını sıfırlıyoruz
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# 0.90'dan büyük korelasyona sahip sütunları listeye alıyoruz
drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > 0.90)]

df.drop(columns=drop_list, inplace=True)
df.shape
df.info()
df.head()
#hiç %90 üzeri korelasyonlu sütun yok ve dolayısıyla silinecek sütun da yok.
############################ AYKIRI DEĞERLERİ YAKALAMA (OUTLIERS) ############################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# ALT VE ÜST LİMİT BELİRLENDİ
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# AYKIRI DEĞER VAR MI YOK MU ONA BAKILDI
def check_outlier(dataframe, col_name):
    if not pd.api.types.is_numeric_dtype(dataframe[col_name]):
        return False
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    return ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)).any()

# AYKIRI DEĞERLERİN KENDİLERİNE ERİŞİLDİ
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    print(
        f" Sütun: '{col_name}' | Toplam aykırı değer: {dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0]}")

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

# AYKIRI DEĞER PROBLEMİNİ ÇÖZME
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    # Orijinal değerleri sakla
    original = dataframe[variable].copy()

    # Aykırı değerleri sınırlar ile değiştir
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

    # Değişen değerleri filtrele
    changed = original != dataframe[variable]

    # Eski ve yeni değerleri yazdır
    changed_values = pd.DataFrame({
        "Index": dataframe[changed].index,
        "Eski Değer": original[changed],
        "Yeni Değer": dataframe[variable][changed]
    })

    print(changed_values.to_string(index=False))

    return changed_values



for col in num_cols:
    print(col, outlier_thresholds(df, col))

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    print(col, grab_outliers(df, col))

for col in num_cols:
    print(col, replace_with_thresholds(df, col))

for col in num_cols:
    print(col, check_outlier(df, col))

############################ MISSING VALUES ############################

#eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum() #Bir satırda en az 1 eksiklik olduğunda da bunu sayar.


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0] #Eksik değere sahip gözlem isimleri

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False) #Her bir değişkende yer alan eksik değer sayıları
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False) #Eksik değerlerin tüm eksik değerler içindeki oranları
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

#Eksik Değerin Bağımlı Değişken İle İlişkisinin İncelenmesi
missing_values_table(df, True) #Eksik değere sahip olan dğeişkenler
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0) #NA ya sahip olan değişkenlerin yanına NA FLAG eklenir


    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "quality_cat", na_cols)

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df) #ilgili veri setindeki değişkenlerdeki tam değerlerin sayısını gösterir.
plt.show(block=True)

#############################################
# Eksik Değer Problemini Çözme
#############################################
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "category" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

############################ ENCODING SCALING ############################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2 and col != 'quality_cat' and col != "quality"]
# one_hot_encoding sütunları

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
#One hot encoder sonucu oluşan değişkenlerin kaçı anlamlı,rare encoder'a bakalım.
df.head()

#Türettiğimiz değişkenlerle target arasında anlamlı bir ilişki var mı ona bakarız,rare_analyser ile.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "quality_binary", cat_cols)

def find_useless_binary_cols(dataframe, target, rare_thresh=0.05, effect_thresh=0.1):
    useless = []
    for col in dataframe.columns:
        if dataframe[col].nunique() == 2:
            # True oranı
            true_ratio = dataframe[col].mean()
            false_ratio = 1 - true_ratio

            # Rare kontrolü
            if true_ratio < rare_thresh or false_ratio < rare_thresh:
                useless.append(col)
                continue

            # Etki kontrolü: iki sınıfın target ortalaması birbirine çok yakınsa
            group_means = dataframe.groupby(col)[target].mean()
            if abs(group_means.diff().iloc[-1]) < effect_thresh:
                useless.append(col)

    return useless
useless_cols = find_useless_binary_cols(df, "quality_binary")
print("Gereksiz sütunlar:", useless_cols)
df.head()

drop_cols = [
    "alcohol_level_medium",
    "pH_category_optimal",
    "pH_category_basic",
]

df.drop(columns=drop_cols, axis=1,inplace=True)

df.shape
df.head()

############################# MACHINE LEARNING #############################
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# 1. Özellik ve hedef değişkenleri ayır
X = df.drop(columns=["quality", "quality_binary", "alcohol_level_high"])  # quality ve belirttiğin sütunlar hariç
y = df["quality_binary"]

# 2. Eğitim ve test verilerini ayır (stratify ile sınıf dengesini koru)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Sayısal sütunları seç ve ölçeklendir
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
scaler = StandardScaler()

# Sadece sayısal sütunları ölçeklendir
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

# 4. Hiperparametre aralıkları belirle

param_dist_rf = {
    "n_estimators": [50, 100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "bootstrap": [True, False]
}

param_dist_lr = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l2"],  # 'l1' için solver farklı gerekir
    "solver": ["lbfgs"],
    "max_iter": [1000]
}

param_dist_svm = {
    "C": np.logspace(-3, 3, 10),
    "kernel": ["rbf", "linear", "poly"],
    "gamma": ["scale", "auto"]
}

# 5. Model nesneleri oluştur
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)
svm = SVC(probability=True, random_state=42)

# 6. RandomizedSearchCV nesneleri oluştur
rs_rf = RandomizedSearchCV(
    rf, param_distributions=param_dist_rf, n_iter=50, cv=5, verbose=2,
    random_state=42, n_jobs=-1
)
rs_lr = RandomizedSearchCV(
    lr, param_distributions=param_dist_lr, n_iter=20, cv=5, verbose=2,
    random_state=42, n_jobs=-1
)
rs_svm = RandomizedSearchCV(
    svm, param_distributions=param_dist_svm, n_iter=20, cv=5, verbose=2,
    random_state=42, n_jobs=-1
)

# 7. Hiperparametre optimizasyonlarını sırayla yap
print("RandomForest hiperparametre optimizasyonu başlıyor...")
rs_rf.fit(X_train, y_train)
print("En iyi RF parametreleri:", rs_rf.best_params_)

print("LogisticRegression hiperparametre optimizasyonu başlıyor...")
rs_lr.fit(X_train, y_train)
print("En iyi LR parametreleri:", rs_lr.best_params_)

print("SVM hiperparametre optimizasyonu başlıyor...")
rs_svm.fit(X_train, y_train)
print("En iyi SVM parametreleri:", rs_svm.best_params_)

# 8. En iyi modelleri al
best_rf = rs_rf.best_estimator_
best_lr = rs_lr.best_estimator_
best_svm = rs_svm.best_estimator_

# 9. Modelleri bir sözlükte topla
models = {
    "RandomForest_Optimized": best_rf,
    "LogisticRegression_Optimized": best_lr,
    "SVM_Optimized": best_svm
}

# 10. Modelleri değerlendir
for name, model in models.items():
    print(f"\nModel: {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # predict_proba metodunun varlığını kontrol et
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # predict_proba yoksa decision_function kullan, veya y_prob'u None yap
        try:
            y_prob = model.decision_function(X_test)
            # decision_function çıktılarını olasılığa dönüştürmek için sigmoid/logistic uygulanabilir
            # ama basitçe AUC hesaplamak için y_prob kullanılabilir
        except:
            y_prob = None

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if y_prob is not None:
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    else:
        print("ROC AUC Score: Hesaplanamadı (predict_proba veya decision_function yok)")

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=True)

    if y_prob is not None:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{name} ROC Curve")
        plt.show(block=True)

        PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{name} Precision-Recall Curve")
        plt.show(block=True)


# 11. En iyi modeli ve scaler'ı kaydet
# Burada örnek olarak RandomForest modeli seçildi, istersen skorları karşılaştırıp değiştirebilirsin
joblib.dump(best_rf, "best_wine_quality_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("En iyi model ve scaler kaydedildi.")

# 12. Overfitting kontrolü için Learning Curve çizimi
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Training accuracy')
plt.plot(train_sizes, test_mean, label='Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Training Set Size')
plt.title('Learning Curve')
plt.legend()
plt.show(block=True)
