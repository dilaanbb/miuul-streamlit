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
try:
    df = pd.read_csv("datasets/winequality-red.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("datasets/winequality-red.csv", encoding="latin1")
############################# FEATURE ENGINEERING #############################
df['quality_cat'] = pd.cut(df['quality'],
                           bins=[2, 4, 6, 9],
                           labels=['low_quality', 'medium_quality', 'high_quality'])
df["quality_cat"].value_counts()
df["quality"].describe().T


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

df["quality_cat_num"] = df["quality_cat"].map({
    "low_quality": 0,
    "medium_quality": 1,
    "high_quality": 2
}).astype("int")

# Hedef Değişkenin Kategorik Değişkenler İle Analizi
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col,observed=False)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "quality_cat_num", col)  # tüm kategorik değişkenler bağımlı değişken ile analize sokuldu.


# Hedef Değişkenin Sayısal Değişkenler İle Analizi
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target,observed=False).agg({numerical_col: "mean"}), end="\n\n")

# Tüm sayısal değişkenler için quality değişkenine göre ortalama değerleri hesapla
for col in num_cols:
    target_summary_with_num(df, "quality_cat_num", col)

df.drop("quality_cat_num", axis=1, inplace=True)
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

#Silmeden önce bir bakalım:
df["quality_cat_num"] = df["quality_cat"].map({
    "low_quality": 0,
    "medium_quality": 1,
    "high_quality": 2
}).astype("int")

df[drop_list].corrwith(df["quality_cat_num"]).sort_values(ascending=False)
df.drop("quality_cat_num", axis=1, inplace=True)

df.shape
df.info()
df.head()

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
import pandas as pd
#col != 'quality' and col != "quality_score"

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

df["quality_cat_num"] = df["quality_cat"].map({
    "low_quality": 0,
    "medium_quality": 1,
    "high_quality": 2
}).astype("int")

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "quality_cat_num", cat_cols)

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
useless_cols = find_useless_binary_cols(df, "quality_cat_num")
print("Gereksiz sütunlar:", useless_cols)
df.head()

drop_cols = [
    "alcohol_level_medium",
    "pH_category_optimal",
    "pH_category_basic",
    "quality_cat_num"
]

df.drop(columns=drop_cols, axis=1,inplace=True)

df.shape
df.head()

############################# MACHINE LEARNING #############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import StackingClassifier

import joblib
import os

# --- 1. Feature Engineering ---
df["acidity_ratio"] = df["fixed_acidity"] / (df["volatile_acidity"] + 1e-5)
df["density_alcohol"] = df["density"] * df["alcohol"]

# --- 2. Veri ve hedef ---
X = df.drop(["quality", "quality_cat", "alcohol_level_high"], axis=1)
y = df["quality_cat"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- 3. class_weight ---
class_weights = {0: 20, 1: 1, 2: 1}
print("Class weights:", class_weights)

# --- 4. Stratified train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- 5. Ölçeklendirme ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Oversampling (SMOTETomek) ---
smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train_scaled, y_train)

print("SMOTETomek sonrası eğitim seti sınıf dağılımı:")
print(pd.Series(y_train_res).value_counts())

# --- 7. ROC için binarize ---
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))

# --- 8. Random Forest Hiperparametre Optimizasyonu ---
rf = RandomForestClassifier(class_weight=class_weights, random_state=42)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=rf_params, n_iter=5,
    cv=3, verbose=2, random_state=42, n_jobs=-1,
    scoring='f1_macro'
)
rf_random.fit(X_train_res, y_train_res)
rf_best = rf_random.best_estimator_
print("Best RF Params:", rf_random.best_params_)
print("Best RF Score:", rf_random.best_score_)

# --- 9. ROC Eğrisi ile Threshold Optimizasyonu (low_quality için) ---
y_prob_rf = rf_best.predict_proba(X_test_scaled)
fpr, tpr, thresholds = roc_curve(y_test_bin[:, 0], y_prob_rf[:, 0])
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold for low_quality: {optimal_threshold:.2f}")

y_pred_rf_thresh = []
for probas in y_prob_rf:
    if probas[0] >= optimal_threshold:
        y_pred_rf_thresh.append(0)
    else:
        y_pred_rf_thresh.append(np.argmax(probas[1:]) + 1)

print("\nRandom Forest Threshold Adjusted Classification Report:")
print(classification_report(y_test, y_pred_rf_thresh, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_thresh))

# --- 10. ROC AUC Skoru ---
rf_auc_thresh = roc_auc_score(y_test_bin, y_prob_rf, average="macro", multi_class="ovr")
print(f"\nRandom Forest (Threshold) ROC AUC: {rf_auc_thresh:.4f}")

# --- 11. Model Kaydet ---
if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(rf_best, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# --- 12. Feature Importances ---
if hasattr(rf_best, 'feature_importances_'):
    importances = rf_best.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title("Feature Importances")
    plt.show()

# --- 13. SHAP Analizi (low_quality sınıfı için) ---
try:
    explainer = shap.TreeExplainer(rf_best)
    shap_values = explainer.shap_values(X_train_res[:100])
    shap.summary_plot(shap_values[0], X_train_res[:100], feature_names=X.columns)
except Exception as e:
    print("SHAP çalıştırılırken hata oluştu:", e)
