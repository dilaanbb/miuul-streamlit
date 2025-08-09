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
df = pd.read_csv("datasets/winequality-red.csv")
df.head()

############################# FEATURE ENGINEERING #############################
df['quality_cat'] = pd.cut(df['quality'],
                           bins=[2, 4, 6, 9],
                           labels=['low_quality', 'medium_quality', 'high_quality'])

df["quality"].describe().T

df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid'] #Toplam asitlik
df.groupby('quality_cat',observed=False)['total_acidity'].mean()


df['acid_balance'] = df["total_acidity"] / df["pH"] #Asidite dengesi,asidite-ph dengesi,acid_balance yüksekse iyi kalite
df.groupby('quality_cat',observed=False)['acid_balance'].mean()

df["aroma_risk"] = df["volatile acidity"] + df["chlorides"] + df["total sulfur dioxide"]/100 #Aroma üzerinde olumsuz etki yaratacak değişkenler
#Yüksekse kalite olumsuz etkilenir.
df.groupby('quality_cat',observed=False)['aroma_risk'].mean()


df["sulfate_boost"] = df["sulphates"] * df["alcohol"] #Sülfat ve alkol birlikte kaliteyi artırabilir.
df.groupby('quality_cat',observed=False)['sulfate_boost'].mean()


df["quality_score"] = (df["alcohol"] + df["citric acid"] +df["sulphates"]) - (df["volatile acidity"] + df["chlorides"] + df["pH"])
df.groupby('quality_cat',observed=False)['quality_score'].mean()


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

def check_def(dataframe, head=5):
    # DataFrame'in satır ve sütun sayısını gösterir
    print("####Shape####")
    print(dataframe.shape)

    # Sütunlardaki veri tiplerini gösterir (int, float, object vb.)
    print("####Types####")
    print(dataframe.dtypes)

    # İlk 'head' kadar satırı gösterir
    print("####Head####")
    print(dataframe.head(head))

    # Son 'head' kadar satırı gösterir
    print("####Tail####")
    print(dataframe.tail(head))

    # Her sütunda kaç adet eksik (NA/null) değer olduğunu gösterir
    print("####NA####")
    print(dataframe.isnull().sum())

    # Sayısal değişkenler için çeşitli yüzdelik dilimlere göre özet istatistikler verir
    print("####Quantiles####")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_def(df)

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

############################# 1- HEDEF DEĞİŞKEN ANALİZİ #############################

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


drop_list = ['total_acidity', 'acid_balance'] #İki değişkenin etkisi zayıf bu yüzden bunları sildik.
df.drop(drop_list, axis=1, inplace=True)
df.shape
df.info()

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
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df.head()
#İki sınıflı kategorik değişken olmadığı için label encoder yapamıyoruz.

# İki sınıftan fazla olan kategorik değişkenleri one_hot_encoder' dan geçirme.Alfabetik sıraya göre ilk sınıf silinir,
# Geriye kalan sınıflar ayrı bir değişken olarak veri setinde yer alır.

df.head()

import pandas as pd
#col != 'quality' and col != "quality_score"

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2 and col != 'quality_cat' and col != "quality"]
# one_hot_encoding sütunları

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
#One hot encoder sonucu oluşan değişkenlerin kaçı anlamlı,rare encoder a bakarım bunun için.
df.head()

#Bu türettiğimiz değişkenlerle target arasında anlamlı bir ilişki var mı ona bakarız,rare_analyser ile.
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

drop_cols = [
    "alcohol_level_medium",
    "pH_category_optimal",
    "pH_category_basic",
    "quality_cat_num"
]

df.drop(columns=drop_cols, axis=1,inplace=True)

df.shape
df.head()
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import shap
import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked folder objects to clean up at shutdown")
warnings.filterwarnings("ignore", message="resource_tracker: .*FileNotFoundError.*")



# --- 1. Veri Yükleme ve Hedef Değişken Hazırlığı ---
df = pd.read_csv("datasets/winequality-red.csv")  # Veri seti
mapping = {
    3: "low_quality",
    4: "low_quality",
    5: "medium_quality",
    6: "medium_quality",
    7: "high_quality",
    8: "high_quality"
}
df["quality_cat"] = df["quality"].map(mapping)

X = df.drop(["quality", "quality_cat"], axis=1)
y = df["quality_cat"]


le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --- class_weight sözlüğü ---
# LabelEncoder sınıflarına göre; örn: 0=high_quality, 1=low_quality, 2=medium_quality
class_weights = {0: 1, 1: 5, 2: 1}  # low_quality için 5 kat ağırlık

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- 2. Ölçeklendirme ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Oversampling: SMOTE + BorderlineSMOTE ---
smote = SMOTE(random_state=42)
border_smote = BorderlineSMOTE(random_state=42, kind='borderline-1')

X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)
X_train_res, y_train_res = border_smote.fit_resample(X_smote, y_smote)

print("SMOTE + BorderlineSMOTE sonrası eğitim seti sınıf dağılımı:")
print(pd.Series(y_train_res).value_counts())

# --- 4. ROC için binarize ---
y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))

# --- 5. Modeller ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=class_weights, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight=class_weights, random_state=42),
    "SVM": SVC(probability=True, class_weight=class_weights, random_state=42),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
    "Balanced RF": BalancedRandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    results[name] = auc

# --- 6. Hiperparametre Optimizasyonu (Random Forest) ---
rf = RandomForestClassifier(class_weight=class_weights, random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=rf_params, n_iter=20,
    cv=3, verbose=2, random_state=42, n_jobs=1
)
rf_random.fit(X_train_res, y_train_res)
print("Best RF Params:", rf_random.best_params_)
print("Best RF Score:", rf_random.best_score_)

# --- 7. Stacking Ensemble Model ---
estimators = [
    ('lr', LogisticRegression(max_iter=1000, class_weight=class_weights, random_state=42)),
    ('rf', RandomForestClassifier(**rf_random.best_params_, class_weight=class_weights, random_state=42)),
    ('svc', SVC(probability=True, class_weight=class_weights, random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
)
stacking_clf.fit(X_train_res, y_train_res)
y_pred_stack = stacking_clf.predict(X_test_scaled)
print(f"\nStacking Ensemble Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(classification_report(y_test, y_pred_stack, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_stack))

# --- 8. ROC AUC Skoru Stacking Model için ---
y_prob_stack = stacking_clf.predict_proba(X_test_scaled)
stack_auc = roc_auc_score(y_test_bin, y_prob_stack, average="macro", multi_class="ovr")
results["Stacking Ensemble"] = stack_auc

# --- 9. Model ROC AUC Skorlarını Yazdır ---
print("\nModel ROC AUC Scores:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

# --- 10. En iyi modeli kaydet ---
best_model_name = max(results, key=results.get)
if best_model_name == "Stacking Ensemble":
    best_model = stacking_clf
else:
    best_model = models.get(best_model_name, stacking_clf)  # default stacking

import os
import joblib
if not os.path.exists("models"):
    os.makedirs("models")

print("Current working directory:", os.getcwd())
print("Models klasörü var mı?", os.path.exists("models"))
print("Models klasörü içeriği:", os.listdir("models") if os.path.exists("models") else "Yok")

import joblib
import os

joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_').lower()}_model.pkl")
joblib.dump(scaler,"models/scaler.pkl")
joblib.dump(le, "models/label_encoder.pkl")
import os
print("Current working directory:", os.getcwd())

# --- 11. Önemli Özellikler (Random Forest için) ---
if best_model_name in ["Random Forest", "Balanced RF", "Stacking Ensemble"]:
    if best_model_name == "Stacking Ensemble":
        rf_model = stacking_clf.named_estimators_['rf']
    else:
        rf_model = best_model

    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index)
        plt.title("Feature Importances")
        plt.show(block=True)

# --- 12. SHAP ile Model Yorumlama (Random Forest) ---
try:
    if best_model_name in ["Random Forest", "Balanced RF", "Stacking Ensemble"]:
        if best_model_name == "Stacking Ensemble":
            rf_model = stacking_clf.named_estimators_['rf']
        else:
            rf_model = best_model

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train_res)

        shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)
except Exception as e:
    print("SHAP çalıştırılırken hata oluştu:", e)


