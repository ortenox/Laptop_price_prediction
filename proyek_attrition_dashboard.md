
# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Maju

## Business Understanding

Perusahaan Jaya Jaya Maju mengalami peningkatan angka attrition atau pengunduran diri karyawan dalam kurun waktu tertentu. Hal ini berdampak pada operasional karena kehilangan SDM yang berpengalaman dapat menghambat produktivitas. Oleh karena itu, perusahaan membutuhkan solusi berbasis data untuk memahami faktor-faktor penyebab attrition dan mengantisipasinya secara proaktif.

## Permasalahan Bisnis

- Mengidentifikasi faktor-faktor yang paling mempengaruhi pengunduran diri karyawan.
- Mengetahui profil karyawan yang berisiko tinggi untuk resign.
- Memberikan rekomendasi kepada manajemen dalam mengurangi angka attrition.
- Menyediakan dashboard interaktif sebagai media analisis.

## Cakupan Proyek

- Melakukan eksplorasi dan preprocessing data employee.
- Membangun model klasifikasi untuk prediksi attrition menggunakan Logistic Regression dan Random Forest.
- Mengevaluasi performa model.
- Menyimpan hasil preprocessing ke dalam database SQLite.
- Menyajikan visualisasi dan dashboard menggunakan Metabase.

## Persiapan

**Sumber Data:** [Dataset karyawan internal yang berisi data demografis, kepuasan kerja, dan informasi kompensasi.](https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv)

**Setup environment:**

```python
# Install library
!pip install pandas matplotlib seaborn scikit-learn --quiet

# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
```

**Load dan Eksplorasi Data:**

```python
df = pd.read_csv('employee_data.csv')
df.info()
df.isnull().sum()
sns.countplot(data=df, x='Attrition')
```

**Preprocessing dan Feature Engineering:**

```python
cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
df.drop(['EmployeeId', 'Over18', 'StandardHours'], axis=1, inplace=True)
df = df.dropna(subset=['Attrition'])
```

**Split dan Scaling Data:**

```python
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Modeling:**

```python
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
```

**Evaluasi Model:**

```python
print(classification_report(y_test, y_pred_log))
print("ROC AUC Logistic:", roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1]))

print(classification_report(y_test, y_pred_rf))
print("ROC AUC Random Forest:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))
```

**Feature Importance:**

```python
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=feature_importance_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Fitur terhadap Attrition')
```

**Simpan Model dan Scaler:**

```python
import joblib
joblib.dump(rf_model, 'rf_attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

**Simpan ke SQLite untuk Dashboard:**

```python
import sqlite3
df_final = df.copy()
df_final['Attrition'] = df_final['Attrition'].map({0: 'Stay', 1: 'Resign'})
conn = sqlite3.connect('employee_attrition.db')
df_final.to_sql('processed_employee_data', conn, if_exists='replace', index=False)
conn.close()
```

## Business Dashboard

Dashboard interaktif dibuat menggunakan **Metabase**. Visualisasi mencakup:
- Jumlah karyawan yang resign vs tetap
- Rata-rata usia berdasarkan status attrition
- Rata-rata JobSatisfaction
- Rata-rata penghasilan bulanan (MonthlyIncome)

Contoh tampilan dashboard:
<img width="540" alt="METABASE" src="https://github.com/user-attachments/assets/65adc784-061f-4561-b8e8-48d3d49dae90" />

## Conclusion

Model machine learning yang dikembangkan mampu mengidentifikasi faktor-faktor utama penyebab attrition. Beberapa insight penting dari dashboard:
- Karyawan yang resign memiliki usia lebih muda dan pendapatan lebih rendah.
- Job Satisfaction yang rendah juga berkontribusi terhadap keputusan untuk resign.
- Perusahaan dapat meningkatkan retensi dengan menaikkan kompensasi dan memperhatikan kepuasan kerja karyawan.

Seluruh data hasil preprocessing disimpan ke database dan dapat digunakan untuk monitoring berkelanjutan melalui Metabase.
