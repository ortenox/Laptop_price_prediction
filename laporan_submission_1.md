# Laporan Proyek Machine Learning - Gema Eka Putra Prameswara

## Domain Proyek

Proyek ini bertujuan untuk memprediksi harga laptop berdasarkan beberapa spesifikasi pada lapto seperti merk, ukuran layar, spesifikasi hardware, dan sistem operasi. Proyek ini menarik untuk dilakukan karena harga laptop sangat dipengaruhi oleh berbagai aspek teknis dan dengan memanfaatkan machine learning, pengguna bisa mendapatkan rekomendasi harga yang sesuai dengan laptop yang mereka butuhkan.

Proyek ini penting untuk dilakukan karena dapat membantu konsumen dalam mengambil keputusan pembelian yang lebih optimal dan bermanfaat bagi produsen dalam menetapkan harga produk yang kompetitif. Dengan prediksi harga yang akurat, produsen juga dapat menganalisis tren pasar dan menyesuaikan strategi penjualan mereka.

**Rubrik/Kriteria Tambahan (Opsional)**:
Menurut penelitian, penggunaan machine learning dalam prediksi harga telah diterapkan pada berbagai domain seperti properti dan mampu memberikan estimasi harga yang cukup akurat. Salah satunya studi pada prediksi harga rumah menunjukkan bahwa algoritma seperti Linear Regression mampu memberikan hasil prediksi yang baik. Pendekatan serupa dapat diaplikasikan pada kasus harga laptop.

[Predicting house price in India using linear regression machine learning algorithms](https://ieeexplore.ieee.org/abstract/document/9853185/)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

- Bagaimana memprediksi harga laptop berdasarkan fitur-fitur teknis dan spesifikasi?

- Algoritma mana yang memberikan hasil prediksi terbaik antara Linear Regression dan Random Forest Regressor?

### Goals

- Membuat model machine learning untuk memprediksi harga laptop dalam satuan euro.

- Menentukan algoritma terbaik berdasarkan metrik evaluasi regresi.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menggunakan algoritma Linear Regression sebagai model baseline karena kesederhanaannya dan kemampuannya untuk interpretasi.

- Menggunakan algoritma Random Forest Regressor sebagai model pembanding untuk menangani hubungan non-linear antar fitur.

- Model dievaluasi menggunakan metrik MSE, MAE, dan RMSE untuk mengukur akurasi prediksi.

## Data Understanding

Dataset yang digunakan berisi 1303 baris data laptop dengan berbagai fitur teknis. Dataset dapat diunduh di: Kaggle Laptop Price Dataset

Contoh: [Laptop Price](https://www.kaggle.com/datasets/muhammetvarl/laptop-price). 

### Variabel-variabel pada Laptop Price dataset adalah sebagai berikut:

- Company: Merk laptop

- TypeName: Jenis laptop (Ultrabook, Notebook, dll)

- Inches: Ukuran layar

- ScreenResolution: Resolusi layar

- Cpu: Jenis prosesor

- Ram: Besaran RAM

- Memory: Jenis dan kapasitas penyimpanan

- Gpu: GPU yang digunakan

- OpSys: Sistem operasi

- Weight: Berat laptop

- Price_euros: Harga laptop (target)

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menggunakan fungsi info() untuk mendapatkan informasi mengenai jumlah dan tipe data.

- Menggunakan fungsi describe(include="all") untuk melihat bentuk data, seperti mean, modus, dan max dari dari setiap kolom pada dataset.

## Data Preparation
- Membersihkan kolom 'Ram' dan 'Weight' agar hanya berisi nilai numerik.

- Menggunakan one-hot encoding pada fitur kategorikal seperti Company, TypeName, dan OpSys.

- Menstandarisasi fitur numerik seperti Inches, Ram, dan Weight.

- Memisahkan data menjadi 80% data pelatihan dan 20% data pengujian menggunakan train_test_split.

## Modeling
Pada tahap ini, dua algoritma machine learning digunakan untuk membangun model prediksi harga laptop, yaitu Linear Regression dan Random Forest Regressor. Kedua model dipilih untuk membandingkan performa antara model sederhana dengan model berbasis ensemble learning.

1. Linear Regression
Linear Regression digunakan sebagai baseline model. Model ini mengasumsikan hubungan linear antara fitur dan target.
Model dilatih menggunakan semua fitur numerik dan hasil encoding dari fitur kategorikal.

### Parameter yang digunakan:

- Menggunakan pengaturan default dari LinearRegression() pada scikit-learn.

#### Kelebihan Linear Regression:

- Sederhana dan cepat untuk dilatih.

- Mudah untuk diinterpretasikan, karena koefisien model dapat menunjukkan pengaruh masing-masing fitur terhadap harga.

### Kekurangan Linear Regression:

- Kurang mampu menangkap hubungan non-linear yang kompleks antar variabel.

- Sensitif terhadap multikolinearitas dan outlier.

2. Random Forest Regressor
Random Forest Regressor digunakan sebagai model lanjutan. Ini adalah algoritma ensemble yang membangun banyak decision tree dan menggabungkannya untuk meningkatkan akurasi prediksi.

### Parameter yang digunakan:

- n_estimators=100: Menggunakan 100 pohon keputusan.

- random_state=42: agar hasil tetap konsisten.

### Kelebihan Random Forest:

- Mampu menangani hubungan non-linear dan interaksi kompleks antar fitur.

- Lebih tahan terhadap overfitting dibandingkan model pohon tunggal.

- Secara otomatis melakukan feature selection melalui proses pemodelan.

### Kekurangan Random Forest:

- Lebih lambat untuk dilatih dan dievaluasi dibanding model linear sederhana.

- Interpretasi model lebih sulit dibanding Linear Regression.

## Evaluation
Model dievaluasi menggunakan tiga metrik regresi:

- MSE (Mean Squared Error): Rata-rata kesalahan kuadrat dari prediksi terhadap nilai aktual. Semakin kecil nilainya, semakin baik model.

- MAE (Mean Absolute Error): Rata-rata selisih absolut antara nilai prediksi dan aktual. Cocok digunakan saat ingin mengetahui seberapa besar rata-rata kesalahan model.

- RMSE (Root Mean Squared Error): Akar kuadrat dari MSE yang memiliki satuan yang sama dengan target dan lebih peka terhadap outlier.

### Hasil evaluasi:

- Model Linear Regression memiliki RMSE dan MSE yang lebih kecil daripada Random Forest, tetapi memiliki MAE yang sedikit lebih besar.

- Model Random Forest Regressor memiliki MAE paling kecil, menandakan secara rata-rata kesalahan prediksinya paling mendekati nilai aktual.

### Hasil Evaluasi Model

#### Linear Regression:
- Mean Squared Error (MSE): 79632.22858702157
- Mean Absolute Error (MAE): 198.8027756391272
- Root Mean Squared Error (RMSE): 282.1918294122308

#### Random Forest Regressor:
- Mean Squared Error (MSE): 101893.13497129994
- Mean Absolute Error (MAE): 191.82069132366357
- Root Mean Squared Error (RMSE): 319.2070409174897

### Interpretasi Hasil
- Linear Regression memiliki nilai MSE dan RMSE yang lebih kecil dibandingkan Random Forest Regressor, yang menunjukkan bahwa error rata-rata kuadrat dan deviasi prediksi terhadap harga aktual lebih kecil.

- Random Forest Regressor memiliki nilai MAE sedikit lebih rendah dibanding Linear Regression, yang berarti pada rata-rata absolut, prediksi Random Forest sedikit lebih dekat ke nilai aktual.

Dengan rentang harga laptop antara 174 euro hingga 6099 euro, kesalahan rata-rata sekitar 190–200 euro menunjukkan performa prediksi yang cukup baik untuk kebutuhan prakiraan harga.

**Rubrik/Kriteria Tambahan (Opsional)**: 
1. Mean Squared Error (MSE)
MSE mengukur rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual.
![MSE](https://drive.google.com/file/d/1UZQ8OZT0_lhzHKMAxOHFLHLIg0NkMKDI/view?usp=drive_link)

Semakin kecil nilai MSE, semakin kecil rata-rata kesalahan kuadrat model terhadap data aktual. MSE lebih memperbesar penalti untuk prediksi yang jauh dari nilai sebenarnya karena kuadrat perbedaan.

2. Mean Absolute Error (MAE)
MAE mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual.
![MAE](https://drive.google.com/file/d/1rPNxixFIBQWXg-B7D0i-Nn93ak7XIXmW/view?usp=drive_link)

MAE lebih mudah diinterpretasikan karena berada dalam satuan yang sama dengan target. MAE tidak terlalu memperbesar penalti untuk kesalahan besar (tidak mengkuadratkan error).

3. RMSE adalah akar dari MSE, sehingga memiliki satuan yang sama dengan target. 
![RMSE](https://drive.google.com/file/d/18w_9nzVF-vH2gomhFXzg7GpWuNA0PHg_/view?usp=drive_link)
![image](https://github.com/user-attachments/assets/9fa318b7-bd1f-475f-9c42-369df6a14010)

RMSE lebih sensitif terhadap outlier dibanding MAE, karena error dikalikan dengan dirinya sendiri sebelum dijumlahkan.

## Struktur Laporan

Laporan ini disusun secara sistematis mengikuti alur CRISP-DM dengan bagian:

- Domain Proyek

- Business Understanding

- Data Understanding

- Data Preparation

- Modeling

- Evaluation

**---Ini adalah bagian akhir laporan---**
