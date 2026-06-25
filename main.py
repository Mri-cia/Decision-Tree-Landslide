import pandas as pd
import numpy as np

# Membersihkan Data
# Membaca, memilih, filter
df = pd.read_csv('Global_Landslide_Catalog_Export_rows.csv')

pilihan_data = [
    'country_name',        
    'landslide_trigger',   
    'landslide_category',  
    'landslide_size'       
]

# Pembersihan data-data yang hilang
df_selected = df[pilihan_data].dropna().copy()

# Filter data besaran longsor 
kelas_populer = ['small', 'medium', 'large']
df_selected = df_selected[df_selected['landslide_size'].isin(kelas_populer)].copy()

# Karakteristik untuk pertimbangan
fitur = ['landslide_trigger', 'landslide_category', 'country_name']


# Gini Impurity
# G = 1 - sum(p(i)^2) 
# atau
# G = 1 - p(1)^2 - p(2)^2 - ... - p(n)^2
# Keterangan: Dengan p(n) atau p(i) sebagai proporsi dari setiap kelas dalam node.
#
# .value_counts() digunakan untuk menghitung jumlah data unik.
# Proporsi sudah dalam berupa list untuk operasi sum
# Pengembalian berupa bilangan desimal antara 0-1
#
def hitung_gini(y):
    if len(y) == 0:
        return 0
    proporsi = y.value_counts() / len(y)
    return 1.0 - sum(proporsi ** 2)

# Mencari split
def cari_split_terbaik(data, fitur_list):
    best_gini = 999
    best_fitur = None
    best_kategori = None
    
    for f in fitur_list:
        kategori_unik = data[f].unique()
        
        for kat in kategori_unik:
            kiri = data[data[f] == kat]['landslide_size']
            kanan = data[data[f] != kat]['landslide_size']
            
            if len(kiri) == 0 or len(kanan) == 0: 
                continue
                
            w_gini = (len(kiri)/len(data)) * hitung_gini(kiri) + (len(kanan)/len(data)) * hitung_gini(kanan)
            
            if w_gini < best_gini:
                best_gini = w_gini
                best_fitur = f
                best_kategori = kat
                    
    return best_fitur, best_kategori


def bangun_pohon_kategorikal(data, fitur_list, kedalaman=0, max_depth=8):
    # Base Case: Jika data sudah murni atau mencapai batas kedalaman (diubah ke 8 agar menampung 3 fitur dengan baik)
    if hitung_gini(data['landslide_size']) == 0 or kedalaman >= max_depth or len(data) < 10:
        return data['landslide_size'].value_counts().idxmax()
    
    fitur_pilih, kategori_pilih = cari_split_terbaik(data, fitur_list)
    
    if fitur_pilih is None:
        return data['landslide_size'].value_counts().idxmax()
    
    data_kiri = data[data[fitur_pilih] == kategori_pilih]
    data_kanan = data[data[fitur_pilih] != kategori_pilih]
        
    node = {
        'fitur': fitur_pilih,
        'kategori_aturan': kategori_pilih,
        'kiri': bangun_pohon_kategorikal(data_kiri, fitur_list, kedalaman+1, max_depth),
        'kanan': bangun_pohon_kategorikal(data_kanan, fitur_list, kedalaman+1, max_depth)
    }
    return node


def prediksi_pohon_kategorikal(pohon, data_baru):
    if not isinstance(pohon, dict):
        return pohon
    
    nilai_input = data_baru[pohon['fitur']]
    
    if nilai_input == pohon['kategori_aturan']:
        return prediksi_pohon_kategorikal(pohon['kiri'], data_baru)
    else:
        return prediksi_pohon_kategorikal(pohon['kanan'], data_baru)

def hitung_akurasi_pohon(pohon, data_uji):
    prediksi_benar = 0
    total_data = len(data_uji)
    
    # Lakukan looping untuk memeriksa setiap baris data uji
    for index, baris in data_uji.iterrows():
        # Ambil tebakan dari pohon keputusan kustom Anda
        tebakan = prediksi_pohon_kategorikal(pohon, baris)
        
        # Bandingkan dengan label asli di dataset
        if tebakan == baris['landslide_size']:
            prediksi_benar += 1
            
    # Hitung persentase akhir
    skor_akurasi = (prediksi_benar / total_data) * 100
    return skor_akurasi


def hitung_metrik_detail(pohon, data_uji, kelas_target=['small', 'medium', 'large']):
# Inisialisasi dictionary komponen matriks untuk tiap kelas
    stats = {k: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for k in kelas_target}
    prediksi_benar_global = 0
    total_data = len(data_uji)
    
    # 1. Proses Looping Utama (Evaluasi Baris demi Baris)
    for index, baris in data_uji.iterrows():
        aktual = baris['landslide_size']
        prediksi = prediksi_pohon_kategorikal(pohon, baris)
        
        # Hitungan untuk Akurasi Global
        if aktual == prediksi:
            prediksi_benar_global += 1
            
        # Hitungan Komponen (TP, FP, FN, TN) menggunakan logika One-vs-Rest untuk setiap kelas
        for k in kelas_target:
            if aktual == k and prediksi == k:
                stats[k]['TP'] += 1
            elif aktual != k and prediksi == k:
                stats[k]['FP'] += 1
            elif aktual == k and prediksi != k:
                stats[k]['FN'] += 1
            elif aktual != k and prediksi != k:
                stats[k]['TN'] += 1

    # 2. Hitung Persentase Akurasi Global
    akurasi_global = (prediksi_benar_global / total_data) * 100 if total_data > 0 else 0.0

    # 2. Hitung Precision, Recall, dan F1-Score per kelas
    tabel_metrik = {}
    total_precision = 0
    total_recall = 0
    
    for k in kelas_target:
        tp = stats[k]['TP']
        fp = stats[k]['FP']
        fn = stats[k]['FN']
        
        # Rumus Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Rumus Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Rumus F1-Score = 2 * (P * R) / (P + R)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        tabel_metrik[k] = {
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100
        }
        
        total_precision += precision
        total_recall += recall

    # 3. Hitung Rata-rata Makro (Macro Average) untuk kesimpulan global
    macro_precision = (total_precision / len(kelas_target)) * 100
    macro_recall = (total_recall / len(kelas_target)) * 100
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0.0
    
    tabel_metrik['Average'] = {
        'Precision': macro_precision,
        'Recall': macro_recall,
        'F1-Score': macro_f1
    }
    
    return akurasi_global, tabel_metrik





# 2. Bagi data menjadi 80% Train dan 20% Test
batas_split = int(0.8 * len(df_selected))
data_train = df_selected.iloc[:batas_split]
data_test = df_selected.iloc[batas_split:]

# 3. Latih pohon HANYA menggunakan data_train
pohon_final = bangun_pohon_kategorikal(data_train, fitur, max_depth=8)

# 3. Hitung Detail Precision, Recall, dan F1-Score
akurasi_global, hasil_metrik = hitung_metrik_detail(pohon_final, data_test)

# 4. Cetak Output Laporan untuk Bab 4 Skripsi
print("\n" + "="*50)
print("         LAPORAN EVALUASI MODEL DETAIL         ")
print("="*50)
print(f"Jumlah Data Latih: {len(data_train)} baris")
print(f"Jumlah Data Uji  : {len(data_test)} baris")
print(f"Akurasi Model Decision Tree: {akurasi_global:.2f}%\n")

print(f"{'Kelas Karakteristik':<18} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-"*57)
for kelas, nilai in hasil_metrik.items():
    if kelas == 'Average':
        print("-"*57) # Garis pembatas untuk rata-rata akhir
    print(f"{kelas:<18} | {nilai['Precision']:<9.2f}% | {nilai['Recall']:<9.2f}% | {nilai['F1-Score']:<9.2f}%")
print("="*50)


longsor_baru = {
    'landslide_trigger': 'construction',
    'landslide_category': 'rockfall',
    'country_name': 'United States'
}

print("\n" + "="*50)
print("         MEMBUAT PREDIKSI BENCANA LONGSOR BARU         ")
print("="*50)

print(f"Data Kejadian Baru: ")
for k, v in longsor_baru.items():
    print(f"  > {k}: {v}")

print("\n\nMemberikan Prediksi Decision Tree...")
hasil_prediksi = prediksi_pohon_kategorikal(pohon_final, longsor_baru)

print("\n=== HASIL PREDIKSI BENCANA LONGSOR ===")
if isinstance(pohon_final, dict):
    print(f"Aturan Akar Pertama: Apakah [{pohon_final['fitur']} == '{pohon_final['kategori_aturan']}'?]")
else:
    print("Pohon langsung menghasilkan keputusan tunggal.")
    

print(f"Prediksi Ukuran/Keparahan Longsor (Size): {hasil_prediksi.upper()}")


print("="*50)