import pandas as pd
import numpy as np

from graphviz import Digraph
# import os

# jalur_graphviz = r"C:\Program Files\Graphviz\bin" 

# os.environ["PATH"] += os.pathsep + jalur_graphviz

def visualisasikan_pohon_graphviz(pohon, dot=None, node_id="0"):
    # Jika pertama kali dipanggil, buat objek Digraph baru
    if dot is None:
        dot = Digraph(comment='Pohon Keputusan Longsor', format='png')
        dot.attr(rankdir='TB', size='10,10')
        dot.attr(dpi='300')          # Meningkatkan ketajaman gambar (High Resolution)
        dot.attr(size='30,30!')      # Memberikan ruang kanvas yang jauh lebih luas (dalam inci)
        dot.attr(ratio='compress')
        # Mengatur desain kotak agar rapi
        dot.attr('node', shape='box', style='filled,rounded', color='black', fontname='helvetica')

    # BASE CASE: Jika simpul adalah daun (string jawaban seperti 'small', 'medium', etc)
    if not isinstance(pohon, dict):
        # Beri warna hijau muda untuk daun keputusan final
        dot.node(node_id, f"PREDIKSI:\n{pohon.upper()}", fillcolor='#95e1d3', style='filled,bold')
        return dot

    # IF NODE: Jika simpul adalah aturan pemisah (dictionary)
    label_node = f"Apakah {pohon['fitur']}\n== '{pohon['kategori_aturan']}'?"
    dot.node(node_id, label_node, fillcolor='#eaffd0')

    # Buat ID unik untuk cabang anak kiri dan anak kanan
    id_kiri = node_id + "_kiri"
    id_kanan = node_id + "_kanan"

    # Rekursi ke cabang KIRI (Memenuhi syarat / YA)
    visualisasikan_pohon_graphviz(pohon['kiri'], dot, id_kiri)
    dot.edge(node_id, id_kiri, label=" Ya ", color='#21bf73', fontcolor='#21bf73', penwidth='2')

    # Rekursi ke cabang KANAN (Tidak memenuhi syarat / TIDAK)
    visualisasikan_pohon_graphviz(pohon['kanan'], dot, id_kanan)
    dot.edge(node_id, id_kanan, label=" Tidak ", color='#fe346e', fontcolor='#fe346e', penwidth='2')

    return dot



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

# Mencari Cabang
# Dikembalikan kombinasi fitur dan kategori yang paling rendah dari Gini Impurity
def cari_split_terbaik(data, fitur_list):
    best_gini = 999
    best_fitur = None
    best_kategori = None
    
    # Mengevaluasi fitur satu per satu
    for f in fitur_list:
        kategori_unik = data[f].unique()
        
        # Mengevaluasi Kategori satu per satu
        for kat in kategori_unik:
            kiri = data[data[f] == kat]['landslide_size']
            kanan = data[data[f] != kat]['landslide_size']
            
            if len(kiri) == 0 or len(kanan) == 0: 
                continue
            
            # Menghitung Weighted Gini Impurity
            w_gini = (len(kiri)/len(data)) * hitung_gini(kiri) + (len(kanan)/len(data)) * hitung_gini(kanan)
            
            # Mengevaluasi apakah sudah lebih baik dari sebelumnya
            if w_gini < best_gini:
                best_gini = w_gini
                best_fitur = f
                best_kategori = kat
                    
    return best_fitur, best_kategori

# Membangun Pohon 
# Mengembalikan node akar
def bangun_pohon_kategorikal(data, fitur_list, kedalaman=0, max_depth=8):
    # Base Case: Jika data sudah murni atau mencapai batas kedalaman (diubah ke 8 agar menampung 3 fitur dengan baik)
    if hitung_gini(data['landslide_size']) == 0 or kedalaman >= max_depth or len(data) < 10:
        return data['landslide_size'].value_counts().idxmax()
    
    # Mengambil kombinasi
    fitur_pilih, kategori_pilih = cari_split_terbaik(data, fitur_list)
    
    # Jika tidak ada split yang valid ditemukan, kembalikan kategori mayoritas
    if fitur_pilih is None:
        return data['landslide_size'].value_counts().idxmax()
    
    data_kiri = data[data[fitur_pilih] == kategori_pilih]
    data_kanan = data[data[fitur_pilih] != kategori_pilih]
        
    # Menggunakan rekursi untuk terus membangun cabang sampai bersih
    node = {
        'fitur': fitur_pilih,
        'kategori_aturan': kategori_pilih,
        'kiri': bangun_pohon_kategorikal(data_kiri, fitur_list, kedalaman+1, max_depth), # <-- rekursi akan mengambalikan kategori sehingga jika dicontohkan [kiri: small]
        'kanan': bangun_pohon_kategorikal(data_kanan, fitur_list, kedalaman+1, max_depth) # <-- rekursi akan mengambalikan kategori sehingga jika dicontohkan [kanan: medium]
    }
    return node

# Prediksi dari Pohon
# Mengembalikan daun yang sesuai dengan input data
def prediksi_pohon_kategorikal(pohon, data_baru):
    if not isinstance(pohon, dict):
        return pohon # Pohon sudah mencapai daun.
    
    # Mengambil nilai baru dari inisialisasi atau rekursi
    nilai_input = data_baru[pohon['fitur']]
    
    # Melakukan rekursi sampai mencapai daun
    if nilai_input == pohon['kategori_aturan']:
        return prediksi_pohon_kategorikal(pohon['kiri'], data_baru)
    else:
        return prediksi_pohon_kategorikal(pohon['kanan'], data_baru)


# Metrik Evaluasi Model
# Mengembalikan akurasi global dan tabel metrik (Precision, Recall, F1-Score)
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
            
        # Hitungan Komponen (TP, FP, FN, TN) 
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
        # Menghitung berapa banyak yang diprediksi benar oleh model
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Rumus Recall = TP / (TP + FN)
        # Menghitung berapa banyak yang berhasil ditemukan oleh model
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Rumus F1-Score = 2 * (P * R) / (P + R)
        # Menghitung Rata-rata harmonik
        # Condong untuk mendekati angka kecil
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Hasil metrik
        tabel_metrik[k] = {
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1-Score': f1 * 100
        }
        
        # Menghitung total untuk keseluruhan
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





# Bagi data menjadi 80% Train dan 20% Test
batas_split = int(0.8 * len(df_selected))
data_train = df_selected.iloc[:batas_split]
data_test = df_selected.iloc[batas_split:]

# Melatih Model untuk pohon dari data yang dilatih
pohon_final = bangun_pohon_kategorikal(data_train, fitur, max_depth=8)

# Menguji model dari data untuk pengujian
akurasi_global, hasil_metrik = hitung_metrik_detail(pohon_final, data_test)

# Mencetak Evaluasi
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

# Membuat data baru untuk prediksi
longsor_baru = {
    'landslide_trigger': 'construction',
    'landslide_category': 'rockfall',
    'country_name': 'United States'
}

# Menampilkan hasil prediksi untuk data baru
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


# Pembuatan visualisasi pohon
grafik = visualisasikan_pohon_graphviz(pohon_final)
grafik.render('visualisasi_pohon_longsor', view=True)
print("Selesai! File 'visualisasi_pohon_longsor.png' telah berhasil dibuat.")