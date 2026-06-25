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

df_selected = df[pilihan_data].dropna().copy()

# Filter data besaran longsor 
kelas_populer = ['small', 'medium', 'large']
df_selected = df_selected[df_selected['landslide_size'].isin(kelas_populer)].copy()

# Filter data kategori longsor
kategori_populer = ['landslide', 'mudslide', 'debris_flow', 'rockfall']
df_selected = df_selected[df_selected['landslide_category'].isin(kategori_populer)].copy()

# Pengambilan sample
df_sample = df_selected.sample(n=1000, random_state=42).reset_index(drop=True)

# Karakteristik untuk pertimbangan
fitur = ['landslide_trigger', 'landslide_category']


# Gini Impurity
# G = 1 - sum(p(i)^2) 
# atau
# G = 1 - p(1)^2 - p(2)^2 - ... - p(n)^2
# Keterangan: Dengan p(n) atau p(i) sebagai proporsi dari setiap kelas dalam node.
#
# 
def hitung_gini(y):
    if len(y) == 0:
        return 0
    proporsi = y.value_counts() / len(y)
    return 1.0 - sum(proporsi ** 2)

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



print("Sedang melatih Decision Tree (Features: Trigger, Category, & Country)...")
pohon_kategorikal = bangun_pohon_kategorikal(df_sample, fitur, max_depth=8)

longsor_baru = {
    'landslide_trigger': 'flooding',
    'landslide_category': 'landslide',
    'country_name': 'Indonesia'
}

hasil_prediksi = prediksi_pohon_kategorikal(pohon_kategorikal, longsor_baru)

print("\n=== HASIL PREDIKSI BENCANA LONGSOR ===")
if isinstance(pohon_kategorikal, dict):
    print(f"Aturan Akar Pertama: Apakah {pohon_kategorikal['fitur']} == '{pohon_kategorikal['kategori_aturan']}'?")
else:
    print("Pohon langsung menghasilkan keputusan tunggal.")
    
print(f"Data Kejadian Baru: ")
for k, v in longsor_baru.items():
    print(f"  > {k}: {v}")
print(f"Prediksi Ukuran/Keparahan Longsor (Size): {hasil_prediksi.upper()}")