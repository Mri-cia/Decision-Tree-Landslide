import pandas as pd
import numpy as np

df = pd.read_csv('Global_Landslide_Catalog_Export_rows.csv')

pilihan_data = [
    'country_name', 
    'longitude', 
    'latitude', 
    'landslide_trigger', 
    'landslide_size'
    ]

df_selected = df[pilihan_data].dropna().copy()

kelas_populer = ['small', 'medium', 'large']
df_selected = df_selected[df_selected['landslide_size'].isin(kelas_populer)].copy()

df_sample = df_selected

fitur = ['landslide_trigger', 'country_name', 'latitude', 'longitude']



def hitung_gini(y):
    if len(y) == 0:
        return 0
    proporsi = y.value_counts() / len(y)
    return 1.0 - sum(proporsi ** 2)


def cari_split_terbaik(data, fitur_list):
    best_gini = 999
    best_fitur = None
    best_threshold = None
    
    for f in fitur_list:
        if data[f].dtype in [np.float64, np.int64]:
            thresholds = np.percentile(data[f], np.linspace(10, 90, 10))
            for t in thresholds:
                kiri = data[data[f] <= t]['landslide_size']
                kanan = data[data[f] > t]['landslide_size']
                
                if len(kiri) == 0 or len(kanan) == 0: continue
                w_gini = (len(kiri)/len(data)) * hitung_gini(kiri) + (len(kanan)/len(data)) * hitung_gini(kanan)
                if w_gini < best_gini:
                    best_gini = w_gini; best_fitur = f; best_threshold = t
        else:
            kategori_unik = data[f].unique()
            for kat in kategori_unik:
                kiri = data[data[f] == kat]['landslide_size']
                kanan = data[data[f] != kat]['landslide_size']
                
                if len(kiri) == 0 or len(kanan) == 0: continue
                w_gini = (len(kiri)/len(data)) * hitung_gini(kiri) + (len(kanan)/len(data)) * hitung_gini(kanan)
                if w_gini < best_gini:
                    best_gini = w_gini; best_fitur = f; best_threshold = kat
                    
    return best_fitur, best_threshold


def bangun_pohon(data, fitur_list, kedalaman=0, max_depth=3):
    if hitung_gini(data['landslide_size']) == 0 or kedalaman >= max_depth or len(data) < 10:
        return data['landslide_size'].value_counts().idxmax()
    
    fitur_pilih, threshold_pilih = cari_split_terbaik(data, fitur_list)
    
    if fitur_pilih is None:
        return data['landslide_size'].value_counts().idxmax()
    
    is_numerik = data[fitur_pilih].dtype in [np.float64, np.int64]
    
    if is_numerik:
        data_kiri = data[data[fitur_pilih] <= threshold_pilih]
        data_kanan = data[data[fitur_pilih] > threshold_pilih]
    else:
        data_kiri = data[data[fitur_pilih] == threshold_pilih]
        data_kanan = data[data[fitur_pilih] != threshold_pilih]
        
    node = {
        'fitur': fitur_pilih,
        'threshold': threshold_pilih,
        'is_numerik': is_numerik,
        'kiri': bangun_pohon(data_kiri, fitur_list, kedalaman+1, max_depth),
        'kanan': bangun_pohon(data_kanan, fitur_list, kedalaman+1, max_depth)
    }
    return node


def prediksi_pohon(pohon, data_baru):
    if not isinstance(pohon, dict):
        return pohon
    
    nilai_fitur = data_baru[pohon['fitur']]
    
    if pohon['is_numerik']:
        kondisi = nilai_fitur <= pohon['threshold']
    else:
        kondisi = nilai_fitur == pohon['threshold']
        
    if kondisi:
        return prediksi_pohon(pohon['kiri'], data_baru)
    else:
        return prediksi_pohon(pohon['kanan'], data_baru)
    
print("Sedang melatih Decision Tree pada data Landslide NASA...")
pohon_longsor = bangun_pohon(df_sample, fitur, max_depth=3)

longsor_baru = {
    'landslide_trigger': 'downpour', # Hujan lebat instan
    'country_name': 'Indonesia',
    'latitude': -7.25,
    'longitude': 112.75
}

hasil_prediksi = prediksi_pohon(pohon_longsor, longsor_baru)

print("\n=== HASIL PREDIKSI BENCANA LONGSOR ===")
if pohon_longsor['is_numerik']:
    print(f"Aturan Utama: Apakah {pohon_longsor['fitur']} <= {pohon_longsor['threshold']}?")
else:
    print(f"Aturan Utama: Apakah {pohon_longsor['fitur']} bertipe '{pohon_longsor['threshold']}'?")
    
print(f"Data Kejadian Baru: ")
for k, v in longsor_baru.items():
    print(f"  > {k}: {v}")
print(f"Prediksi Ukuran/Keparahan Longsor: {hasil_prediksi.upper()}")