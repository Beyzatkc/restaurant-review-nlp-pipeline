import pandas as pd

# Veriyi yükle
df = pd.read_json("data/model_input/silver_summaries.jsonl", lines=True)

# Veri setinden rastgele 1 satır seç (bunu bir Series objesine dönüştürüyoruz)
rastgele_satir = df.sample(n=1).iloc[0]

# YÖNTEM 1: Rastgele satırı sözlük (dictionary) formatında alt alta ve eksiksiz yazdırmak
print("\n--- Rastgele Seçilen Satır (Tüm İçerik) ---")
print(rastgele_satir.to_dict())

# YÖNTEM 2: Sadece o kesilen uzun özet metnini tam okumak
print("\n--- Sadece Hedef Özet (Gemma'nın Çıktısı) ---")
print(rastgele_satir["target_summary"])

# YÖNTEM 3: Pandas'ın metin kısaltma limitlerini tamamen kaldırmak
pd.set_option('display.max_colwidth', None)  # Sütun genişliği limitini kaldırır
pd.set_option('display.max_columns', None)   # Tüm sütunları gösterir

print("\n--- Tablo Görünümü (Limit Yok) ---")
# Rastgele seçtiğimiz satırı ekranda düzgün bir tablo olarak görmek için tekrar DataFrame'e çeviriyoruz
print(rastgele_satir.to_frame().T)