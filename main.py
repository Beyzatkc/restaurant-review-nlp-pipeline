import pandas as pd
from config import RAW_DATA_PATH

# Parquet dosyasını oku
df = pd.read_parquet(RAW_DATA_PATH)

# Sütun isimlerini ekrana yazdır
print("Veri Setindeki Sütunlar:")
print(df.columns.tolist())