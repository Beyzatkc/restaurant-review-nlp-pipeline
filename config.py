from pathlib import Path

# Proje ana dizini (config.py kök dizinde olduğu için parent yeterlidir)
BASE_DIR = Path(__file__).resolve().parent

# Veri yolları (Görseldeki dosya adına göre güncellendi)
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "restoran_kafe_verisi.parquet"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_reviews.parquet"
PROCESSED2_DATA_PATH = BASE_DIR / "data" / "processed" / "curated_reviews.parquet"

# Veri seti sütun ismi (Verinizdeki yorum sütununun adı neyse onu yazmalısınız)
TEXT_COLUMN = "review_text"  # Filtrelemek istediğiniz yorumların bulunduğu sütun adı

# Dil filtresi parametreleri
TARGET_LANGUAGE = "tr"

FINAL_COLUMNS = [
    "place_id",
    "place_name",
    "category",
    "category_list",
    "rating",
    "visit_details",
    "cleaned_text",
]
