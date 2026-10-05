import logging
import pandas as pd
from src.preprocessing import TrainingDataPreparer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    input_path = "data/processed/cleaned_reviews.parquet"
    output_path = "data/processed/curated_reviews.parquet"

    logging.info(f"💾 İşlenmiş (Çevirileri yapılmış) veri okunuyor: {input_path}")
    df = pd.read_parquet(input_path)
    logging.info(f"Giriş verisi boyutu: {df.shape}")

    preparer = TrainingDataPreparer(text_col="cleaned_text")

    # 1. Aşama: Kalite Filtresi (Spam, klavye çorbası, kısa yorumlar elenir)
    # 2. Aşama: Sütun Ayıklama
    df_ready = preparer.prepare_pipeline(df)

    # Nihai veriyi kaydetme
    df_ready.to_parquet(output_path, index=False)
    logging.info(f"🎉 Model eğitim verisi başarıyla kaydedildi: {output_path}")
    logging.info(f"Çıktı verisi boyutu: {df_ready.shape}")


if __name__ == "__main__":
    main()