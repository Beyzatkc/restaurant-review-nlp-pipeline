import logging
import shutil

import pandas as pd
import pyarrow.parquet as pq
import config
import os
import gc
import torch
import glob
from src.preprocessing import RestaurantDataCleaner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    logging.info("🚀 6 Milyonluk Veri İşleme Hattı Başlatılıyor (Güvenli Mod)...")

    output_dir = "processed_batches"
    os.makedirs(output_dir, exist_ok=True)

    cleaner = RestaurantDataCleaner(
        text_col=config.TEXT_COLUMN,
        target_lang=config.TARGET_LANGUAGE
    )

    parquet_file = pq.ParquetFile(config.RAW_DATA_PATH)
    total_rows = parquet_file.metadata.num_rows
    logging.info(f"Toplam Ham Veri Sayısı: {total_rows:,}")

    batch_size = 100_000

    existing_files = glob.glob(os.path.join(output_dir, "batch_*.parquet"))
    start_batch = len(existing_files)

    if start_batch > 0:
        logging.info(
            f"⚠️ Daha önce işlenmiş {start_batch} paket bulundu. {start_batch + 1}. paketten devam ediliyor...")

    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):

        if i < start_batch:
            continue

        chunk_df = batch.to_pandas()
        logging.info(f"\n--- [Paket {i + 1}] {len(chunk_df):,} Satır İşleniyor ---")
        cleaned_chunk = cleaner.pipeline(chunk_df)

        batch_filename = os.path.join(output_dir, f"batch_{i:04d}.parquet")
        cleaned_chunk.to_parquet(batch_filename, index=False)
        logging.info(f"💾 Paket {i + 1} diske kaydedildi: {batch_filename}")

        del chunk_df
        del cleaned_chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info("\n✅ TÜM VERİLER İŞLENDİ. Parçalar birleştiriliyor...")

    all_files = sorted(glob.glob(os.path.join(output_dir, "batch_*.parquet")))
    combined_df = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)

    combined_df.to_parquet(config.PROCESSED_DATA_PATH, index=False)
    logging.info(f"🎉 İşlem Tamam! Tekil dosya kaydedildi: {config.PROCESSED_DATA_PATH}")

    shutil.rmtree(output_dir)
    logging.info(f"🗑️ Geçici klasör ({output_dir}) ve içindeki tüm dosyalar otomatik olarak silindi.")


if __name__ == "__main__":
    main()