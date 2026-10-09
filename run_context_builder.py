import logging
import pandas as pd
from src.context import ReviewContextBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    input_path = "data/processed/curated_reviews.parquet"
    output_path = "data/model_input/llm_ready_contexts.parquet"

    logging.info(f"Temizlenmiş veri okunuyor: {input_path}")
    df = pd.read_parquet(input_path)

    # 20-25 arası yorum LLM context window'u (örneğin Llama-3 veya Mistral 8k) için
    # ideal uzunluktadır (~1500-2500 token yapar).
    builder = ReviewContextBuilder(
        max_reviews_per_place=40,
        min_reviews_per_place=10,
        ideal_min_words=10,
        ideal_max_words=80
    )

    df_llm_ready = builder.build_context(df)

    # Örnek olarak ilk mekanın nasıl göründüğüne bakalım
    sample_place = df_llm_ready.iloc[0]
    logging.info(f"\n--- ÖRNEK LLM GİRDİSİ ({sample_place['place_name']}) ---")
    logging.info(f"Toplam Yorum: {sample_place['total_reviews_analyzed']} | Ort. Puan: {sample_place['avg_rating']}")
    logging.info(f"\n{sample_place['llm_input_context'][:500]}...\n--------------------------")

    # Modeli besleyeceğimiz nihai veriyi kaydetme
    df_llm_ready.to_parquet(output_path, index=False)
    logging.info(f"🎉 LLM Eğitim / Çıkarım verisi kaydedildi: {output_path}")


if __name__ == "__main__":
    main()