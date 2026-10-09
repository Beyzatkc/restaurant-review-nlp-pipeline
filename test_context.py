import pandas as pd


def inspect_random_place():
    # Dosya yolunu yeni mimarimize göre belirtiyoruz
    file_path = "data/model_input/llm_ready_contexts.parquet"

    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"❌ Dosya bulunamadı: {file_path}")
        return

    # Veri setinden rastgele 1 mekan seç (her çalıştırdığında farklı gelir)
    random_place = df.sample(n=1).iloc[0]

    # Konsola temiz ve okunaklı bir şekilde yazdır
    print("\n" + "=" * 70)
    print(f"🍽️ MEKAN ADI       : {random_place['place_name']}")
    print(f"📊 ORTALAMA PUAN   : {random_place['avg_rating']} / 5.0")
    print(f"📈 TOPLAM YORUM    : {random_place['total_reviews_analyzed']} (İçinden en iyi ~25 tanesi seçildi)")
    print("=" * 70)
    print("📝 LLM'E GÖNDERİLECEK BAĞLAM (PROMPT İÇERİĞİ):\n")
    print(random_place['llm_input_context'])
    print("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_random_place()