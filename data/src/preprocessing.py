import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor

from fsspec import available_protocols
from langdetect import DetectorFactory, LangDetectException, detect
import pandas as pd

from src.quality import AdvancedReviewQualityFilter, TurkishTextNormalizer
from src.translator import SeniorTextTranslator

DetectorFactory.seed = 0
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def _detect_lang_worker(text: str) -> str:
    """ProcessPoolExecutor için global dil tespit fonksiyonu."""
    if not isinstance(text, str) or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


class RestaurantDataCleaner:

    def __init__(self, text_col: str, target_lang: str = "tr") -> None:
        self.text_col = text_col
        self.target_lang = target_lang
        self.translator = SeniorTextTranslator(target_lang=target_lang)

        # Regex kalıpları
        self.html_pattern = re.compile(r"<[^>]+>")
        self.url_pattern = re.compile(r"http\S+|www\.\S+")
        self.whitespace_pattern = re.compile(r"\s+")

    def remove_nulls(self, df: pd.DataFrame) -> pd.DataFrame:

        initial_count = len(df)
        df = df.dropna(subset=[self.text_col]).copy()
        df[self.text_col] = df[self.text_col].astype(str).str.strip()
        df = df[df[self.text_col] != ""]

        removed = initial_count - len(df)
        logging.info(f"Eksik/Boş veri temizliği: {removed} satır çıkarıldı.")
        return df

    def clean_text_basic(self, df: pd.DataFrame) -> pd.DataFrame:

        def _clean_string(text: str) -> str:
            text = self.html_pattern.sub(" ", text)
            text = self.url_pattern.sub(" ", text)
            text = self.whitespace_pattern.sub(" ", text).strip()
            return text

        if "raw_text" not in df.columns:
            df["raw_text"] = df[self.text_col]

        df[self.text_col] = df[self.text_col].apply(_clean_string)
        return df

    def filter_and_detect_languages(self, df: pd.DataFrame) -> pd.DataFrame:

        num_workers = max(1, os.cpu_count() - 1)
        texts = df[self.text_col].tolist()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            languages = list(executor.map(_detect_lang_worker, texts, chunksize=1000))

        df["detected_lang"] = languages

        allowed_langs = self.translator.SUPPORTED_LANGUAGES.union({self.target_lang})

        initial_count = len(df)
        df = df[df["detected_lang"].isin(allowed_langs)].copy()

        removed = initial_count - len(df)
        logging.info(
            f"Dil Filtreleme: Desteklenmeyen/belirsiz {removed} satır çıkarıldı. "
            f"Kalan satırların dilleri: {allowed_langs}"
        )
        return df

    def translate_foreign_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        df["is_translated"] = False
        df["cleaned_text"] = df[self.text_col]

        unique_langs = df["detected_lang"].unique()

        for lang in unique_langs:
            if lang == self.target_lang:
                continue

            mask = df["detected_lang"] == lang
            foreign_texts = df.loc[mask, self.text_col].tolist()

            logging.info(
                f"'{lang}' dilindeki {len(foreign_texts)} satır Türkçe'ye çevriliyor..."
            )
            translated_texts = self.translator.translate_batch(
                foreign_texts, src_lang=lang
            )

            df.loc[mask, "cleaned_text"] = translated_texts
            df.loc[mask, "is_translated"] = True

        return df

    def pipeline(self, df: pd.DataFrame) -> pd.DataFrame:

        logging.info("Ön işleme hattı (pipeline) başlatıldı...")
        return (
            df.pipe(self.remove_nulls)
            .pipe(self.clean_text_basic)
            .pipe(self.filter_and_detect_languages)
            .pipe(self.translate_foreign_texts)
        )
class TrainingDataPreparer:
    """Modelleme aşaması öncesi normalizasyon, kalite filtreleme ve sütun ayıklama işlemlerini yönetir."""

    def __init__(self, text_col: str = "cleaned_text") -> None:
        self.text_col = text_col
        self.normalizer = TurkishTextNormalizer()
        self.quality_filter = AdvancedReviewQualityFilter(text_col=self.text_col)
        self.keep_columns = [
            "place_id",
            "place_name",
            "category",
            "category_list",
            "rating",
            "detected_lang",
            "visit_details",
            "cleaned_text",
        ]

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Metinlerdeki yapısal hataları ve boşlukları düzeltir."""
        logging.info("🛠️ Metin Normalizasyonu uygulanıyor (Hatalı boşluklar, uzatmalar, ekler)...")
        original_texts = df[self.text_col].copy()

        # Normalizasyonu uygula
        df[self.text_col] = df[self.text_col].apply(self.normalizer.normalize)

        # Değişen (özellikle aşırı tekrarları temizlenen) örnekleri logla
        changed_mask = original_texts != df[self.text_col]
        changed_samples = df[changed_mask]

        logging.info(f"🔄 Toplam {len(changed_samples):,} adet metinde normalizasyon/tekrar düzeltmesi yapıldı.")

        # Örnek olması açısından ilk 5 tanesini loglara yazdıralım
        count = 0
        for idx, row in changed_samples.head(30).iterrows():
            logging.info(
                f"   [Değişiklik Örneği {idx}] Önce: '{original_texts[idx]}' ---> Sonra: '{row[self.text_col]}'")
            count += 1

        return df

    def apply_quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Kısa, anlamsız, spam ve klavye çorbası yorumları eler ve elenenleri detaylı loglar."""
        logging.info("🧠 Kalite Filtresi uygulanıyor...")

        initial_count = len(df)

        # AdvancedReviewQualityFilter içindeki filtreleme maskesini burada da takip edebiliriz
        valid_mask = df[self.text_col].apply(self.quality_filter.is_high_quality_review)

        removed_reviews = df[~valid_mask].copy()
        df_filtered = df[valid_mask].copy()

        removed_count = initial_count - len(df_filtered)

        logging.info(
            f"✅ Kalite Filtresi Tamamlandı:\n"
            f"   - Elenen çöp/anlamsız/spam yorum: {removed_count:,} adet\n"
            f"   - Kalan model kalitesinde yorum: {len(df_filtered):,} adet"
        )

        # Elenenlerden örnekleri (ilk 5 tanesini) loglara yazdıralım ki neden elendiklerini görebilelim
        if not removed_reviews.empty:
            logging.info("🗑️ Elenen Yorumlardan Örnekler:")
            for idx, row in removed_reviews.iterrows():
                logging.info(f"   [Elenen {idx}] '{row[self.text_col]}'")

        return df_filtered

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gereksiz sütunları eler ve sadece modelleme için gerekenleri tutar."""
        logging.info("🧹 Gereksiz sütunlar temizleniyor...")
        available_columns = [col for col in self.keep_columns if col in df.columns]
        return df[available_columns].copy()

    def prepare_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm adımları sırayla (Chain of Responsibility) çalıştırır."""
        return (
            df.pipe(self.normalize_data)
              .pipe(self.apply_quality_filter)
              .pipe(self.filter_columns)
        )