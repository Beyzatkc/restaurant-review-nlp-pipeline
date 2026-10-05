import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from langdetect import DetectorFactory, LangDetectException, detect
import pandas as pd
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