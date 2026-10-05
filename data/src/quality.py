import math
import re
import logging
import pandas as pd
from typing import Dict, Any


class TurkishTextNormalizer:
    """
    Modern NLP modelleri için metinleri standartlaştıran, işlemci dostu (Regex tabanlı)
    hızlı metin normalizasyon sınıfı.
    """

    def __init__(self) -> None:
        self.char_rep_pattern = re.compile(r"([a-zA-ZçğıöşüÇĞIÖŞÜ])\1{2,}")
        self.punct_rep_pattern = re.compile(r"([!?.,])\1+")
        self.punct_space_pattern = re.compile(r"\s+([!?.,;:])")
        self.plural_suffix_pattern = re.compile(r"\b([a-zA-ZçğıöşüÇĞIÖŞÜ]+)\s+(lar|ler)\b", flags=re.IGNORECASE)
        self.whitespace_pattern = re.compile(r"\s+")
        self.keyboard_mash_pattern = re.compile(r"\b[b-df-hj-np-tv-zBC-DF-HJ-NP-TV-Z]{5,}\b")

        # YENİ: Kelimelerin arasına konulmuş bitişik noktaları (örn: "Hizmetleri.cok.guzel")
        # boşluğa çeviren regex (harf ile harf arasındaki noktayı yakalar)
        self.dot_to_space_pattern = re.compile(r"(?<=[a-zA-ZçğıöşüÇĞIÖŞÜ])\.(?=[a-zA-ZçğıöşüÇĞIÖŞÜ])")

    def normalize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # YENİ: Noktalarla ayrılmış kelimeleri önce düzgün boşluklu hale getir
        text = self.dot_to_space_pattern.sub(" ", text)

        text = self.keyboard_mash_pattern.sub("", text)
        text = self.char_rep_pattern.sub(r"\1", text)
        text = self.punct_rep_pattern.sub(r"\1", text)
        text = self.plural_suffix_pattern.sub(r"\1\2", text)
        text = self.punct_space_pattern.sub(r"\1", text)
        text = self.whitespace_pattern.sub(" ", text).strip()

        return text


class AdvancedReviewQualityFilter:
    TURKISH_VOWELS = set("aeıioöuüAEIİOÖUÜ")

    def __init__(
            self,
            min_word_count: int = 2,
            max_consecutive_consonants: int = 7,
            min_vowel_ratio: float = 0.15,
            max_bad_word_ratio: float = 0.25,
            text_col: str = "cleaned_text"
    ) -> None:
        self.min_word_count = min_word_count
        self.max_consecutive_consonants = max_consecutive_consonants
        self.min_vowel_ratio = min_vowel_ratio
        self.max_bad_word_ratio = max_bad_word_ratio
        self.text_col = text_col

        self.word_consonant_pattern = re.compile(
            r"[bcçdfgğhjklmnprsştvyzBCÇDFGĞHJKLMNPRSŞTVYZ]{" + str(self.max_consecutive_consonants) + r",}"
        )

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        return -sum(p * math.log2(p) for p in prob)

    def is_high_quality_review(self, text: str) -> bool:
        if not isinstance(text, str) or len(text) < 5:
            return False

        words = text.split()
        if len(words) < self.min_word_count:
            return False

        is_long_review = len(words) >= 12

        # 1. Tekrarlayan kelime kontrolü
        unique_ratio = len(set(words)) / len(words)
        if len(words) >= 6 and unique_ratio < 0.25:
            return False

        # 2. Kelime Bazlı Klavye Çorbası Oranı Kontrolü
        bad_word_count = sum(1 for word in words if self.word_consonant_pattern.search(word))
        bad_word_ratio = bad_word_count / len(words)
        if bad_word_ratio > self.max_bad_word_ratio:
            return False

        # 3. Sesli Harf Oranı Kontrolü
        alphabetic_chars = [c for c in text if c.isalpha()]
        if len(alphabetic_chars) >= 6:
            vowel_count = sum(1 for c in alphabetic_chars if c in self.TURKISH_VOWELS)
            current_vowel_ratio = vowel_count / len(alphabetic_chars)

            effective_vowel_limit = self.min_vowel_ratio - 0.03 if is_long_review else self.min_vowel_ratio
            if current_vowel_ratio < effective_vowel_limit:
                return False

        # 4. Entropi Kontrolü
        if len(text) > 20 and self._calculate_entropy(text) < 1.3:
            return False

        return True

    def filter_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)

        valid_mask = df[self.text_col].apply(self.is_high_quality_review)

        removed_reviews = df[~valid_mask].copy()
        df_filtered = df[valid_mask].copy()

        removed = initial_count - len(df_filtered)

        logging.info(
            f"✅ Kalite Filtresi:\n"
            f"   - Elenen çöp/anlamsız yorum: {removed:,} adet\n"
            f"   - Kalan model kalitesinde yorum: {len(df_filtered):,} adet"
        )

        logging.info("🗑️ ELENEN YORUMLAR:")
        for index, row in removed_reviews.iterrows():
            logging.info(f"   [{index}] {row[self.text_col]}")

        return df_filtered