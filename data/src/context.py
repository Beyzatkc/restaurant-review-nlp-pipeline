
"""
    LLM (Büyük Dil Modeli) özetlemesi için mekan bazlı yorumları
    duygu dağılımını (rating) koruyarak ve token sınırlarını gözeterek birleştirir.
    """
import logging
from itertools import groupby

import numpy as np
import pandas as pd
from sympy.strategies import condition


class ReviewContextBuilder:
    def __init__(
            self,
            max_reviews_per_place: int = 25,
            min_reviews_per_place: int = 10,
            ideal_min_words: int = 10,
            ideal_max_words: int = 80,
            text_col: str = "cleaned_text",
    )-> None:
        self.max_reviews = max_reviews_per_place
        self.min_reviews = min_reviews_per_place
        self.ideal_min_words = ideal_min_words
        self.ideal_max_words = ideal_max_words
        self.text_col = text_col

    def _calculate_informativeness_score(self,df: pd.DataFrame) -> pd.DataFrame:
        """
                Her yoruma bir 'Bilgi Yoğunluğu Skoru' atar.
                Çok kısa veya aşırı uzun yorumlar cezalandırılır; ideal aralıktaki yorumlar ödüllendirilir.
                """
        logging.info("Yorumlar için 'Bilgi Yoğunluğu Skoru' hesaplanıyor...")
        df['word_count'] = df[self.text_col].str.split().str.len()
        conditions = [
            (df['word_count'] < self.ideal_min_words),
            (df['word_count'] > self.ideal_max_words),
        ]
        choices = [
            df['word_count'] * 0.5,
            self.ideal_max_words
        ]

        df['info_score'] = np.select(conditions, choices,default=df['word_count'])
        return df

    def _stratified_sample_group(self, group: pd.DataFrame) -> str:
        """
                Tek bir mekan (place_id) için rating dağılımını koruyarak örnekleme yapar
                ve yorumları LLM'in anlayacağı bir prompt formatında birleştirir.
                """
        total_reviews = len(group)
        if total_reviews == 0:
            return ""

        if total_reviews <= self.max_reviews:
            sampled = group.sort_values(by='info_score', ascending=False)
        else:
            rating_props = group['rating'].value_counts(normalize=True)
            selected_indices = []
            for rating , prop in rating_props.items():
                alloc = int(round(prop * self.max_reviews))
                if alloc == 0:
                    continue

                subset = group[group['rating'] == rating].nlargest(alloc, 'info_score')
                selected_indices.extend(subset.index.tolist())
            sampled = group.loc[selected_indices]

        sampled = sampled.sample(frac=1.0, random_state=42)
        formatted_reviews = sampled.apply(
            lambda row: f"- [Puan: {int(row['rating'])}/5]: {row[self.text_col]}",
            axis=1
        ).tolist()

        return "\n".join(formatted_reviews)

    def build_context(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_places = df['place_id'].nunique()
        logging.info(f"{initial_places:,} farklı mekan için LLM bağlamı inşa ediliyor...")

        logging.info(f"🧹 {self.min_reviews} yorumdan az olan mekanlar eleniyor...")
        review_counts = df['place_id'].value_counts()
        valid_places = review_counts[review_counts >= self.min_reviews].index
        df = df[df['place_id'].isin(valid_places)].copy()

        filtered_places = df['place_id'].nunique()
        removed_places = initial_places - filtered_places
        logging.info(f"Filtreleme tamamlandı. {removed_places:,} mekan elendi. Kalan mekan: {filtered_places:,}")

        df = self._calculate_informativeness_score(df)

        logging.info("⚖️ Stratified Sampling (Duygu korumalı örnekleme) uygulanıyor...")

        context_series = df.groupby('place_id')[['rating', self.text_col, 'info_score']].apply(
            self._stratified_sample_group
        )

        result_df = context_series.reset_index(name='llm_input_context')
        meta_df = df.groupby('place_id').agg(
            place_name=('place_name', 'first'),
            total_reviews_analyzed=('place_id', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()

        result_df = result_df.merge(meta_df, on='place_id')

        result_df['avg_rating'] = result_df['avg_rating'].round(2)

        logging.info(f"Bağlam inşası tamamlandı. Toplam benzersiz mekan: {len(result_df):,}")
        return result_df





