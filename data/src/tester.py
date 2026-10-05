import pandas as pd
import config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ProcessedDataTester:
    def __init__(self, data_path: str = config.PROCESSED_DATA_PATH):
        self.data_path = data_path


        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)

    def load_data(self) -> pd.DataFrame:

        logging.info(f"Veri yükleniyor: {self.data_path}")
        return pd.read_parquet(self.data_path)

    def show_summary(self, df: pd.DataFrame) -> None:
        logging.info(f"Toplam Satır Sayısı: {len(df):,}")
        if 'is_translated' in df.columns:
            logging.info(f"Çevrilen Yorum Sayısı: {df['is_translated'].sum():,}")

    def show_random_samples(self, df: pd.DataFrame, n: int = 10) -> None:
        sample_df = df.sample(n=n, random_state=None)

        columns_to_show = [config.TEXT_COLUMN, "detected_lang", "is_translated", "cleaned_text"]
        available_columns = [col for col in columns_to_show if col in sample_df.columns]

        print(f"\n--- RASTGELE {n} ÖRNEK ---")
        print(sample_df[available_columns])

    def run_test(self, sample_size: int = 10) -> None:
        try:
            df = self.load_data()
            self.show_summary(df)
            self.show_random_samples(df, n=sample_size)
        except Exception as e:
            logging.error(f"Veri test edilirken hata oluştu: {e}")