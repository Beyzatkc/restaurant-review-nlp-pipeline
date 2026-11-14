import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.generation import SilverDataGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    load_dotenv()
    # GROQ_API_KEY değişkenini .env dosyasından okuyoruz
    API_KEY = os.getenv("GROQ_API_KEY")

    if not API_KEY:
        logging.error(
            "❌ Groq API Anahtarı bulunamadı! .env dosyasındaki GROQ_API_KEY alanını kontrol edin."
        )
        return

    base_dir = Path(__file__).resolve().parent / "data"
    generator = SilverDataGenerator(data_dir=base_dir, api_key=API_KEY)

    # Groq ile günde 2000 mekanı rahatlıkla işleyebilirsiniz
    generator.generate_daily_batch(daily_limit=2000)


if __name__ == "__main__":
    main()