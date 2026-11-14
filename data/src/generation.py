import json
import logging
import os
import time
from pathlib import Path
import pandas as pd
from groq import Groq


class SilverDataGenerator:
    """
    Groq API (Llama 3.3 70B) üzerinden yüksek hızlı ve checkpoint destekli
    sentetik özet üretici.
    """

    def __init__(self, data_dir: Path, api_key: str):
        self.data_dir = data_dir
        self.input_file = data_dir / "model_input" / "llm_ready_contexts.parquet"
        self.output_file = data_dir / "model_input" / "silver_summaries.jsonl"

        # Önceki oturumlarda (Gemini dahil) üretilmiş mekan ID'lerini yükler
        self.completed_ids = self._load_completed_ids()
        self.client = Groq(api_key=api_key)

    def _load_completed_ids(self) -> set:
        completed = set()
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        completed.add(data["place_id"])
                    except json.JSONDecodeError:
                        continue
        logging.info(
            f"🔄 Önceki oturumlardan {len(completed):,} adet tamamlanmış özet bulundu."
        )
        return completed

    def _get_system_prompt(self) -> str:
        return (
            "Sen profesyonel ve tarafsız bir restoran eleştirmenisin. "
            "Aşağıdaki müşteri yorumlarını analiz et ve mekan hakkında kısa, net, tek paragraflık "
            "yapılandırılmış bir özet çıkar. Sadece 'Lezzet', 'Servis' ve 'Atmosfer' odaklı "
            "gerçekçi bir sentez yap. Yorum yapılmamış özellikleri kesinlikle uydurma."
        )

    def _call_llm_api(self, context: str) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt()
                        },
                        {
                            "role": "user",
                            "content": context
                        },
                    ],
                    temperature=0.3,
                    max_tokens=250,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "rate_limit" in err_msg:
                    wait_time = 10 * (attempt + 1)
                    logging.warning(
                        f"⏳ Groq Limiti uyarısı. {wait_time} sn bekleniyor..."
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(f"❌ API Hatası: {e}")
                    raise e
        raise Exception("Maksimum deneme sayısına ulaşıldı.")

    def generate_daily_batch(self, daily_limit: int = 2000):
        if not self.input_file.exists():
            logging.error(f"❌ Girdi dosyası bulunamadı: {self.input_file}")
            return

        df = pd.read_parquet(self.input_file)

        # Gemini'nin zaten işlediği 17 mekanı otomatik olarak filtreler
        df_pending = df[~df["place_id"].isin(self.completed_ids)]

        if df_pending.empty:
            logging.info("✅ Tüm veri seti için özetler zaten üretilmiş!")
            return

        logging.info(
            f"⏳ Kalan mekan sayısı: {len(df_pending):,}. Bugünkü hedef: {daily_limit}"
        )
        df_today = df_pending.head(daily_limit)
        success_count = 0

        with open(self.output_file, "a", encoding="utf-8") as f:
            for idx, row in df_today.iterrows():
                try:
                    summary = self._call_llm_api(row["llm_input_context"])

                    result_record = {
                        "place_id": row["place_id"],
                        "place_name": row["place_name"],
                        "input_context": row["llm_input_context"],
                        "target_summary": summary,
                    }

                    f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    f.flush()

                    success_count += 1
                    if success_count % 25 == 0 or success_count == daily_limit:
                        logging.info(
                            f"⚡ {success_count}/{daily_limit} mekan tamamlandı."
                        )

                    # Groq çok hızlı olduğu için 1.5 saniye bekleme yeterlidir
                    time.sleep(1.5)

                except Exception as e:
                    logging.error(f"⚠️ Hata nedeniyle işlem durduruldu: {e}")
                    break

        logging.info(
            f"🏁 Bugünkü işlem tamamlandı. Yeni üretilen özet: {success_count}"
        )