import json
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class ChatMLFormatter:
    """
    Silver veriyi (JSONL) Qwen/Llama modellerinin Fine-Tuning
    aşamasında kullanabileceği System/User/Assistant formatına çevirir.
    """

    def __init__(self, data_dir: Path):
        self.input_file = data_dir / "model_input" / "silver_summaries.jsonl"
        self.train_file = data_dir / "model_input" / "qwen_train.jsonl"
        self.test_file = data_dir / "model_input" / "qwen_test.jsonl"

        self.system_prompt = (
            "Sen profesyonel ve tarafsız bir restoran eleştirmenisin. "
            "Aşağıdaki müşteri yorumlarını analiz et ve mekan hakkında sadece "
            "'Lezzet', 'Servis' ve 'Atmosfer' odaklı kısa, tek paragraflık "
            "yapılandırılmış bir özet çıkar."
        )

    def _format_row(self, row) -> dict:
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Müşteri Yorumları:\n{row['input_context']}"},
                {"role": "assistant", "content": row['target_summary']}
            ]
        }

    def process_and_split(self, test_size: float = 0.03):
        logging.info(f"Silver veri okunuyor: {self.input_file}")

        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        df = pd.DataFrame(data)
        formatted_data = df.apply(self._format_row, axis=1).tolist()

        train_data, test_data = train_test_split(
            formatted_data,
            test_size=test_size,
            random_state=42
        )

        # Train dosyasını kaydet
        with open(self.train_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # Test dosyasını kaydet
        with open(self.test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logging.info(f"✅ Hazırlık Tamamlandı! Train: {len(train_data)} | Test: {len(test_data)}")