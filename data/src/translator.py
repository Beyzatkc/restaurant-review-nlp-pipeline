import logging
from typing import List

logging.basicConfig(level=logging.INFO)


class SeniorTextTranslator:
    MODEL_MAPPINGS = {
        "en": "Helsinki-NLP/opus-mt-tc-big-en-tr",
        "ar": "Helsinki-NLP/opus-mt-ar-tr",
    }
    SUPPORTED_LANGUAGES = set(MODEL_MAPPINGS.keys())

    def __init__(self, target_lang: str = "tr") -> None:
        import torch

        self.torch = torch
        self.target_lang = target_lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = {}
        self._tokenizers = {}
        logging.info(f"Translator başlatıldı. Çalışma Cihazı: {self.device.upper()}")

    def _get_model(self, src_lang: str):
        from transformers import MarianMTModel, MarianTokenizer

        if src_lang not in self.MODEL_MAPPINGS:
            return None, None

        model_name = self.MODEL_MAPPINGS[src_lang]

        if src_lang not in self._models:
            try:
                logging.info(f"Çeviri modeli yükleniyor: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                if self.device == "cuda":
                    model = model.half().to(self.device)
                else:
                    model = model.to(self.device)

                model.eval()
                self._tokenizers[src_lang] = tokenizer
                self._models[src_lang] = model
            except Exception as e:
                logging.warning(f"'{src_lang}' için model yüklenemedi: {e}")
                return None, None

        return self._models[src_lang], self._tokenizers[src_lang]

    def translate_batch(self, texts: List[str], src_lang: str, batch_size: int = 64) -> List[str]:
        if not texts or src_lang == self.target_lang:
            return texts
        if src_lang not in self.SUPPORTED_LANGUAGES:
            return texts

        model, tokenizer = self._get_model(src_lang)
        if not model or not tokenizer:
            return texts

        translated_texts = []

        with self.torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i: i + batch_size]
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(self.device)

                translated_tokens = model.generate(**inputs, num_beams=1, max_new_tokens=128)
                decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                translated_texts.extend(decoded)

        return translated_texts