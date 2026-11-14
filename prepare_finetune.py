import logging
from pathlib import Path
from src.finetune_prep import ChatMLFormatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def main():
    base_dir = Path(__file__).resolve().parent / "data"

    formatter = ChatMLFormatter(data_dir=base_dir)
    formatter.process_and_split(test_size=0.03)


if __name__ == "__main__":
    main()