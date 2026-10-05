from src.tester import ProcessedDataTester

if __name__ == "__main__":
    # Sınıfı başlat
    tester = ProcessedDataTester()

    # 10 rastgele satır ile testi çalıştır
    tester.run_test(sample_size=1000)