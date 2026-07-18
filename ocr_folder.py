# run_ocr_single.py
from aic26.pipeline.ocr import OCRExtractor

if __name__ == "__main__":
    ocr = OCRExtractor()
    ocr.process(
        input_dir=r"D:\AIC\AI-Challenge\AIC2025\data\keyframes\K11_V001",
        output_csv=r"D:\AIC\AI-Challenge\AIC2025\data\ocr_results.csv",
        parallel_mode="image",
        num_workers=4,
    )