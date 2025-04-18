import sys
from data_processor import DataProcessor

if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    plot_asset = sys.argv[3] if len(sys.argv) > 3 else None

    processor = DataProcessor(input_csv_path=input_csv, drop_na=True)
    processor.process_all(output_csv_path=output_csv, plot_asset=plot_asset)
