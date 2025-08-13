
import argparse
from pathlib import Path
from embed_index import build_from_csv
from ranker import train_ranker
from clarifier import tune_from_ambiguous_json

DEF_CSV = str(Path(__file__).resolve().parent / "enhanced_ground_truth.csv")
DEF_TRAIN = str(Path(__file__).resolve().parent / "training_data.json")
DEF_AMBIG = str(Path(__file__).resolve().parent / "ambiguous_queries.json")

def main(csv_path=DEF_CSV, train_json=DEF_TRAIN, ambig_json=DEF_AMBIG):
    print("[1/3] Building embedding index from:", csv_path)
    build_from_csv(csv_path)
    print("[2/3] Training ranker from:", train_json)
    train_ranker(train_json)
    print("[3/3] Tuning clarifier from:", ambig_json)
    info = tune_from_ambiguous_json(ambig_json)
    print("Clarifier:", info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEF_CSV)
    parser.add_argument("--train", default=DEF_TRAIN)
    parser.add_argument("--ambig", default=DEF_AMBIG)
    args = parser.parse_args()
    main(args.csv, args.train, args.ambig)
