import argparse
import os
import pandas as pd
import joblib
import logging
from pipeliner.recommendations.recommender import UserBasedRecommender

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--input", type=str, default=os.environ.get("SM_INPUT_DIR"))

    args = parser.parse_args()

    logging.info(f"SM_MODEL_DIR: {args.model_dir}")
    logging.info(f"SM_INPUT_DIR: {args.input}")

    user_item_matrix = pd.read_csv(
        f"{args.input}/data/user_item_matrix/user_item_matrix.csv",
        index_col="user_id",
    )
    similarity_matrix = pd.read_csv(
        f"{args.input}/data/user_similarity_matrix/user_similarity_matrix.csv",
        index_col="user_id",
    )

    rec = UserBasedRecommender(10, 10).fit((similarity_matrix, user_item_matrix))

    joblib.dump(rec, os.path.join(args.model_dir, "rec.joblib"))
