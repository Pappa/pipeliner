import json
import pathlib
import tarfile
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import math
import logging
import glob


from sklearn.metrics import mean_squared_error


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SM_INPUT_MODEL"),
    )

    args = parser.parse_args()

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        logging.info([i.name for i in tar.getmembers()])
        tar.extractall(path="/opt/ml/models")

    logging.info(os.listdir("/opt/ml/models"))

    try:
        model = joblib.load("/opt/ml/models/rec.joblib")
    except Exception as e:
        logging.info(e)

    preds = model.predict(["U1001"])
    logging.info(f"preds: {preds}")

    test_path = "/opt/ml/processing/test/test.csv"
    # df = pd.read_csv(test_path, header=None)
    # df.columns = ["target"] + [f"feature_{x}" for x in range(df.shape[1] - 1)]

    # y_test = df.iloc[:, 0].to_numpy()
    # df.drop(df.columns[0], axis=1, inplace=True)

    # X_test = xgboost.DMatrix(df.values)

    # predictions = model.predict(X_test)

    # mse = mean_squared_error(y_test, predictions)
    # std = np.std(y_test - predictions)
    # report_dict = {
    #     "regression_metrics": {
    #         "mse": {"value": math.sqrt(mse), "standard_deviation": std},
    #     },
    # }

    # output_dir = "/opt/ml/processing/evaluation"
    # pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # evaluation_path = f"{output_dir}/evaluation.json"
    # with open(evaluation_path, "w") as f:
    #     f.write(json.dumps(report_dict))