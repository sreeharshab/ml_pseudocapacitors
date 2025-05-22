import pickle
import sys
import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import cross_val_score

if __name__=="__main__":
    X = pd.read_pickle("X.pkl")
    Y = pd.read_pickle("Y.pkl")
    job_id = int(sys.argv[1])
    with open(f"jobs/job_{job_id}.pkl", "rb") as f:
        selected_features = pickle.load(f)

    X_subset = X[list(selected_features)]

    tpot = TPOTRegressor(generations=50, population_size=50, verbosity=2, random_state=42, scoring="r2")
    tpot.fit(X_subset, Y)
    tpot.export(f"models/model_{job_id}.py")
    score = tpot._optimized_pipeline_score

    result = {"job_id": job_id, "features": selected_features, "cv_score": score}
    df_results = pd.DataFrame([result])
    df_results.to_pickle(f"results/result_{job_id}.pkl")