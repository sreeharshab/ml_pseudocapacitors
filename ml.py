import pandas as pd
import numpy as np
import os
import pickle
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

if __name__=="__main__":
    X = pd.read_pickle("cached_df.pkl")
    materials = X["material"].tolist()
    excluded_materials = ['MoS2','VO2-M','VO2-R']
    X = X[[material not in excluded_materials for material in materials]]
    X = X.reset_index(drop=True)
    X.loc[X["material"]=="Nb2O5-T", "Band Gap"] = 1.925
    X.loc[X["material"]=="Nb2O5-TT", "Band Gap"] = 1.773
    X = X.drop(columns=["material", "formula", "structure", "composition"])
    X = X.drop([col for col in X.columns if "0.5" in col], axis=1)
    X = X.loc[:, (X!=0).any(axis=0)]
    X = X.loc[:, X.nunique() > 10]

    corr = X.corr(method="pearson").fillna(0)
    corr = corr.stack().reset_index()
    corr.columns = ['Feature1', 'Feature2', 'Correlation']
    corr = corr[corr['Feature1'] < corr['Feature2']]
    high_corr_pairs = corr[abs(corr['Correlation']) > 0.8]
    high_corr_pairs = high_corr_pairs.sort_values(by='Correlation', ascending=False)
    high_corr_pairs = high_corr_pairs.reset_index(drop=True)
    feature1 = high_corr_pairs['Feature1'].values
    feature2 = high_corr_pairs['Feature2'].values

    X = X.drop(columns=np.unique(feature2))
    X_arr = X.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    n_components = min(X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    loadings = np.abs(pca.components_.T)
    explained_variance_ratio = pca.explained_variance_ratio_
    feature_importance = np.dot(loadings, explained_variance_ratio)
    feature_ranking_indices = np.argsort(-feature_importance)   # Descending order
    selected_features_PCA = X.columns[feature_ranking_indices].values
    X = X[selected_features_PCA[0:20]]
    X.to_pickle("X.pkl")

    Y = pd.read_excel("exp_data.xlsx", header=0)
    Y = Y[["System", "Rev. Cap at 0.1C, mAh/g", "Rev. Cap at 5C, mAh/g"]]
    Y = Y.set_index("System")
    Y = Y.loc[materials, ["Rev. Cap at 0.1C, mAh/g", "Rev. Cap at 5C, mAh/g"]]
    Y = Y[[material not in excluded_materials for material in materials]]
    Y["capacity_ratio"] = (Y["Rev. Cap at 0.1C, mAh/g"].values - Y["Rev. Cap at 5C, mAh/g"].values)/Y["Rev. Cap at 0.1C, mAh/g"].values
    Y = Y["capacity_ratio"]
    Y.to_pickle("Y.pkl")

    column_combinations = list(combinations(X.columns,4))
    os.makedirs('jobs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    for index,combination in enumerate(column_combinations):
        with open(f"jobs/job_{index}.pkl", "wb") as f:
            pickle.dump(combination, f)
    