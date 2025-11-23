from tabpfn import TabPFNRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer

N_FOLDS = 10
df = pd.read_csv("clean_train.csv")
X = df.drop(columns=["aveOralM"])
y = df.loc[:,"aveOralM"]

def rmpse_metric(y_true, y_pred):
    percentage_error = (y_true - y_pred)
    rmpse = np.sqrt(np.mean(percentage_error**2))
    return rmpse

rmpse_scorer = make_scorer(rmpse_metric, greater_is_better=False)

regressor = TabPFNRegressor(device='cpu')

cv_scores = cross_val_score(
    estimator=regressor,
    X=X,
    y=y,
    cv=N_FOLDS,
    scoring=rmpse_scorer
)

print("\n=======================================================")
print(f"TabPFN RMPSE Scores (k={N_FOLDS} folds): {cv_scores}")
print(f"Mean RMPSE: {cv_scores.mean():.4f}")
print(f"Std Dev RMPSE: {cv_scores.std():.4f}")
print("=======================================================")


#=======================================================
#TabPFN RMPSE Scores (k=10 folds): [-0.30616777 -0.39665562 -0.35545356 -0.29752001 -0.26621467 -0.31434249
# -0.27894408 -0.2587421  -0.49847551 -0.44344374]
#Mean RMPSE: -0.3416
#Std Dev RMPSE: 0.0766
#=======================================================