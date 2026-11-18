# Train / CV / test summary

Train = model scored on its own training data (80 train trips). CV = 5-fold GroupKFold by trip on the train trips. Test = 10 student-test trips.

### Top-1 accuracy

| Model | Train | CV | Test |
|---|---|---|---|
| LogisticRegression | 0.899 | 0.898 | 0.893 |
| GaussianNB | 0.885 | 0.885 | 0.879 |
| KNN | 0.917 | 0.890 | 0.880 |
| DecisionTree | 0.920 | 0.894 | 0.888 |
| RandomForest | 0.930 | 0.902 | 0.895 |
| HistGradientBoosting | 0.918 | 0.900 | 0.894 |
| LightGBM | 0.949 | 0.903 | 0.898 |
| MLP | 0.921 | 0.898 | 0.888 |
| LightGBM_tuned | 0.988 | 0.905 | 0.902 |
| LightGBM_tuned + HMM (final) | n/a | n/a | 0.952 |

### Macro F1 (candidate level)

| Model | Train | CV | Test |
|---|---|---|---|
| LogisticRegression | 0.935 | 0.935 | 0.930 |
| GaussianNB | 0.914 | 0.913 | 0.907 |
| KNN | 0.944 | 0.931 | 0.926 |
| DecisionTree | 0.950 | 0.934 | 0.930 |
| RandomForest | 0.959 | 0.942 | 0.939 |
| HistGradientBoosting | 0.951 | 0.941 | 0.938 |
| LightGBM | 0.968 | 0.942 | 0.940 |
| MLP | 0.951 | 0.934 | 0.931 |
| LightGBM_tuned | 0.994 | 0.941 | 0.940 |
| LightGBM_tuned + HMM (final) | n/a | n/a | n/a |
