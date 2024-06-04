import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

# Set seed for reproducibility
np.random.seed(42)

# Define directories
data_dir = Path('data/competition_data')  # Directory containing the competition data
output_dir = Path('.')  # Directory to output results
csv_dir = Path('tmp')  # Directory containing CSV files from previous steps
retrain = False  # Flag to determine whether to retrain models

# Load base interim submissions
base_interim_submissions = pd.read_csv(csv_dir / 'fixed_interim_submission.csv')


# Function to get test data
def get_tests(test_path):
    """
    Function to get test data from a given path.

    Args:
    test_path (Path): Path to the test data.

    Returns:
    DataFrame: DataFrame containing the test data.
    """
    test_data = {
        'essay_id_comp': [],
        'essay_text': []
    }
    for test_file in test_path.glob('*.txt'):
        with open(test_file) as f:
            essay_id = test_file.stem
            essay_text = f.read()
            test_data['essay_id_comp'].append(essay_id)
            test_data['essay_text'].append(essay_text)
    return pd.DataFrame(test_data)


# Load test data
test_df = get_tests(data_dir / 'test')


# Function to map indexes to text
def indexes_to_text(indexes, essay_text):
    """
    Function to map indexes to text.

    Args:
    indexes (str): String of indexes separated by spaces.
    essay_text (str): Essay text to map indexes to.

    Returns:
    str: Text corresponding to the given indexes.
    """
    indexes = [int(index) for index in indexes.split()]
    return ''.join([essay_text[index] for index in indexes])


# Add discourse text to base interim submissions
base_interim_submissions['discourse_text'] = base_interim_submissions.apply(
    lambda x: indexes_to_text(x['predictionstring'],
                              test_df[test_df['essay_id_comp'] == x['essay_id_comp']]['essay_text'].values[0]),
    axis=1)

# Load and preprocess training data
columns = ['id', 'discourse_text', 'discourse_type', 'discourse_effectiveness']
train_df = pd.read_csv(data_dir / "train.csv", low_memory=False)[columns]
train_df.rename(columns={'id': 'essay_id'}, inplace=True)
train_df = train_df[~(train_df['discourse_type'] == 'Unannotated')]


# Function to load essay text
def load_essay_text(essay_id, is_train=True, data_dir=data_dir):
    """
    Function to load essay text.

    Args:
    essay_id (str): ID of the essay to load.
    is_train (bool, optional): Whether the essay is from the training set. Defaults to True.
    data_dir (Path, optional): Directory containing the data. Defaults to data_dir.

    Returns:
    str: Text of the essay.
    """
    if is_train:
        essay_path = Path(data_dir / f"train/{essay_id}.txt")
    else:
        essay_path = Path(data_dir / f"test/{essay_id}.txt")
    with open(essay_path, 'r') as f:
        return f.read()


# Add essay text to training and test dataframes
train_df['essay_text'] = train_df['essay_id'].apply(lambda x: load_essay_text(x, is_train=True))
base_interim_submissions['essay_text'] = base_interim_submissions['essay_id_comp'].apply(
    lambda x: load_essay_text(x, is_train=False))

# Combine discourse text, type, and essay text
train_df['discourse_text_plus_type'] = train_df['discourse_text'] + "\t" + train_df['discourse_type'] + '\t' + \
                                       train_df['essay_text'].apply(lambda x: x[:500])

# Reverse mapping for discourse type
reverse_discourse_map = {0: 'Lead', 1: 'Position', 2: 'Claim', 3: 'Evidence', 4: 'Counterclaim', 5: 'Rebuttal',
                         6: 'Concluding Statement'}

# Add discourse text, type, and essay text to base interim submissions
discourse_type_text = base_interim_submissions['discourse_type'].apply(lambda x: reverse_discourse_map[int(x)])
base_interim_submissions['discourse_text_plus_type'] = base_interim_submissions['discourse_text'] + "\t" + \
                                                       discourse_type_text + '\t' + \
                                                       base_interim_submissions['essay_text'].apply(lambda x: x[:500])

# Check if models are available
try:
    with io.open(output_dir / 'models/effectiveness_models.pkl', 'rb') as f:
        models = pickle.load(f)
except FileNotFoundError:
    retrain = True
    print("Models not found. Retraining models.")
else:
    print("Models found. Skipping retraining.")

# Retrain model if required
if retrain:
    train_df['label'] = train_df['discourse_effectiveness'].apply(lambda x: 1 if x == 'Effective' else 0)

    X = train_df['discourse_text_plus_type']
    y = train_df['label']
    groups = train_df['essay_id']

    class_weight = {
        0: 0.5,
        1: 0.99
    }

    kf = GroupKFold(n_splits=5)

    # Initialize params dictionary
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'subsample': 0.8,
        'subsample_freq': 1
    }

    models = []
    accuracy_scores = []
    fold = 1

    for train_index, test_index in kf.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_train, groups_test = groups.iloc[train_index], groups.iloc[test_index]

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lgbm', LGBMClassifier(class_weight=class_weight, **params))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"Fold: {fold}, Accuracy: {accuracy}")
        fold += 1

        accuracy_scores.append(accuracy)
        models.append(pipeline)

    print(f"Mean Accuracy: {np.mean(accuracy_scores)} with std: {np.std(accuracy_scores)}")

    # Save models
    (output_dir / 'tmp').mkdir(exist_ok=True)
    with io.open(output_dir / 'models/effectiveness_models.pkl', 'wb') as f:
        pickle.dump(models, f)

else:
    # Load pre-trained models
    with io.open(output_dir / 'models/effectiveness_models.pkl', 'rb') as f:
        models = pickle.load(f)


# Function to predict effectiveness
def predict_effectiveness(disourse_texts, models):
    """
    Function to predict effectiveness.

    Args:
    disourse_texts (Series): Series of discourse texts to predict effectiveness for.
    models (list): List of models to use for prediction.

    Returns:
    ndarray: Array of predicted effectiveness scores.
    """
    predictions = []
    for model in models:
        predictions.append(model.predict(disourse_texts))
    return np.mean(predictions, axis=0)


# Predict effectiveness and add labels to base interim submissions
base_interim_submissions['label'] = predict_effectiveness(base_interim_submissions['discourse_text'], models)

# Create final submission dataframe
final_submission_df = base_interim_submissions.copy()

# Add scores for discourse effectiveness
final_submission_df['score_discourse_effectiveness_0'] = 1.0 - base_interim_submissions['label']
final_submission_df['score_discourse_effectiveness_1'] = base_interim_submissions['label']

# Save final submission
columns_in_submission = ['essay_id_comp', 'predictionstring',
                         'score_discourse_effectiveness_0',
                         'score_discourse_effectiveness_1', 'discourse_type']
final_submission_df = final_submission_df[columns_in_submission]
final_submission_df.to_csv(output_dir / "final_submission.csv", index=False)
