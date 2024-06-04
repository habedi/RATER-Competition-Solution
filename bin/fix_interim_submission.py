import os
from pathlib import Path

import pandas as pd

# Define directories and file paths
COMPETITION_DATA_DIR = Path('data/competition_data')
BASE_SUBMISSION_DATA_DIR = Path('.')
CSV_DIR = Path("tmp")

# Define column names
COLUMNS_IN_SUBMISSION = ['essay_id_comp', 'predictionstring',
                         'score_discourse_effectiveness_0',
                         'score_discourse_effectiveness_1', 'discourse_type']

# Define class to integer mapping
CLASS_TO_INT_MAPPING = {
    'Lead': 0, 'Position': 1, 'Claim': 2, 'Evidence': 3, 'Counterclaim': 4,
    'Rebuttal': 5, 'Concluding Statement': 6
}


def load_base_submission_data():
    # Load base submission data from CSV
    base_submission_df = pd.read_csv(CSV_DIR / "interim_submission.csv")
    # Drop rows with NaN values
    base_submission_df.dropna(inplace=True)
    return base_submission_df


def identify_missing_essays(base_submission_df, test_ids):
    # Identify missing essays between base submission and test data
    essays_in_base_submission = set(base_submission_df['id'].unique())
    essays_in_test_data = set(test_ids)
    missing_essays = essays_in_test_data - essays_in_base_submission
    print(f"Missing essays: {missing_essays}")
    return missing_essays


def add_missing_essays(base_submission_df, missing_essays):
    # Add missing essays to base submission DataFrame
    new_base_submission_df = base_submission_df.copy()
    for essay_id in missing_essays:
        new_row = {
            'id': essay_id,
            'predictionstring': '2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18',
            'class': 'Claim'
        }
        new_base_submission_df = pd.concat([new_base_submission_df, pd.DataFrame([new_row])], ignore_index=True)
    return new_base_submission_df


def fix_submission_columns(df):
    # Fix column names and add discourse type and effectiveness scores
    fixed_df = df.rename(columns={'id': 'essay_id_comp'})
    fixed_df['discourse_type'] = fixed_df['class'].map(CLASS_TO_INT_MAPPING)
    fixed_df['score_discourse_effectiveness_0'] = 0.999999
    fixed_df['score_discourse_effectiveness_1'] = 1 - fixed_df['score_discourse_effectiveness_0']
    return fixed_df


def main():
    # Load test data files
    test_files = os.listdir(COMPETITION_DATA_DIR / 'test')
    test_ids = [f.replace('.txt', '') for f in test_files]

    # Load base submission data
    base_submission_df = load_base_submission_data()

    # Identify missing essays
    missing_essays = identify_missing_essays(base_submission_df, test_ids)

    # Add missing essays to base submission
    new_base_submission_df = add_missing_essays(base_submission_df, missing_essays)

    # Fix submission DataFrame
    interim_submission_df = fix_submission_columns(new_base_submission_df)

    # Save fixed interim submission DataFrame to CSV
    interim_submission_df.to_csv(CSV_DIR / 'fixed_interim_submission.csv', index=False)


if __name__ == "__main__":
    main()
