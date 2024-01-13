import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from scipy.sparse import hstack

def prepare_data(categorical_file, numeric_file, encoder=None, scaler=None):
    categorical_df = pd.read_csv(categorical_file)
    numeric_df = pd.read_csv(numeric_file)
    combined_df = pd.merge(categorical_df, numeric_df, left_on='Target', right_on='Entity', how='inner')

    categorical_columns = ['Relation', 'Target', 'Relatum']
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        X_categorical = encoder.fit_transform(combined_df[categorical_columns])
    else:
        X_categorical = encoder.transform(combined_df[categorical_columns])

    if scaler is None:
        scaler = StandardScaler()
        scaled_numeric_values = scaler.fit_transform(combined_df[['Numeric Value']])
    else:
        scaled_numeric_values = scaler.transform(combined_df[['Numeric Value']])

    X = hstack([X_categorical, scaled_numeric_values])

    bins = [0, 0.33, 0.66, 1.0]
    labels = [0, 1, 2]
    combined_df['Numeric Value'] = pd.cut(combined_df['Numeric Value'], bins=bins, labels=labels)
    y = combined_df['Numeric Value'].astype(int)

    return X, y, encoder, scaler


X_train, y_train, encoder, scaler = prepare_data('./train/categorical_train.csv', './train/numeric_train.csv')


X_test, y_test, _, _ = prepare_data('./test/categorical_test.csv', './test/numeric_test.csv', encoder, scaler)
X_val, y_val, _, _ = prepare_data('./validation/categorical_validation.csv', './validation/numeric_validation.csv', encoder, scaler)


print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")
print(f"Doğrulama seti boyutu: {X_val.shape}")

def save_processed_data(X, y, file_path):

    X_dense = X.toarray() if hasattr(X, "toarray") else X


    df = pd.DataFrame(X_dense)


    df['label'] = y


    df.to_csv(file_path, index=False)

save_processed_data(X_train, y_train, './train/processed_train.csv')
save_processed_data(X_test, y_test, './test/processed_test.csv')
save_processed_data(X_val, y_val, './validation/processed_validation.csv')
