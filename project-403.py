from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from scipy import sparse
import pandas as pd
import numpy as np

def prepare_data(file_path, encoder, scaler):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    df[categorical_columns] = df[categorical_columns].astype(str)
    
    X_categorical = encoder.transform(df[categorical_columns])
    X_numeric = scaler.transform(df[['Numeric Value']])

    # Negatif değerler ayarlanmalıydı çünkü NB modeli negatif değerleri kabul etmiyor
    X_numeric_adjusted = X_numeric + abs(X_numeric.min()) + 1

    X = sparse.hstack((X_categorical, sparse.csr_matrix(X_numeric_adjusted)))
    y = df['Numeric Value'].astype(int)

    return X, y

# OneHotEncoder ve StandardScaler nesneleri
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
scaler = StandardScaler()

# Kategorik sütunlar
categorical_columns = ['Relation', 'Target', 'Relatum']

# Eğitim seti
df_train = pd.read_csv('./processed_train_combined.csv')
df_train.fillna(0, inplace=True)
df_train[categorical_columns] = df_train[categorical_columns].astype(str)
X_train_categorical = encoder.fit_transform(df_train[categorical_columns])
X_train_numeric = scaler.fit_transform(df_train[['Numeric Value']])

# Negatif değerleri ayarlayın
X_train_numeric += abs(X_train_numeric.min()) + 1

X_train = sparse.hstack((X_train_categorical, sparse.csr_matrix(X_train_numeric)))
y_train = df_train['Numeric Value'].astype(int)

# Test ve doğrulama setleri
X_test, y_test = prepare_data('./processed_test_combined.csv', encoder, scaler)
X_val, y_val = prepare_data('./processed_validation_combined.csv', encoder, scaler)

# Model değerlendirme fonksiyonu
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))

# Model değerlendirme
for k in [3, 7, 11]:
    evaluate_model(KNeighborsClassifier(n_neighbors=k), X_train, y_train, X_val, y_val, f"KNN with K={k}")

for layers in [(32,), (32, 32), (32, 32, 32)]:
    evaluate_model(MLPClassifier(hidden_layer_sizes=layers, max_iter=1000), X_train, y_train, X_val, y_val, f"MLP with layers {layers}")

evaluate_model(MultinomialNB(), X_train, y_train, X_val, y_val, "Naive Bayes")
