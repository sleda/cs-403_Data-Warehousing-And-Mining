import pandas as pd

def combine_and_save_datasets(train_files, test_files, validation_files, output_prefix):
    #yükle
    categorical_train, numeric_train = pd.read_csv(train_files[0]), pd.read_csv(train_files[1])
    categorical_test, numeric_test = pd.read_csv(test_files[0]), pd.read_csv(test_files[1])
    categorical_validation, numeric_validation = pd.read_csv(validation_files[0]), pd.read_csv(validation_files[1])

    #birleştir
    train_combined = pd.merge(categorical_train, numeric_train, how='outer', left_on=['Target'], right_on=['Entity'])
    test_combined = pd.merge(categorical_test, numeric_test, how='outer', left_on=['Target'], right_on=['Entity'])
    validation_combined = pd.merge(categorical_validation, numeric_validation, how='outer', left_on=['Target'], right_on=['Entity'])

    #kaydet
    train_combined.to_csv(f'{output_prefix}_train_combined.csv', index=False)
    test_combined.to_csv(f'{output_prefix}_test_combined.csv', index=False)
    validation_combined.to_csv(f'{output_prefix}_validation_combined.csv', index=False)

train_files = ['./train/categorical_train.csv', './train/numeric_train.csv']
test_files = ['./test/categorical_test.csv', './test/numeric_test.csv']
validation_files = ['./validation/categorical_validation.csv', './validation/numeric_validation.csv']


combine_and_save_datasets(train_files, test_files, validation_files, 'processed')
