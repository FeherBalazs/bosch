import pandas as pd
import numpy as np
from IPython.display import display
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU

DATA_DIR = "../input"

ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'

NROWS = 10000

TRAIN_NUMERIC = "{0}/train_numeric.csv".format(DATA_DIR)
TRAIN_CATEGORICAL = "{0}/train_categorical.csv".format(DATA_DIR)
TRAIN_DATE = "{0}/train_date.csv".format(DATA_DIR)

TEST_NUMERIC = "{0}/test_numeric.csv".format(DATA_DIR)
TEST_CATEGORICAL = "{0}/test_categorical.csv".format(DATA_DIR)
TEST_DATE = "{0}/test_date.csv".format(DATA_DIR)

print('Loading test data...')
test = pd.read_csv(TEST_NUMERIC, usecols=[ID_COLUMN], nrows=NROWS)
print('Test data loaded.')

# Preprocessing categorical data

# Load data
print('Loading train_categorical data...')
train_df1 = pd.read_csv(TRAIN_CATEGORICAL, nrows=NROWS, low_memory=False)
dtypes_train_df1 = train_df1.dtypes
train_df1 = pd.read_csv(TRAIN_CATEGORICAL, dtype=dtypes_train_df1)
print('Train_categorical data loaded.')

print('Loading test_categorical data...')
test_df1 = pd.read_csv(TEST_CATEGORICAL, nrows=NROWS, low_memory=False)
dtypes_test_df1 = test_df1.dtypes
test_df1 = pd.read_csv(TRAIN_CATEGORICAL, dtype=dtypes_test_df1)
print('Test_categorical data loaded.')

# Merge into single DataFrame such that we get consistent dummies at train and test
print('Merging categorical data...')
merged = pd.concat([train_df1, test_df1])
print('Categorical data merged.')

# Preprocess
print('Preprocessing data merged...')
merged.fillna(0, inplace=True)
merged = pd.get_dummies(merged)
print('Preprocessing finished.')

# Load back
print('Loading back train data...')
train_df1 = merged[:train_df1.shape[0]]
print('Train data loaded back.')

print('Loading back test data...')
test_df1 = merged[train_df1.shape[0]:]
print('Test data loaded back.')

# Preprocessing remaining train data

min_max_scaler = preprocessing.MinMaxScaler()

print('Loading train_date data...')
train_df2 = pd.read_csv(TRAIN_DATE, nrows=NROWS, low_memory=False)
dtypes_train_df2 = train_df2.dtypes
train_df2 = pd.read_csv(TRAIN_DATE, dtype=dtypes_train_df2)
print('Train_date data loaded.')

print('Preprocessing train_date data...')
train_df2.fillna(0, inplace=True)
train_df2.ix[:,1:] = min_max_scaler.fit_transform(train_df2.ix[:,1:])
print('Preprocessing finished.')

print('Loading train_numeric data...')
train_df3 = pd.read_csv(TRAIN_NUMERIC, nrows=NROWS, low_memory=False)
dtypes_train_df3 = train_df3.dtypes
train_df3 = pd.read_csv(TRAIN_NUMERIC, dtype=dtypes_train_df3)
print('Train_numeric data loaded.')

print('Preprocessing train_numeric data...')
train_df3.fillna(0, inplace=True)
train_df3.ix[:,1:] = min_max_scaler.fit_transform(train_df3.ix[:,1:])
print('Preprocessing finished.')

# Merge DataFrames
print('Merging DataFrames...')
train_df = pd.merge(pd.merge(train_df1, train_df2, on='Id'), train_df3, on='Id')
print('DataFrames merged.')

print('Preprocessing merged DataFrame...')
# Drop Id as we don't need it anymore
train_df = train_df.drop("Id", 1)
# Fill NAs
train_df.fillna(0, inplace=True)
print('Preprocessing finished.')

# Preprocessing test data

print('Loading test_date data...')
test_df2 = pd.read_csv(TEST_DATE, nrows=NROWS, low_memory=False)
dtypes_test_df2 = test_df2.dtypes
test_df2 = pd.read_csv(TEST_DATE, dtype=dtypes_test_df2)
print('Test_date data loaded.')

print('Preprocessing test_date data...')
test_df2.fillna(0, inplace=True)
test_df2.ix[:,1:] = min_max_scaler.fit_transform(test_df2.ix[:,1:])
print('Preprocessing finished.')

print('Loading test_numeric data...')
test_df3 = pd.read_csv(TEST_NUMERIC, nrows=NROWS, low_memory=False)
dtypes_test_df3 = test_df3.dtypes
test_df3 = pd.read_csv(TEST_NUMERIC, dtype=dtypes_test_df3)
print('Test_numeric data loaded.')

print('Preprocessing test_date data...')
test_df3.fillna(0, inplace=True)
test_df3.ix[:,1:] = min_max_scaler.fit_transform(test_df3.ix[:,1:])
print('Preprocessing finished.')

# Merge DataFrames
print('Merging DataFrames...')
test_df = pd.merge(pd.merge(test_df1, test_df2, on='Id'), test_df3, on='Id')
print('DataFrames merged.')

# Getting Id column for creating the submission file at the end
id_test = test['Id'].values

print('Preprocessing merged DataFrame...')
# Drop Id as we don't need it anymore
test_df = test_df.drop("Id", 1)
# Fill NAs
test_df.fillna(0, inplace=True)
print('Preprocessing finished.')

####################################################################################################
####################################################################################################
####################################################################################################

# Extract feature (X) and target (y) columns
feature_cols = list(train_df.columns[:-1])  # all columns but last are features
target_col = train_df.columns[-1]  # last column is the target/label
X_all = train_df[feature_cols]  # feature values for all products
y_all = train_df[target_col]  # corresponding targets
print('Feature and target columns separated.')

# Select features (X) and corresponding labels (y) for the training and validation sets
# Note: Shuffle the data or randomly select samples to avoid any bias due to ordering in the dataset

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size = 0.85, random_state=42)

# Convert data to Numpy matrices for Keras
X_train = np.asmatrix(X_train)
X_test = np.asmatrix(X_test)
test_df = np.asmatrix(test_df)

print "Training set: {} samples".format(X_train.shape[0])
print "Validation set: {} samples".format(X_test.shape[0])
# Note: If you need a validation set, extract it from within training data

####################################################################################################
####################################################################################################
####################################################################################################

def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1], init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.4))
    model.add(Dense(128, init = 'he_normal'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, init = 'he_normal'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'Adam')
    return(model)

nb_epoch = 1
batch_size = 256

model = build_model()
fit = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch = nb_epoch, verbose = 2 ,validation_data=[X_test, y_test])
pred = model.predict_classes(test_df, batch_size=batch_size)
    
print('Exporting final result...')
df = pd.DataFrame({'Id': id_test})
df['Response'] = pred
df.to_csv('../output/submission_keras.csv', index = False)

print('Results ready for submission.')

