import pandas as pd
import numpy as np
import keras
import re
import tensorflow as tf
import random
import os

from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from keras import layers


# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  


# Load Tarin Data
train=pd.read_csv('./datasets/credit_score/train.csv')
# Load Test Data
test=pd.read_csv('./datasets/credit_score/test.csv')

# Combine the training and test datasets into a single DataFrame
data=pd.concat([train,test],ignore_index=True)


#data cleaning

# A function for removing symboles in Data
def text_cleaning(data):
  """
    Cleans a given text data by:
    - Returning the data as-is if it is NaN or not a string.
    - Otherwise, converting the data to a string and removing unwanted characters
      such as leading/trailing underscores ('_'), spaces (' '), commas (','), and double quotes ('"').

    Args:
        data: The input data to clean, which can be of any type.

    Returns:
        The cleaned string if the input is a valid string, or the original data if it is NaN or not a string.
    """

  if data is np.nan or not isinstance(data, str): #  Check if data is NaN or not a string.
        return data #  Return the data without modification.
  else:
        return str(data).strip('_ ,"')  # Convert to string and remove specific leading/trailing characters.
  
# Apply the text_cleaning function to every element in the DataFrame
# and replace invalid values with NaN.
data = data.applymap(text_cleaning).replace(['', 'nan', '!@9#%8', '#F%$D@*&8'], np.nan)



# Ensure proper formatting of numerical columns by converting them to their intended data types

data['Age']                     = data.Age.astype(int)                          # Convert to integer for age
data['Num_of_Loan']             = data.Num_of_Loan.astype(int)                  # Convert to integer for loan count
data['Num_Bank_Accounts']       = data.Num_Bank_Accounts.astype(int)            # Convert to integer for account count
data['Annual_Income']           = data.Annual_Income.astype(float)              # Convert to float for precise income values
data['Changed_Credit_Limit']    = data.Changed_Credit_Limit.astype(float)       # Convert to float for credit limit changes
data['Outstanding_Debt']        = data.Outstanding_Debt.astype(float)           # Convert to float for outstanding debt
data['Amount_invested_monthly'] = data.Amount_invested_monthly.astype(float)    # Convert to float for monthly investments
data['Monthly_Balance']         = data.Monthly_Balance.astype(float)            # Convert to float for monthly balances
data['Num_of_Delayed_Payment'] = pd.to_numeric(data['Num_of_Delayed_Payment'],  # Convert to numeric for delayed payment count
                                               errors='coerce')



# Define the mapping for loan types
mapping_priority = [
    'not specified',
    'credit-builder loan',
    'personal loan',
    'consolidation loan',
    'student loan'
]

# Function to select a single value from multi-value loan types
def map_loan_types(loan_types):
    if pd.isna(loan_types):
        return loan_types  # Return NaN as is
    # Split by commas and convert to lowercase
    loans = re.split(r',| and ', loan_types.lower())
    # Find the first matching value based on mapping priority
    for priority in mapping_priority:
        if any(priority in loan.strip() for loan in loans):
            return priority
    return 'other'  # Fallback value if no mapping is found

# Apply the mapping function to the original column
data['Type_of_Loan'] = data['Type_of_Loan'].apply(map_loan_types)


# As Customer_ID is not null, we can use it as a unique identifier for our data.
# This ensures that each Customer in the dataset can be uniquely identified by the Customer_ID.
# Setting Customer_ID as the index
data = data.set_index('Customer_ID')


# Group by 'Customer_ID' and apply ffill
# Exclude 'Credit_History_Age' from forward-filling.
exclude_column = 'Credit_History_Age'

# Separate the excluded column for later use.
excluded_data = data[exclude_column]

# Drop the excluded column and reset the index
data = data.drop(columns=[exclude_column]).reset_index()

# Apply forward-fill to the remaining columns grouped by 'Customer_ID'.
# This fills NaN values with the last valid observation for each customer.
data = data.groupby('Customer_ID').apply(lambda group: group.fillna(method='ffill')).reset_index(drop=True)

# Add the excluded column back
excluded_data = excluded_data.reset_index(drop=True)
data[exclude_column] = excluded_data

# Optional: Set the index back to the original
data.set_index('Customer_ID', inplace=True)


# Function to convert Credit_History_Age to a start date in YYYY-MM-DD format
def convert_to_start_date(age_str):
   # Check if the input is NaN; if so, return it unchanged
    if pd.isna(age_str):
        return age_str

  # Split the string into parts based on the word 'and' to separate years and months
    parts = age_str.split('and')
    years = months = 0        # Initialize years and months to 0


  # Iterate through each part to extract years and months
    for part in parts:
        part = part.strip()      # Remove leading and trailing whitespace
        if 'Year' in part:
            years = int(part.split()[0])       # Extract the number of years
        elif 'Month' in part:
            months = int(part.split()[0])      # Extract the number of months

    # Calculate total months
    total_months = years * 12 + months
    # Calculate the start date
    start_date = datetime.now() - pd.DateOffset(months=total_months)
    # Format the date to YYYY-MM-DD and return it
    return start_date.strftime('%Y-%m-%d')


# Apply the function to create a new start date column
data['Credit_History_Age']=data['Credit_History_Age'].apply(convert_to_start_date)



#  Convert 'Credit_History_Age' to datetime format, keeping NaN values as NaT (Not a Time)
data['Credit_History_Age'] = pd.to_datetime(data['Credit_History_Age'], errors='coerce')

# Convert datetime to ordinal values for interpolation
data['Ordinal'] = data['Credit_History_Age'].apply(lambda x: x.toordinal() if pd.notnull(x) else None)

# Perform linear interpolation on ordinal values to fill NaN entries
data['Ordinal'] = data['Ordinal'].interpolate(method='linear')

# Convert interpolated ordinal values back to datetime format; NaN entries remain as NaT
data['Credit_History_Age'] = data['Ordinal'].apply(lambda x: datetime.fromordinal(int(x)) if pd.notnull(x) else pd.NaT)

# Drop the helper column used for interpolation
data.drop(columns=['Ordinal'], inplace=True)

# Group by 'Customer_ID' and apply bfill to 'Occupation' column.
data['Occupation'] = data.groupby(data.index)['Occupation'].transform(lambda group: group.fillna(method='bfill'))

"""
    Fill missing values in a specific column using KNN imputation.

    Parameters:
    data (DataFrame): The DataFrame containing the column to be imputed.
    column_name (str): The name of the column to be imputed.
    n_neighbors (int): Number of neighbors to consider for imputation. Default is 11.

    Returns:
    DataFrame: The DataFrame with missing values in the specified column imputed.
    """
def knn_impute(data, column_name, n_neighbors=5):
    # Extract the column to be imputed
    column_to_impute = data[[column_name]]

    # Create an instance of the KNNImputer class
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Fit and transform the imputer to fill missing values in the column
    column_imputed = imputer.fit_transform(column_to_impute)

    # Replace the original column with the imputed values
    data[column_name] = column_imputed

    return data


# Apply KNN imputation on specific columns to fill missing values
data = knn_impute(data, 'Monthly_Balance')
data = knn_impute(data, 'Monthly_Inhand_Salary')
data = knn_impute(data, 'Num_of_Delayed_Payment')
data = knn_impute(data, 'Num_Credit_Inquiries')
data = knn_impute(data, 'Changed_Credit_Limit')
data = knn_impute(data,'Amount_invested_monthly')

# Fill missing values in 'Payment_Behaviour' within each Customer_ID group
data['Payment_Behaviour'] = (
    data.groupby('Customer_ID')['Payment_Behaviour']  # Group the data by 'Customer_ID'
    .transform(lambda group: group.fillna(group.mode()[0] if not group.mode().empty else 'Unknown'))
    # Fill NaN values with the mode of the group (most frequent value)
)

# Fill categorical columns
# Forward filling is useful in this context as
# customer's 'credit mix' doesn't change frequently
data['Credit_Mix'].fillna(method='ffill', inplace=True)

# customer's 'Type_of_Loan mix' doesn't change frequently
data['Type_of_Loan'].fillna(method='ffill', inplace=True)


# Change format of some float , object values that supposed to be Integers
data['Num_of_Delayed_Payment'] = data['Num_of_Delayed_Payment'].astype(int)
data['Num_Credit_Inquiries'] = data['Num_Credit_Inquiries'].astype(int)

# List of selected columns to check for negative values
selected_columns = [
    'Delay_from_due_date',
    'Changed_Credit_Limit',
    'Num_Bank_Accounts',
    'Num_of_Loan',
    'Num_of_Delayed_Payment',
    'Monthly_Balance',
    'Age'
]

# Iterate over each column in the selected columns
for column in selected_columns:
    # Replace negative values with 0 in the current column
    data[column] = data[column].apply(lambda x: max(x, 0))

def cap_outliers_iqr(data, columns=None):
    """
    Caps outliers in the specified columns of a DataFrame using the IQR method,
    directly modifying the original columns, and visualizes the changes.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list, optional): List of column names to apply outlier capping.
                                   If None, all numerical columns are used.

    Returns:
        pd.DataFrame: DataFrame with capped outlier columns.
    """

    # If no specific columns are provided, select numerical columns
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Create a copy of the original DataFrame to avoid modifying it directly
    data_capped = data.copy()

    for column in columns:
        # Calculate Q1, Q3, and IQR for the current column
        Q1 = data_capped[column].quantile(0.25)
        Q3 = data_capped[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap outliers directly in the original column
        data_capped[column] = np.where(data_capped[column] > upper_bound, upper_bound, data_capped[column])
        data_capped[column] = np.where(data_capped[column] < lower_bound, lower_bound, data_capped[column])



    return data_capped

data= cap_outliers_iqr(data, columns=['Age'])
data=cap_outliers_iqr(data, columns=['Monthly_Inhand_Salary'])
data=cap_outliers_iqr(data, columns=['Interest_Rate'])
data=cap_outliers_iqr(data, columns=['Num_Bank_Accounts'])
data=cap_outliers_iqr(data, columns=['Num_Credit_Card'])
data=cap_outliers_iqr(data, columns=['Num_of_Loan'])
data=cap_outliers_iqr(data, columns=['Num_of_Delayed_Payment'])
data=cap_outliers_iqr(data, columns=['Num_Credit_Inquiries'])
data=cap_outliers_iqr(data, columns=['Outstanding_Debt'])
data=cap_outliers_iqr(data, columns=['Total_EMI_per_month'])

# Drop columns that are not useful for trainig model
# The 'Annual_Income' feature may not be useful for our analysis
# because it is correlated with 'Monthly_Inhand_Salary
columns_to_drop = ['ID', 'Month', 'Name', 'SSN','Credit_History_Age','Annual_Income']

data.drop(columns=columns_to_drop, inplace=True)


# Define a mapping for target encoding
status_mapping = {
    'Poor':0,
   'Standard':1,
   'Good':2
}

# Map STATUS to target categories
data['Credit_Score'] = data['Credit_Score'].map(status_mapping)

#Label Encoding
LE=LabelEncoder() # Initialize Label Encoder

# List of columns to encode
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

# Apply Label Encoding to each specified column
for column in categorical_cols:
    data[column] = LE.fit_transform(data[column])

# Separate features and target variable
X_selection= data.drop('Credit_Score', axis=1)
y = data['Credit_Score']



# Initialize SMOTE
smote = SMOTE(random_state=SEED)

# Apply SMOTE to the data
X_resampled, y_resampled = smote.fit_resample(X_selection, y)


# Label Onehot-encoding 
y_Onehot = to_categorical(y_resampled)


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_Onehot, test_size=0.25,stratify=y_Onehot, random_state=SEED)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

np.save('./datasets/credit_score/X_test_cs.npy', X_test)
np.save('./datasets/credit_score/y_test_cs.npy', y_test)

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train.shape[1]]),
    layers.Dense(2048,activation='relu'),
    layers.Dropout(0.5,seed=SEED),
    layers.Dense(1024,activation='relu'),
    layers.Dropout(0.5,seed=SEED),
    layers.Dense(512,activation='relu'),
    layers.Dropout(0.5,seed=SEED),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.5,seed=SEED),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5,seed=SEED),
    layers.Dense(3,activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=256,
    epochs=200,
    callbacks=[early_stopping],
)
# Evaluate the model on the test set
accuracy = model.evaluate(X_test, y_test)
#print('Accuracy: {}'.format(accuracy))

model.save('./datasets/credit_score/cs.keras')