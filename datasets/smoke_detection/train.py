import random
import numpy as np
import tensorflow as tf
import os
import pandas as pd



# Set random seed for Python, NumPy, and TensorFlow
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  



# Đọc file CSV từ thư mục trên Kaggle
df = pd.read_csv('./datasets/smoke_detection/smoke_detection_iot.csv')

df.drop(['Unnamed: 0', 'CNT', 'UTC'], axis=1, inplace=True, errors='ignore')


num_duplicates = df.duplicated().sum()
num_total = len(df)

# Tạo dữ liệu cho biểu đồ
labels = ['Unique Rows', 'Duplicate Rows']
sizes = [num_total - num_duplicates, num_duplicates]
colors = ['lightblue', 'lightcoral']
explode = (0.1, 0)  # Nổi bật phần trùng lặp

myexplode = [0.2, 0]
df['Fire Alarm'].value_counts().plot(kind ='pie',autopct = '%.2f',explode = myexplode)

# Tính toán IQR
Q1 = df['eCO2[ppm]'].quantile(0.25)
Q3 = df['eCO2[ppm]'].quantile(0.75)
IQR = Q3 - Q1

# Xác định ngưỡng để phát hiện outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Phát hiện outliers
outliers = df[(df['eCO2[ppm]'] < lower_bound) | (df['eCO2[ppm]'] > upper_bound)]

# Xóa outliers từ DataFrame
df_no_outliers = df[(df['eCO2[ppm]'] >= lower_bound) & (df['eCO2[ppm]'] <= upper_bound)]

# Tính toán Q1, Q3 và IQR cho Raw H2
Q1 = df_no_outliers['Raw H2'].quantile(0.25)
Q3 = df_no_outliers['Raw H2'].quantile(0.75)
IQR = Q3 - Q1

# Xác định giới hạn cho các giá trị ngoại lai
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cập nhật DataFrame df_no_outliers bằng cách loại bỏ các giá trị ngoại lai
df_no_outliers = df_no_outliers[(df_no_outliers['Raw H2'] >= lower_bound) & (df_no_outliers['Raw H2'] <= upper_bound)]


# Chuyển đổi các giá trị vô hạn thành NaN
df_no_outliers.replace([np.inf, -np.inf], np.nan, inplace=True)

# Loại bỏ các hàng có giá trị NaN (nếu cần)
df_no_outliers.dropna(subset=['PM2.5'], inplace=True)

import pandas as pd

# Giả sử df_no_outliers là DataFrame của bạn
# Tính toán Q1 và Q3
Q1 = df_no_outliers['PM2.5'].quantile(0.25)
Q3 = df_no_outliers['PM2.5'].quantile(0.75)

# Tính IQR
IQR = Q3 - Q1

# Xác định ngưỡng dưới và ngưỡng trên
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Lọc dữ liệu để loại bỏ các giá trị ngoại lai
df_test = df_no_outliers[(df_no_outliers['PM2.5'] >= lower_bound) & (df_no_outliers['PM2.5'] <= upper_bound)]


df_no_outliers = df_test.copy()

# Chuyển đổi các giá trị vô hạn thành NaN
df_no_outliers.replace([np.inf, -np.inf], np.nan, inplace=True)

# Loại bỏ các hàng có giá trị NaN (nếu cần)
df_no_outliers.dropna(subset=['NC0.5'], inplace=True)

# Chuyển đổi các giá trị vô hạn thành NaN
df_no_outliers.replace([np.inf, -np.inf], np.nan, inplace=True)

# Loại bỏ các hàng có giá trị NaN (nếu cần)
df_no_outliers.dropna(subset=['NC1.0'], inplace=True)

# Chuyển đổi các giá trị vô hạn thành NaN
df_no_outliers.replace([np.inf, -np.inf], np.nan, inplace=True)

# Loại bỏ các hàng có giá trị NaN (nếu cần)
df_no_outliers.dropna(subset=['NC2.5'], inplace=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Giả sử df_no_outliers là DataFrame đã chuẩn bị
# và cột 'Fire Alarm' là nhãn (target)

# Chia dữ liệu thành đặc trưng (features) và nhãn (target)
X = df_no_outliers.drop(columns=['Fire Alarm'])
y = df_no_outliers['Fire Alarm']

from tensorflow.keras.utils import to_categorical
# Label Onehot-encoding 
y_Onehot = to_categorical(y)


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y_Onehot, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import numpy as np
from collections import Counter
def analyze_class_imbalance(labels):
    """
    Analyze class imbalance for binary or multiclass classification.
    
    Args:
        labels (list or array-like): The list of class labels.
        
    Returns:
        dict: Contains class counts and max-to-min ratio.
    """
    counts = Counter(labels)
    class_counts = dict(counts)
    
    max_count = max(counts.values())
    min_count = min(counts.values())
    
    max_to_min_ratio = max_count / min_count if min_count > 0 else np.inf
    
    results = {
        'class_counts': class_counts,
        'max_to_min_ratio': max_to_min_ratio
    }
    
    return results

import keras
from keras import layers

SEED = 42

model = keras.Sequential([
    layers.BatchNormalization(input_shape=[X_train_scaled.shape[1]]),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(128,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(64,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(32,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(16,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5, seed = SEED),
    layers.Dense(2,activation='softmax')
])

model.build(input_shape=(None,X_train_scaled.shape[1]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    batch_size=128,
    epochs=200,
    callbacks=[early_stopping],
)

np.save('./datasets/smoke_detection/X_test_sd.npy', X_test_scaled)
np.save('./datasets/smoke_detection/y_test_sd.npy', y_test)

model.save('./datasets/smoke_detection/sd.keras')