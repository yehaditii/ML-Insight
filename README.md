# ML-Insight
Hands-on machine learning project featuring classification, regression, and neural network models
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision training for better performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load and preprocess data
def load_and_preprocess_data():
    try:
        train_df = pd.read_csv('/kaggle/input/praxis-24-ml-hackathon-hosted-by-gdgc-dseu/train.csv')
        test_df = pd.read_csv('/kaggle/input/praxis-24-ml-hackathon-hosted-by-gdgc-dseu/test.csv')
        print("Data loaded successfully!")
        
        print("\nTraining data shape:", train_df.shape)
        print("Test data shape:", test_df.shape)
        
        return train_df, test_df
    
    except FileNotFoundError:
        print("Error: Please ensure train.csv and test.csv are in the correct directory")
        return None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def preprocess_features(train_df, test_df):
    train_cleaned = train_df.copy()
    test_cleaned = test_df.copy()
    
    # Initialize encoders
    le = LabelEncoder()
    scaler = StandardScaler()
    knn_imputer = KNNImputer(n_neighbors=5)  # Using KNN imputation
    
    # Identify column types
    categorical_columns = train_cleaned.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in ['ID', 'Segmentation', 'Description']]
    
    numeric_columns = train_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['ID', 'Segmentation']]
    
    print("\nProcessing features:")
    print("Categorical:", categorical_columns)
    print("Numerical:", numeric_columns)
    
    # Process categorical features
    for col in categorical_columns:
        combined_data = pd.concat([train_cleaned[col], test_cleaned[col]]).astype(str)
        le_fitted = le.fit(combined_data)
        train_cleaned[col] = le_fitted.transform(train_cleaned[col].astype(str))
        test_cleaned[col] = le_fitted.transform(test_cleaned[col].astype(str))
    
    # Process numerical features with KNN imputation
    for col in numeric_columns:
        train_cleaned[col] = pd.to_numeric(train_cleaned[col], errors='coerce')
        test_cleaned[col] = pd.to_numeric(test_cleaned[col], errors='coerce')
    
    # Impute missing values using KNN
    train_cleaned[numeric_columns] = knn_imputer.fit_transform(train_cleaned[numeric_columns])
    test_cleaned[numeric_columns] = knn_imputer.transform(test_cleaned[numeric_columns])
    
    # Scale numerical features
    if numeric_columns:
        scaler_fitted = scaler.fit(train_cleaned[numeric_columns])
        train_cleaned[numeric_columns] = scaler_fitted.transform(train_cleaned[numeric_columns])
        test_cleaned[numeric_columns] = scaler_fitted.transform(test_cleaned[numeric_columns])
    
    feature_columns = categorical_columns + numeric_columns
    X_train = train_cleaned[feature_columns].values
    y_train = train_cleaned['Segmentation'].values
    X_test = test_cleaned[feature_columns].values
    
    return X_train, y_train, X_test, test_df['ID'], feature_columns

# Create neural network model
def create_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        
        # First dense layer with LeakyReLU and Dropout
        tf.keras.layers.Dense(512, kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.4),
        
        # Second dense layer with LeakyReLU and Dropout
        tf.keras.layers.Dense(256, kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        # Third dense layer with LeakyReLU and Dropout
        tf.keras.layers.Dense(128, kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Training configuration
def configure_training():
    return {
        'batch_size': 256,  # Increased batch size for better convergence
        'epochs': 1000,
        'learning_rate': 0.001,
        'early_stopping_patience': 50,  # Reduced patience for faster convergence
        'reduce_lr_patience': 5,
        'proto_output_ver': 8
    }

# Main training loop
def train_model():
    
    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data()
    if train_df is None or test_df is None:
        return
    
    X_train, y_train, X_test, test_ids, feature_columns = preprocess_features(train_df, test_df)
    
    # Convert target to categorical
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    num_classes = len(le.classes_)
    y_train_cat = tf.keras.utils.to_categorical(y_train_encoded)
    
    # Split data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train_cat, test_size=0.05, random_state=42
    )
    
    # Create and compile model
    model = create_model(X_train.shape[1], num_classes)
    config = configure_training()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['reduce_lr_patience'],
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\nTraining neural network...")
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions_prob = model.predict(X_test)
    predictions = le.inverse_transform(np.argmax(predictions_prob, axis=1))
    
    # Create submission file
    submission = pd.DataFrame({
        'ID': test_ids,
        'Segmentation': predictions
    })
    
    submission.to_csv('/kaggle/working/manualtuning{}.csv'.format(config['proto_output_ver']), index=False) 
    print("\nSubmission file 'manualtuning{}.csv' generated successfully!".format(config['proto_output_ver']))
    
    # Display sample predictions
    print("\nFirst few predictions:")
    print(submission.head())
    
    # Display feature importance using gradient-based method
    print("\nCalculating feature importance...")
    importance = calculate_feature_importance(model, X_train_split, feature_columns)
    print("\nTop 5 Most Important Features:")
    print(importance.head())

def calculate_feature_importance(model, X_train, feature_columns):
    with tf.GradientTape() as tape:
        X_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        mean_pred = tf.reduce_mean(predictions)
    
    gradients = tape.gradient(mean_pred, X_tensor)
    importance_scores = np.abs(gradients.numpy()).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    return importance_df

if __name__ == "__main__":
    train_model()
