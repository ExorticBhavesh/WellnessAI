"""
Data Preprocessing Module for Health Prediction Model
Handles data cleaning, feature engineering, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

class HealthDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, filepath='../data/health_dataset.csv'):
        """Load the health dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
    
    def clean_data(self, df):
        """Clean and handle missing values"""
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Fill categorical missing values with mode
        for col in categorical_columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        print(f"Data cleaned. Remaining nulls: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def encode_categorical(self, df, categorical_columns=None):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        if categorical_columns is None:
            categorical_columns = df_encoded.select_dtypes(include=['object']).columns
            # Exclude target if present
            categorical_columns = [col for col in categorical_columns if col != 'disease']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """Scale numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def prepare_features(self, df, target_column='disease'):
        """Prepare features and target for training"""
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
        else:
            X = df
            y = None
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def preprocess_pipeline(self, filepath='../data/health_dataset.csv', 
                           target_column='disease', test_size=0.2):
        """Complete preprocessing pipeline"""
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None, None, None, None
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode categorical variables (excluding target)
        df_encoded = self.encode_categorical(df_clean)
        
        # Prepare features and target
        X, y = self.prepare_features(df_encoded, target_column)
        
        # Encode target variable if it exists
        if y is not None and y.dtype == 'object':
            if target_column not in self.label_encoders:
                self.label_encoders[target_column] = LabelEncoder()
                y = self.label_encoders[target_column].fit_transform(y)
            else:
                y = self.label_encoders[target_column].transform(y)
        
        # Scale features
        X_scaled = self.scale_features(X, fit=True)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data if target exists
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            return X_scaled, None, None, None
    
    def save_preprocessor(self, filepath='preprocessor.pkl'):
        """Save the preprocessor state"""
        preprocessor_state = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_state, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.pkl'):
        """Load the preprocessor state"""
        try:
            with open(filepath, 'rb') as f:
                preprocessor_state = pickle.load(f)
            self.scaler = preprocessor_state['scaler']
            self.label_encoders = preprocessor_state['label_encoders']
            self.feature_columns = preprocessor_state['feature_columns']
            print(f"Preprocessor loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Error: Preprocessor file not found at {filepath}")
            return False
    
    def preprocess_input(self, input_data):
        """Preprocess new input data for prediction"""
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Encode categorical variables
        for col in input_df.columns:
            if col in self.label_encoders:
                input_df[col] = self.label_encoders[col].transform(input_df[col].astype(str))
        
        # Ensure correct feature order
        if self.feature_columns:
            input_df = input_df[self.feature_columns]
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        return input_scaled


if __name__ == "__main__":
    # Example usage
    preprocessor = HealthDataPreprocessor()
    
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    
    if X_train is not None:
        print("\nPreprocessing complete!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Feature columns: {preprocessor.feature_columns}")
        
        # Save preprocessor
        preprocessor.save_preprocessor('preprocessor.pkl')