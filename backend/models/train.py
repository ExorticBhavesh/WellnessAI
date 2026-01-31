"""
Model Training Module for Health Prediction
Trains and evaluates ML models for disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from preprocess import HealthDataPreprocessor


class HealthModelTrainer:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.classes = None
        self.feature_importance = None
        
    def train_random_forest(self, X_train, y_train, n_estimators=100):
        """Train Random Forest model"""
        print("\nTraining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.model_name = "Random Forest"
        
        # Store feature importance
        if hasattr(X_train, 'columns'):
            self.feature_importance = dict(zip(
                X_train.columns, 
                self.model.feature_importances_
            ))
        
        print(f"✓ {self.model_name} training complete")
        return self.model
    
    def train_gradient_boosting(self, X_train, y_train, n_estimators=100):
        """Train Gradient Boosting model"""
        print("\nTraining Gradient Boosting model...")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.model_name = "Gradient Boosting"
        
        # Store feature importance
        if hasattr(X_train, 'columns'):
            self.feature_importance = dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        
        print(f"✓ {self.model_name} training complete")
        return self.model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        print("\nTraining Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )
        self.model.fit(X_train, y_train)
        self.model_name = "Logistic Regression"
        print(f"✓ {self.model_name} training complete")
        return self.model
    
    def evaluate_model(self, X_test, y_test, preprocessor=None):
        """Evaluate model performance"""
        if self.model is None:
            print("Error: No model trained yet")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print(f"Model Evaluation - {self.model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nClassification Report:")
        
        # Get class names if preprocessor available
        target_names = None
        if preprocessor and 'disease' in preprocessor.label_encoders:
            target_names = preprocessor.label_encoders['disease'].classes_
        
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            print("Error: No model trained yet")
            return None
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"\nCross-Validation Results ({cv}-fold):")
        print(f"Accuracy scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, filepath='model.pkl'):
        """Save trained model"""
        if self.model is None:
            print("Error: No model to save")
            return False
        
        model_package = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n✓ Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='model.pkl'):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_package = pickle.load(f)
            
            self.model = model_package['model']
            self.model_name = model_package.get('model_name', 'Unknown')
            self.feature_importance = model_package.get('feature_importance')
            
            print(f"✓ Model loaded from {filepath}")
            print(f"  Model type: {self.model_name}")
            return True
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return False
    
    def compare_models(self, X_train, y_train, X_test, y_test):
        """Compare different models"""
        models = {
            'Random Forest': lambda: self.train_random_forest(X_train, y_train),
            'Gradient Boosting': lambda: self.train_gradient_boosting(X_train, y_train),
            'Logistic Regression': lambda: self.train_logistic_regression(X_train, y_train)
        }
        
        results = {}
        
        for name, train_func in models.items():
            print(f"\n{'='*60}")
            print(f"Training {name}...")
            print(f"{'='*60}")
            
            train_func()
            metrics = self.evaluate_model(X_test, y_test)
            results[name] = metrics['accuracy']
        
        # Print comparison
        print(f"\n{'='*60}")
        print("Model Comparison Summary")
        print(f"{'='*60}")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:.<40} {acc:.4f} ({acc*100:.2f}%)")
        
        # Select best model
        best_model = max(results.items(), key=lambda x: x[1])[0]
        print(f"\n🏆 Best Model: {best_model}")
        
        return results


def main():
    """Main training pipeline"""
    print("="*60)
    print("Health Prediction Model Training Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = HealthDataPreprocessor()
    
    # Preprocess data
    print("\n[Step 1] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        filepath='../data/health_dataset.csv'
    )
    
    if X_train is None:
        print("Error: Data preprocessing failed")
        return
    
    # Save preprocessor
    preprocessor.save_preprocessor('preprocessor.pkl')
    
    # Initialize trainer
    trainer = HealthModelTrainer()
    
    # Train model (Random Forest by default)
    print("\n[Step 2] Training model...")
    trainer.train_random_forest(X_train, y_train, n_estimators=100)
    
    # Evaluate model
    print("\n[Step 3] Evaluating model...")
    trainer.evaluate_model(X_test, y_test, preprocessor)
    
    # Cross-validation
    print("\n[Step 4] Cross-validation...")
    trainer.cross_validate(X_train, y_train, cv=5)
    
    # Save model
    print("\n[Step 5] Saving model...")
    trainer.save_model('model.pkl')
    
    # Display feature importance
    if trainer.feature_importance:
        print("\n[Step 6] Feature Importance:")
        sorted_features = sorted(
            trainer.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"  {feature:.<40} {importance:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete! ✓")
    print("="*60)
    print(f"\nModel saved as: model.pkl")
    print(f"Preprocessor saved as: preprocessor.pkl")


if __name__ == "__main__":
    main()