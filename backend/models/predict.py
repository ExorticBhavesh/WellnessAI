"""
Prediction Module for Health ML Model
Makes predictions on new health data
"""

from typing import TYPE_CHECKING

pd = None
if TYPE_CHECKING:
    # Allow static type checkers to see pandas types without requiring it at runtime
    import pandas as pd  # type: ignore
else:
    try:
        import pandas as pd
    except ImportError:
        pd = None
import pickle
import warnings
warnings.filterwarnings('ignore')

from preprocess import HealthDataPreprocessor


class HealthPredictor:
    def __init__(self, model_path='model.pkl', preprocessor_path='preprocessor.pkl'):
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.is_loaded = False
        
    def load_model(self):
        """Load trained model and preprocessor"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            self.model = model_package['model']
            
            # Load preprocessor
            self.preprocessor = HealthDataPreprocessor()
            self.preprocessor.load_preprocessor(self.preprocessor_path)
            
            self.is_loaded = True
            print("✓ Model and preprocessor loaded successfully")
            return True
            
        except FileNotFoundError as e:
            print(f"Error: Required files not found - {e}")
            print("Please run train.py first to generate model files")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single(self, input_data):
        """
        Predict disease for a single patient
        
        Args:
            input_data (dict): Patient health data
            
        Returns:
            dict: Prediction results with confidence
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        try:
            # Preprocess input
            input_processed = self.preprocessor.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(input_processed)[0]
            probabilities = self.model.predict_proba(input_processed)[0]
            
            # Get disease name if label encoder exists
            if 'disease' in self.preprocessor.label_encoders:
                disease_name = self.preprocessor.label_encoders['disease'].inverse_transform([prediction])[0]
            else:
                disease_name = str(prediction)
            
            # Get top 3 predictions
            # Get top 3 predictions
            top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
            top_predictions = []
            for idx in top_indices:
                if 'disease' in self.preprocessor.label_encoders:
                    disease = self.preprocessor.label_encoders['disease'].inverse_transform([idx])[0]
                else:
                    disease = str(idx)
                
                top_predictions.append({
                    'disease': disease,
                    'probability': float(probabilities[idx]),
                    'confidence': f"{probabilities[idx]*100:.2f}%"
                })
            
            result = {
                'predicted_disease': disease_name,
                'confidence': float(probabilities[prediction]),
                'confidence_percentage': f"{probabilities[prediction]*100:.2f}%",
                'top_predictions': top_predictions
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_batch(self, input_data_list):
        """
        Predict diseases for multiple patients
        
        Args:
            input_data_list (list): List of patient data dictionaries
            
        Returns:
            list: List of prediction results
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        results = []
        for i, data in enumerate(input_data_list):
            print(f"Processing patient {i+1}/{len(input_data_list)}...")
            result = self.predict_single(data)
            results.append(result)
    def predict_from_csv(self, filepath, output_filepath=None):
        """
        Make predictions from CSV file
        
        Args:
            filepath (str): Path to input CSV
            output_filepath (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame or list: DataFrame with predictions if pandas is available, otherwise a list of dicts
        """
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        try:
            # If pandas is available use it, otherwise fall back to csv module
            if pd is not None:
                # Load data with pandas
                df = pd.read_csv(filepath)
                print(f"Loaded {len(df)} records from {filepath}")
                
                # Make predictions
                predictions = []
                confidences = []
                
                for idx, row in df.iterrows():
                    input_data = row.to_dict()
                    result = self.predict_single(input_data)
                    
                    if result:
                        predictions.append(result['predicted_disease'])
                        confidences.append(result['confidence'])
                    else:
                        predictions.append('Error')
                        confidences.append(0.0)
                
                # Add predictions to dataframe
                df['predicted_disease'] = predictions
                df['confidence'] = confidences
                df['confidence_percentage'] = [f"{c*100:.2f}%" for c in confidences]
                
                # Save if output path provided
                if output_filepath:
                    df.to_csv(output_filepath, index=False)
                    print(f"Results saved to {output_filepath}")
                
                return df
            else:
                # Fallback using csv module if pandas is not installed
                import csv
                
                with open(filepath, newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = [row for row in reader]
                
                print(f"Loaded {len(rows)} records from {filepath} (csv fallback)")
                
                predictions = []
                confidences = []
                
                for row in rows:
                    # csv returns strings; preprocess should handle conversions where necessary
                    result = self.predict_single(row)
                    
                    if result:
                        predictions.append(result['predicted_disease'])
                        confidences.append(result['confidence'])
                    else:
                        predictions.append('Error')
                        confidences.append(0.0)
                
                # Combine original rows with prediction results
                results = []
                for orig, pred, conf in zip(rows, predictions, confidences):
                    entry = orig.copy()
                    entry['predicted_disease'] = pred
                    entry['confidence'] = conf
                    entry['confidence_percentage'] = f"{conf*100:.2f}%"
                    results.append(entry)
                
                # Save if output path provided
                if output_filepath:
                    # determine header (union of all keys)
                    headers = set()
                    for r in results:
                        headers.update(r.keys())
                    headers = list(headers)
                    with open(output_filepath, 'w', newline='', encoding='utf-8') as outcsv:
                        writer = csv.DictWriter(outcsv, fieldnames=headers)
                        writer.writeheader()
                        for r in results:
                            writer.writerow(r)
                    print(f"Results saved to {output_filepath} (csv fallback)")
                
                return results
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None
    
    def explain_prediction(self, input_data):
        """
        Provide explanation for prediction
        
        Args:
            input_data (dict): Patient health data
            
        Returns:
            dict: Prediction with explanation
        """
        result = self.predict_single(input_data)
        
        if result is None:
            return None
        
        explanation = {
            'prediction': result,
            'input_features': input_data,
            'interpretation': self._generate_interpretation(result, input_data)
        }
        
        return explanation
    
    def _generate_interpretation(self, result, input_data):
        """Generate human-readable interpretation"""
        disease = result['predicted_disease']
        confidence = result['confidence_percentage']
        
        interpretation = f"Based on the provided health indicators, the model predicts '{disease}' "
        interpretation += f"with {confidence} confidence.\n\n"
        
        if result['confidence'] > 0.8:
            interpretation += "This is a high-confidence prediction."
        elif result['confidence'] > 0.6:
            interpretation += "This is a moderate-confidence prediction."
        else:
            interpretation += "This prediction has lower confidence. Consider consulting a healthcare professional."
        
        interpretation += f"\n\nAlternative possibilities:\n"
        for pred in result['top_predictions'][1:]:
            interpretation += f"- {pred['disease']}: {pred['confidence']}\n"
        
        return interpretation


def demo_prediction():
    """Demo function showing how to use the predictor"""
    print("="*60)
    print("Health Prediction Demo")
    print("="*60)
    
    # Initialize predictor
    predictor = HealthPredictor()
    
    # Example patient data (adjust based on your actual features)
    sample_patient = {
        'age': 45,
        'gender': 'Male',
        'blood_pressure': 140,
        'cholesterol': 240,
        'glucose': 110,
        'bmi': 28.5,
        'smoking': 'Yes',
        'exercise': 'Moderate',
        'family_history': 'Yes'
    }
    
    print("\nSample Patient Data:")
    for key, value in sample_patient.items():
        print(f"  {key}: {value}")
    
    # Make prediction
    print("\nMaking prediction...")
    result = predictor.predict_single(sample_patient)
    
    if result:
        print("\n" + "="*60)
        print("Prediction Results")
        print("="*60)
        print(f"Predicted Disease: {result['predicted_disease']}")
        print(f"Confidence: {result['confidence_percentage']}")
        print(f"\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['disease']}: {pred['confidence']}")
    else:
        print("Prediction failed. Make sure model is trained first.")


if __name__ == "__main__":
    demo_prediction()