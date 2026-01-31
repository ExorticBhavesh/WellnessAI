"""
Health Chatbot Logic
Handles conversation flow and integrates ML predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from predict import HealthPredictor
import re


class HealthChatbot:
    def __init__(self):
        self.predictor = HealthPredictor(
            model_path='../model/model.pkl',
            preprocessor_path='../model/preprocessor.pkl'
        )
        self.conversation_state = 'greeting'
        self.user_data = {}
        self.questions = self._get_health_questions()
        self.current_question_index = 0
        
    def _get_health_questions(self):
        """Define health assessment questions"""
        return [
            {
                'key': 'age',
                'question': 'What is your age?',
                'type': 'number',
                'validation': lambda x: 0 < x < 120
            },
            {
                'key': 'gender',
                'question': 'What is your gender? (Male/Female/Other)',
                'type': 'text',
                'validation': lambda x: x.lower() in ['male', 'female', 'other']
            },
            {
                'key': 'blood_pressure',
                'question': 'What is your systolic blood pressure? (e.g., 120)',
                'type': 'number',
                'validation': lambda x: 70 < x < 250
            },
            {
                'key': 'cholesterol',
                'question': 'What is your cholesterol level? (mg/dL)',
                'type': 'number',
                'validation': lambda x: 100 < x < 400
            },
            {
                'key': 'glucose',
                'question': 'What is your blood glucose level? (mg/dL)',
                'type': 'number',
                'validation': lambda x: 50 < x < 300
            },
            {
                'key': 'bmi',
                'question': 'What is your BMI? (Body Mass Index)',
                'type': 'number',
                'validation': lambda x: 10 < x < 60
            },
            {
                'key': 'smoking',
                'question': 'Do you smoke? (Yes/No)',
                'type': 'text',
                'validation': lambda x: x.lower() in ['yes', 'no']
            },
            {
                'key': 'exercise',
                'question': 'How often do you exercise? (Low/Moderate/High)',
                'type': 'text',
                'validation': lambda x: x.lower() in ['low', 'moderate', 'high']
            },
            {
                'key': 'family_history',
                'question': 'Do you have a family history of chronic diseases? (Yes/No)',
                'type': 'text',
                'validation': lambda x: x.lower() in ['yes', 'no']
            }
        ]
    
    def greet(self):
        """Greeting message"""
        greeting = """
🏥 Welcome to the Health Assessment Chatbot! 👋

I'm here to help assess your health status based on various indicators.
I'll ask you a series of questions about your health, and then provide
a prediction using our AI model.

⚠️  DISCLAIMER: This is an AI-based assessment tool and should NOT replace
professional medical advice. Always consult with healthcare professionals
for medical concerns.

Would you like to start the health assessment? (yes/no)
"""
        return greeting
    
    def process_message(self, user_input):
        """Process user message and return bot response"""
        user_input = user_input.strip()
        
        # Handle greeting state
        if self.conversation_state == 'greeting':
            if user_input.lower() in ['yes', 'y', 'yeah', 'sure', 'ok', 'okay']:
                self.conversation_state = 'collecting_data'
                return self._ask_next_question()
            elif user_input.lower() in ['no', 'n', 'nah', 'nope']:
                return "No problem! Feel free to come back when you're ready. Take care! 👋"
            else:
                return "Please respond with 'yes' or 'no' to continue."
        
        # Handle data collection
        elif self.conversation_state == 'collecting_data':
            return self._handle_answer(user_input)
        
        # Handle general queries
        else:
            return self._handle_general_query(user_input)
    
    def _ask_next_question(self):
        """Get next question in sequence"""
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            return f"📋 Question {self.current_question_index + 1}/{len(self.questions)}:\n{question['question']}"
        else:
            return self._generate_prediction()
    
    def _handle_answer(self, user_input):
        """Process user's answer to current question"""
        question = self.questions[self.current_question_index]
        
        try:
            # Validate and store answer
            if question['type'] == 'number':
                value = float(user_input)
                if not question['validation'](value):
                    return f"⚠️  Invalid value. Please provide a valid {question['key']}."
                self.user_data[question['key']] = value
            else:
                value = user_input.strip()
                if not question['validation'](value):
                    return f"⚠️  Invalid input. Please provide a valid response."
                self.user_data[question['key']] = value.capitalize()
            
            # Move to next question
            self.current_question_index += 1
            return self._ask_next_question()
            
        except ValueError:
            return f"⚠️  Please provide a valid {question['type']} value."
    
    def _generate_prediction(self):
        """Generate health prediction based on collected data"""
        self.conversation_state = 'completed'
        
        response = "\n✅ Thank you for completing the assessment!\n"
        response += "\n📊 Analyzing your health data...\n"
        
        # Make prediction
        result = self.predictor.predict_single(self.user_data)
        
        if result:
            response += "\n" + "="*50 + "\n"
            response += "🔍 HEALTH ASSESSMENT RESULTS\n"
            response += "="*50 + "\n\n"
            
            response += f"🏥 Primary Prediction: {result['predicted_disease']}\n"
            response += f"📈 Confidence Level: {result['confidence_percentage']}\n\n"
            
            response += "🔝 Top 3 Possible Conditions:\n"
            for i, pred in enumerate(result['top_predictions'], 1):
                response += f"   {i}. {pred['disease']} - {pred['confidence']}\n"
            
            response += "\n" + "="*50 + "\n"
            response += "\n⚠️  IMPORTANT REMINDERS:\n"
            response += "• This is an AI-based prediction, not a medical diagnosis\n"
            response += "• Always consult healthcare professionals for medical advice\n"
            response += "• Regular health check-ups are recommended\n"
            response += "• Maintain a healthy lifestyle with proper diet and exercise\n"
            
            response += "\n💬 Would you like to:\n"
            response += "1. Start a new assessment\n"
            response += "2. Ask health-related questions\n"
            response += "3. Exit\n"
            
        else:
            response += "\n❌ Sorry, I couldn't generate a prediction. "
            response += "Please make sure the model is trained properly.\n"
        
        return response
    
    def _handle_general_query(self, user_input):
        """Handle general health queries"""
        user_input_lower = user_input.lower()
        
        # Check for common keywords
        if any(word in user_input_lower for word in ['new', 'restart', 'again', 'start']):
            self._reset_conversation()
            return "Starting a new assessment!\n\n" + self._ask_next_question()
        
        elif any(word in user_input_lower for word in ['exit', 'quit', 'bye', 'goodbye']):
            return "Thank you for using the Health Assessment Chatbot! Take care! 👋"
        
        elif any(word in user_input_lower for word in ['help', 'what', 'how']):
            return self._get_help_message()
        
        else:
            return self._get_health_info(user_input_lower)
    
    def _get_help_message(self):
        """Provide help information"""
        help_text = """
📚 HELP INFORMATION

This chatbot can:
✓ Assess your health status based on key indicators
✓ Predict potential health conditions
✓ Provide general health information

Commands:
• 'new' or 'restart' - Start a new assessment
• 'help' - Show this help message
• 'exit' or 'quit' - Exit the chatbot

Health topics I can discuss:
• Diabetes
• Heart Disease
• Hypertension
• General health tips
• Disease prevention

What would you like to know?
"""
        return help_text
    
    def _get_health_info(self, query):
        """Provide basic health information"""
        health_info = {
            'diabetes': """
🩺 DIABETES INFORMATION

Diabetes is a chronic condition that affects how your body processes blood sugar.

Key Points:
• Type 1: Body doesn't produce insulin
• Type 2: Body doesn't use insulin properly (most common)
• Risk factors: Obesity, family history, age, inactivity

Prevention Tips:
✓ Maintain healthy weight
✓ Exercise regularly (30+ min/day)
✓ Eat balanced diet (whole grains, vegetables)
✓ Limit sugar and refined carbs
✓ Regular health screenings
""",
            'heart': """
❤️ HEART DISEASE INFORMATION

Heart disease refers to various conditions affecting the heart and blood vessels.

Common Types:
• Coronary artery disease
• Heart rhythm problems
• Heart defects

Risk Factors:
• High blood pressure
• High cholesterol
• Smoking
• Diabetes
• Obesity

Prevention:
✓ Regular exercise
✓ Healthy diet
✓ Don't smoke
✓ Manage stress
✓ Regular check-ups
""",
            'hypertension': """
🩺 HYPERTENSION (HIGH BLOOD PRESSURE)

Hypertension is when blood pressure is consistently too high.

Normal BP: Less than 120/80 mmHg
High BP: 140/90 mmHg or higher

Risk Factors:
• Age
• Family history
• Obesity
• High salt intake
• Lack of exercise

Management:
✓ Reduce salt intake
✓ Regular exercise
✓ Maintain healthy weight
✓ Limit alcohol
✓ Manage stress
✓ Take medications as prescribed
""",
            'exercise': """
💪 EXERCISE & HEALTH

Regular exercise is crucial for good health!

Benefits:
• Strengthens heart and lungs
• Helps maintain healthy weight
• Reduces disease risk
• Improves mental health
• Boosts energy levels

Recommendations:
✓ 150 min moderate exercise/week
✓ Or 75 min vigorous exercise/week
✓ Include strength training 2x/week
✓ Start slowly and increase gradually
""",
            'nutrition': """
🥗 NUTRITION BASICS

Healthy eating is fundamental to wellness!

Key Principles:
• Eat variety of foods
• Focus on whole foods
• Limit processed foods
• Stay hydrated
• Watch portion sizes

Include:
✓ Fruits & vegetables (5+ servings/day)
✓ Whole grains
✓ Lean proteins
✓ Healthy fats
✓ Low-fat dairy

Limit:
✗ Added sugars
✗ Saturated fats
✗ Sodium
✗ Processed foods
"""
        }
        
        # Check which topic matches query
        for topic, info in health_info.items():
            if topic in query:
                return info
        
        # Default response
        return """
I can provide information about:
• Diabetes
• Heart disease
• Hypertension
• Exercise
• Nutrition

Try asking about one of these topics, or type 'help' for more options!
"""
    
    def _reset_conversation(self):
        """Reset conversation for new assessment"""
        self.conversation_state = 'collecting_data'
        self.user_data = {}
        self.current_question_index = 0


def main():
    """Command-line chatbot interface"""
    print("\n" + "="*60)
    chatbot = HealthChatbot()
    print(chatbot.greet())
    
    while True:

        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
        
        response = chatbot.process_message(user_input)
        print(f"\nBot: {response}")
        
        if 'goodbye' in response.lower() or 'take care' in response.lower():
            if chatbot.conversation_state == 'completed':
                break


if __name__ == "__main__":
    main()