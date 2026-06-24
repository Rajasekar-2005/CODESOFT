import re
import random
from datetime import datetime

class RuleBasedChatbot:
    def __init__(self):
        self.name = "ChatBot"
        self.user_name = None
        
        # Define response patterns
        self.patterns = {
            'greeting': [
                r'\b(hi|hello|hey|greetings|sup|what\'s up)\b',
            ],
            'goodbye': [
                r'\b(bye|goodbye|see you|exit|quit|farewell)\b',
            ],
            'name_query': [
                r'\b(what is your name|your name|who are you)\b',
            ],
            'user_name': [
                r'my name is (\w+)',
                r'i am (\w+)',
                r'call me (\w+)',
            ],
            'how_are_you': [
                r'\b(how are you|how do you do|how\'s it going)\b',
            ],
            'thanks': [
                r'\b(thanks|thank you|appreciate it|thx)\b',
            ],
            'help': [
                r'\b(help|what can you do|your capabilities)\b',
            ],
            'time': [
                r'\b(what time|current time|time now)\b',
            ],
            'date': [
                r'\b(what date|today\'s date|current date)\b',
            ],
            'weather': [
                r'\b(weather|temperature|forecast)\b',
            ],
            'joke': [
                r'\b(joke|make me laugh|tell me something funny)\b',
            ],
            'age': [
                r'\b(how old are you|your age|age)\b',
            ],
            'location': [
                r'\b(where are you|your location|where do you live)\b',
            ],
            'creator': [
                r'\b(who created you|who made you|your creator)\b',
            ],
            'love': [
                r'\b(i love you|love you)\b',
            ],
        }
        
        # Define responses
        self.responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! Great to see you!",
                "Greetings! How may I assist you?",
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! It was nice talking to you!",
                "Farewell! Come back soon!",
            ],
            'name_query': [
                f"My name is {self.name}. Nice to meet you!",
                f"I'm {self.name}, your virtual assistant!",
                f"You can call me {self.name}!",
            ],
            'how_are_you': [
                "I'm doing great, thank you for asking! How about you?",
                "I'm fantastic! How are you doing?",
                "I'm excellent! Thanks for asking!",
            ],
            'thanks': [
                "You're welcome!",
                "Happy to help!",
                "Anytime!",
                "My pleasure!",
            ],
            'help': [
                "I can help you with:\n- Greetings and basic conversation\n- Tell you the current time and date\n- Share jokes\n- Answer questions about myself\n- Have a friendly chat\nJust ask me anything!",
            ],
            'time': [
                f"The current time is {datetime.now().strftime('%H:%M:%S')}",
            ],
            'date': [
                f"Today's date is {datetime.now().strftime('%Y-%m-%d')}",
            ],
            'weather': [
                "I'm sorry, I don't have access to real-time weather data. You might want to check a weather website!",
            ],
            'joke': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "Why don't eggs tell jokes? They'd crack each other up!",
                "What do you call a bear with no teeth? A gummy bear!",
                "Why did the math book look sad? Because it had too many problems!",
            ],
            'age': [
                "I'm timeless! I was just created, but I learn quickly!",
                "Age is just a number for AI like me!",
            ],
            'location': [
                "I exist in the digital realm, everywhere and nowhere at once!",
                "I live in the cloud! No rent to pay!",
            ],
            'creator': [
                "I was created by a developer to help answer your questions!",
                "A talented programmer brought me to life!",
            ],
            'love': [
                "That's sweet! I appreciate you too!",
                "Aww, you're making me blush! (If I could blush)",
            ],
            'default': [
                "I'm not sure I understand. Can you rephrase that?",
                "Interesting! Tell me more.",
                "I'm still learning. Could you ask something else?",
                "Hmm, I don't have information about that. Ask me something else!",
            ],
        }
    
    def match_pattern(self, user_input):
        """Match user input against predefined patterns"""
        user_input = user_input.lower().strip()
        
        # Check for user name patterns first
        for pattern in self.patterns['user_name']:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                self.user_name = match.group(1).capitalize()
                return 'user_name', self.user_name
        
        # Check other patterns
        for intent, patterns in self.patterns.items():
            if intent == 'user_name':
                continue
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return intent, None
        
        return 'default', None
    
    def get_response(self, user_input):
        """Generate appropriate response based on user input"""
        intent, extracted_data = self.match_pattern(user_input)
        
        # Handle user name
        if intent == 'user_name':
            return f"Nice to meet you, {self.user_name}! How can I help you today?"
        
        # Handle time (dynamic)
        if intent == 'time':
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
        
        # Handle date (dynamic)
        if intent == 'date':
            return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"
        
        # Get random response from the intent's response list
        if intent in self.responses:
            response = random.choice(self.responses[intent])
            
            # Personalize greeting if we know the user's name
            if intent == 'greeting' and self.user_name:
                response = f"Hello {self.user_name}! " + response
            
            return response
        
        # Default response
        return random.choice(self.responses['default'])
    
    def start(self):
        """Start the chatbot conversation"""
        print("=" * 60)
        print(f"  Welcome to {self.name} - Your Rule-Based Assistant!")
        print("=" * 60)
        print("\nType 'bye', 'exit', or 'quit' to end the conversation.")
        print("Type 'help' to see what I can do.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    print(f"{self.name}: Please say something!")
                    continue
                
                # Check for exit commands
                if re.search(r'\b(bye|exit|quit|goodbye)\b', user_input, re.IGNORECASE):
                    print(f"{self.name}: {random.choice(self.responses['goodbye'])}")
                    break
                
                # Get and print response
                response = self.get_response(user_input)
                print(f"{self.name}: {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n{self.name}: Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"{self.name}: Oops! Something went wrong. Let's continue!")
                continue

def main():
    """Main function to run the chatbot"""
    chatbot = RuleBasedChatbot()
    chatbot.start()

if __name__ == "__main__":
    main()