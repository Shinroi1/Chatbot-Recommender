import json
import pickle
import random
import re

from recommender import (
    recommend_dishes_exclude_ingredient,
    recommend_dishes_by_category,
    recommend_dishes_by_budget,
    finalize_recommendation,
    parse_ingredients,
    known_ingredients,
    known_categories 
)

# Load intent recognition model and vectorizer
with open('random forest.pkl', 'rb') as file:
    rfc_model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Conversation state to remember user preferences
conversation_state = {
    "budget": None,
    "excluded_ingredients": [],
    "included_dishes": [],
}

def predict_intent(text):
    X_input = vectorizer.transform([text])
    return rfc_model.predict(X_input)[0]

def handle_customer_service(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm sorry I do not understand."

def parse_amount(text):
    match = re.search(r'\b\d+\b', text)
    if match:
        return int(match.group())
    return None

def parse_category(text):
    for category in known_categories:
        if category in text.lower():
            return category
    return None

def chatbot_response(user_query):
    # Predict the intent
    intent = predict_intent(user_query)
    
    print(f"User query: {user_query}")
    print(f"Predicted intent: {intent}")
    print(f"Conversation state: {conversation_state}")
    
    preferences_updated = False
    response = ""

    # Map intents to appropriate functions
    if intent == "recommended_dishes_by_budget":
        # Set user budget and recommend dishes by budget
        conversation_state['budget'] = parse_amount(user_query)
        categories = conversation_state.get('included_dishes', [])
        recommendations, total_price = recommend_dishes_by_budget(conversation_state['budget'], categories)
        if recommendations:
            response = f"I recommend: {', '.join(recommendations)}. Total price: {total_price:.2f}. Would you like to refine your preferences?"
        else:
            response = "Sorry, I couldn't find any dishes within your budget. Would you like to adjust your budget or preferences?"
            
            preferences_updated = True

    elif intent == "recommended_dishes_by_category":
        # Add user-preferred categories and recommend dishes by category
        category = parse_category(user_query)
        if category:
            conversation_state['included_dishes'].append(category)
            recommendations = recommend_dishes_by_category(conversation_state['included_dishes'])
            if recommendations:
                response = f"I recommend: {', '.join(recommendations)} from the {category.capitalize()} category. Would you like to add more preferences?"
            else:
                response = f"Sorry, I couldn't find any dishes in the {category.capitalize()} category. Would you like to choose another category?"
        else:
            response = "I couldn't identify the category. Could you specify again?"
        preferences_updated = True

    elif intent == "exclude_ingredients":
        # Exclude ingredients from the recommendations
        ingredients = parse_ingredients(user_query, known_ingredients)
        
        if ingredients:
            conversation_state['excluded_ingredients'].extend(ingredients)
            conversation_state['excluded_ingredients'] = list(set(conversation_state['excluded_ingredients']))
            
            # Generate recomemndations after excluding the ingredients
            recommendations = recommend_dishes_exclude_ingredient(conversation_state['excluded_ingredients'])
            
            if recommendations:
                response = f"I have excluded {'the'.join(ingredients)} from your recommendations. Would you like to add any more preferences?"
            else:
                response = f"Sorry, I couldn't find any dishes after excluding {''.join(ingredients)}. Would you like to refine your preferences?"
        else:
            response = "I couldn't identify the ingredients. Could you specify again?"
        preferences_updated = True

    elif intent == "menu_recommendation":
        # Prompt for more preferences
        response = "Would you like to add any filters, like: \n - Preferred Budget \n - Excluded ingredients (for allergies) \n - Category \n - Included Ingredients?"

    elif intent == "category_recommendation":
        # Same as 'recommended_dishes_by_category'
        category = parse_category(user_query)
        if category:
            conversation_state['included_dishes'].append(category)
            response = f"Added {category.capitalize()} to your preferences. Should I proceed with the recommendation?"
        else:
            response = "I couldn't identify the category. Could you specify again?"
        preferences_updated = True

    elif intent == "budget_recommendation":
        # Same as 'recommended_dishes_by_budget'
        conversation_state['budget'] = parse_amount(user_query)
        response = f"Your budget is set to {conversation_state['budget']}. Would you like to specify any other preferences?"
        preferences_updated = True

    elif intent == "accept_recommendation":
        # Finalize and recommend dishes based on conversation state
        response = finalize_recommendation(conversation_state)

    elif intent == "reject_recommendation":
        # Try again with a different recommendation
        response = "I see. Let me recommend something else for you."
        response += finalize_recommendation(conversation_state)

    else:
        # Fallback to intents.json-based responses if no hardcoded response exists
        response = handle_customer_service(intent)

    return response

if __name__ == "__main__":
    print("Welcome to La Nuova Pasteleria's chatbot!")
    while True:
        user_query = input("You: ")
        response = chatbot_response(user_query)
        print(f"Bot: {response}")
        
        if "proceed" in user_query.lower() or "finalize" in user_query.lower():
            final_response = finalize_recommendation(conversation_state)
            print(f"Bot: {final_response}")
