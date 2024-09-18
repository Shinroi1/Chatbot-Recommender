import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import difflib
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')

# Load data
df = pd.read_csv('menu.csv')
df.columns = ['id', 'dish_name', 'price', 'category', 'sub_category', 'ingredients']

df['dish_name'] = df['dish_name'].str.lower().str.strip()

# Handle NaN values and ensure ingredients are strings
df['ingredients'] = df['ingredients'].fillna('').astype(str).str.lower()

# Ensure ingredients are split into lists
df['ingredients'] = df['ingredients'].apply(lambda x: x.split(','))


# Tokenize ingredients
df['tokenized_ingredients'] = df['ingredients'].apply(lambda x: word_tokenize(' '.join(x)))

# Print to verify
# print(f"Tokenized:")
# print(df[['id', 'dish_name', 'ingredients']].head(46))

# Stemming
stemmer = PorterStemmer()

# Apply stemming to ingredients
df['stemmed_ingredients'] = df['tokenized_ingredients'].apply(lambda x: [stemmer.stem(word) for word in x])

# Print to verify
# print(f"Stemmed:")
# print(df[['id', 'dish_name', 'ingredients']].head(47))

# Lemmatize
lemmatizer = nltk.stem.WordNetLemmatizer()

# Apply lemmatization to ingredients
df['lemmatized_ingredients'] = df['tokenized_ingredients'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Print to verify
# print(f"Lemmatized:")
# print(df[['id', 'dish_name', 'ingredients']].head(47))

# Flatten the list of known ingredients
known_ingredients = list(set([item.strip() for sublist in df['ingredients'] for item in sublist]))

# Vectorize ingredients (optional, in case of NLP-based filtering)
ingredient_vectorizer = TfidfVectorizer()
X_ingredients = ingredient_vectorizer.fit_transform(df['ingredients'].apply(lambda x: ' '.join(x)))

# Get the unique categories in lowercase for matching
known_categories = list(df['category'].str.lower().unique())

# Functions to recommend dishes based on user preferences
# def parse_ingredients(text, ingredients):
#     text = text.lower()
#     matched_ingredients = []
#     for ingredient in ingredients:
#         if re.search(r'\b' + re.escape(ingredient) + r'\b', text):
#             matched_ingredients.append(ingredient)
#     matched_ingredients = [ingredient for ingredient in matched_ingredients if ingredient.strip()]
#     return matched_ingredients

def parse_ingredients(text, ingredients):
    # Split user text by common delimiters such as commas and 'and'
    text = text.lower()
    possible_excluded_ingredient = re.split(r'[,\s]+and\+|,|\sand\s',text)
    
    matched_ingredients = []
    for ingredient in possible_excluded_ingredient:
        
        ingredient = ingredient.strip()
        
        # Use fuzzy matching to detect close matches
        closest_matches = difflib.get_close_matches(ingredient, ingredients, cutoff=0.7)
        if closest_matches:
            matched_ingredients.append(closest_matches[0])
            
    # matched_ingredients = [ingredient for ingredient in matched_ingredients if ingredient.strip()]
    
    return list(set(matched_ingredients))

def recommend_dishes_exclude_ingredient(excluded_ingredients):
    # Lemmatize and stem user input to ensure better matching
    excluded_ingredients = [lemmatizer.lemmatize(ingredient.lower().strip()) for ingredient in excluded_ingredients if ingredient.strip()]

    def has_excluded_ingredients(ingredients_list):
        lemmatized_ingredients = [lemmatizer.lemmatize(ingredient) for ingredient in ingredients_list]
        return any(ingredient in lemmatized_ingredients for ingredient in excluded_ingredients)

    # Filter out dishes with the excluded ingredients
    filtered_dishes = df[~df['ingredients'].apply(has_excluded_ingredients)]
    return filtered_dishes['dish_name'].tolist()

# def recommend_dishes_exclude_ingredient(excluded_ingredients):
#     # excluded_ingredients = [ingredient.lower().strip() for ingredient in excluded_ingredients if ingredient.strip()]
        
#     excluded_ingredients = [lemmatizer.lemmatize(ingredient.lower().strip()) for ingredient in excluded_ingredients if ingredient.strip()]

#     def has_excluded_ingredients(ingredients_list):
#         lemmatized_ingredients = [lemmatizer.lemmatize(ingredient) for ingredient in ingredients_list]
#         return any(ingredient in lemmatized_ingredients for ingredient in excluded_ingredients)
#         # return any(ingredient in excluded_ingredients for ingredient in ingredients_list)
    
#     filtered_dishes = df[~df['ingredients'].apply(has_excluded_ingredients)]
#     return filtered_dishes['dish_name'].tolist()

def recommend_dishes_by_category(categories):
    # Ensure all categories match those in the dataset
    categories = [cat.lower() for cat in categories]
    filtered_dishes = df[df['category'].str.lower().isin(categories)]
    return filtered_dishes['dish_name'].tolist()

def recommend_dishes_by_budget(budget, categories=None):
    filtered_dishes = df[df['price'] <= budget]
    if categories:
        filtered_dishes = filtered_dishes[filtered_dishes['category'].str.lower().isin(categories)]
    
    if filtered_dishes.empty:
        return [], 0
    
    recommendations = filtered_dishes.sort_values(by='price')['dish_name'].tolist()
    total_price = filtered_dishes['price'].sum()
    return recommendations, total_price

def finalize_recommendation(conversation_state):
    
    conversation_state['excluded_ingredients'] = list(set(conversation_state['excluded_ingredients']))
        
    # budget = conversation_state.get('budget')
    # excluded_ingredients = conversation_state.get('excluded_ingredients', [])
    
    excluded_ingredients = [ingredient.lower().strip() for ingredient in conversation_state.get('excluded_ingredients', []) if ingredient.strip()]
    included_dishes = [dish.lower().strip() for dish in conversation_state.get('included_dishes', []) if dish.strip()]
    budget = conversation_state.get('budget')
    
    # Step 1: Filter dishes based on user preferences (category, budget, etc.)
    if budget:
        filtered_dishes = df[df['price'] <= budget]
        if included_dishes:
            filtered_dishes = filtered_dishes[filtered_dishes['category'].str.lower().isin(included_dishes)]
        for ingredient in excluded_ingredients:
            filtered_dishes = filtered_dishes[~filtered_dishes['ingredients'].apply(lambda x: ingredient in x)]
                
        # Step 2: Exclude unwanted ingredients
        
        # if excluded_ingredients:
        if excluded_ingredients:
            filtered_dishes = filtered_dishes[~filtered_dishes['ingredients'].apply(lambda x: any(ingredient in x for ingredient in excluded_ingredients))]
            
        print(f"Filtered dishes after excluding: {filtered_dishes[['dish_name', 'ingredients']]}")
                
        # Step 3: Sort dishes by price, and start recommending
        filtered_dishes = filtered_dishes.sort_values(by='price')
        recommendations = []
        total_price = 0
        
        # Step 4: Loop through filtered dishes and recommend within budget
        for _, row in filtered_dishes.iterrows():
            if total_price + row['price'] <= budget:
                recommendations.append(row['dish_name'].capitalize())
                total_price += row['price']
       
        if recommendations:
            return f"I recommend: {', '.join(recommendations)}. Total price: {total_price:.2f}. Is this acceptable?"
        else:
            return "Sorry, I couldn't find any dishes matching your preferences."
