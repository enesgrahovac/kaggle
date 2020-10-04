'''
You're a consultant for DelFalco's Italian Restaurant. The owner asked you to identify whether there are any foods on their menu 
that diners find disappointing.

The business owner suggested you use diner reviews from the Yelp website to determine which dishes people liked and disliked. 
You pulled the data from Yelp. Before you get to analysis, run the code cell below for a quick look at the data you have to work with.

'''

import pandas as pd

# Load in the data from JSON file
data = pd.read_json('./restaurant.json')

'''
The owner also gave you this list of menu items and common alternate spellings.
'''

menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]

import spacy
from spacy.matcher import PhraseMatcher

index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

# Load the SpaCy model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]

# Add the item patterns to the matcher. 
# Look at https://spacy.io/api/phrasematcher#add in the docs for help with this step
# Then uncomment the lines below 

matcher.add("MENU",            # Just a name for the set of rules we're matching to
           menu_tokens_list  
          )

# Find matches in the review_doc
matches = matcher(review_doc)


### Step 3: Matching on the whole dataset
 
from collections import defaultdict
item_ratings = defaultdict(list)

for idx, review in data.iterrows():
    doc = nlp(review.text)
    # Using the matcher from the previous exercise
    matches = matcher(doc)
    
    # Create a set of the items found in the review text
    found_items = set([doc[foundItem[1]:foundItem[2]].text.lower() for foundItem in matches])
    
    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for found_item in found_items: item_ratings[found_item].append(review.stars) 


### Step 4: What's the worst item reviewed?

# Calculate the mean ratings for each menu item as a dictionary
mean_ratings = {item: (sum(item_ratings[item])/len(item_ratings[item])) for item in item_ratings.keys()}
print(mean_ratings)

# Find the worst item, and write it as a string in worst_text. This can be multiple lines of code if you want.
worst_item = min(mean_ratings.keys(), key=(lambda k: mean_ratings[k]))
print(worst_item, mean_ratings[worst_item])

'''
Here is code to print the 10 best and 10 worst rated items. Look at the results, and decide whether you think 
it's important to consider the number of reviews when interpreting scores of which items are best and worst.
'''

counts = {item: len(ratings) for item, ratings in item_ratings.items()}

item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts:
    print(f"{item:>25}{counts[item]:>5}")

sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)

print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
    
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")

