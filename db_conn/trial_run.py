# dummy_recommender.py



import time


def main():
    # Step 1: Greet the user and get the initial query.
    print("Assistant: Hello! How can I help you today?")
    initial_query = input("User: ")
    
    # Simulate an LLM-based acknowledgement of the initial query.
    print("\nAssistant: Okay, I hear you want recommendations similar to 'Inception'.")
    
    # Step 2: Ask clarifying questions.
    # Question 1: Preferred genre.
    genre = input("Assistant: What genre do you prefer? (e.g., Sci-Fi, Action, Drama):\nUser: ")
    time.sleep(5)
    
    # Question 2: Era preference.
    era = input("Assistant: Do you prefer recent movies or classics? (Enter 'recent' or 'classic'):\nUser: ")
    time.sleep(5)

    # Question 3: Favorite actor (optional).
    actor = input("Assistant: Is there an actor or actress you particularly like? (If none, just press Enter):\nUser: ")
    time.sleep(5)

    # Question 4: Director (optional).
    director = input("Assistant: Any favorite director? (If none, just press Enter):\nUser: ")
    time.sleep(5)

    print("\nAssistant: Thank you for the details!")
    
    # (Optional) You could simulate updating a user database here based on the answers.
    
    # Step 3: Provide hard-coded recommendations based on the input.
    print("\nAssistant: Based on your input, here are five movie recommendations:")
    
    recommendations = [
        "Interstellar (2014) - A visually stunning sci-fi that explores space, time, and human emotion.",
        "The Matrix (1999) - A revolutionary action-packed sci-fi that challenges reality.",
        "Inception (2010) - A mind-bending thriller about dreams and the subconscious.",
        "The Prestige (2006) - A mystery drama with intense rivalries and twists, directed by Christopher Nolan.",
        "Memento (2000) - A uniquely structured thriller that invites you to piece together the story in reverse."
    ]
    
    for rec in recommendations:
        print("  -", rec)
    
    print("\nAssistant: Enjoy watching your recommended movies!")

if __name__ == "__main__":
    main()
