# Russian Cube Text Adventure Game

def show_title():
    print("""
    ========================================
            RUSSIAN CUBIC ADVENTURE
    ========================================
    """)

def start_game():
    print("You are standing at the beginning of a dark forest.")
    print("There is a path to your left and right.")
    choice = input("Which path do you choose? (left/right): ").lower()
    
    if choice == "left":
        left_path()
    elif choice == "right":
        right_path()
    else:
        print("Invalid choice. Game over.")
        play_again()

def left_path():
    print("\nYou head down the left path and find a cottage.")
    print("An old woman greets you and offers you food.")
    choice = input("Do you accept? (yes/no): ").lower()
    
    if choice == "yes":
        print("\nYou ate the food and felt strange...")
        print("The old woman cackles and you fall asleep.")
        game_over()
    elif choice == "no":
        print("\nYou politely decline and continue on your journey.")
        victory()
    else:
        print("Invalid choice. Game over.")
        play_again()

def right_path():
    print("\nYou head down the right path and find a river.")
    print("There's a small boat to cross it.")
    choice = input("Do you get in the boat? (yes/no): ").lower()
    
    if choice == "yes":
        print("\nYou row across the river and find a treasure chest.")
        victory()
    elif choice == "no":
        print("\nYou decide to go back to the forest.")
        start_game()
    else:
        print("Invalid choice. Game over.")
        play_again()

def game_over():
    print("\nGame Over!")
    play_again()

def victory():
    print("\nCongratulations! You found the treasure!")
    play_again()

def play_again():
    choice = input("\nWould you like to play again? (yes/no): ").lower()
    if choice == "yes":
        start_game()
    elif choice == "no":
        print("Thank you for playing!")
    else:
        print("Invalid choice. Game over.")

show_title()
start_game()
