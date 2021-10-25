import pyperclip


def ask_user(question: str):
    answer = input(question + " [y/n]").lower().strip()
    print(answer)

    try:
        if len(answer) == 0:
            return True
        elif answer[0] == "y" or answer[0] == "yes":
            return True
        elif answer[0] == "n" or answer[0] == "no":
            return False
        else:
            print("Invalid Input")
            return ask_user(question)

    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user(question)


def get_name_input(formulation: str, proposed_input: str = ""):

    # Print proposed input
    print(formulation, proposed_input)

    # Copy propesed_input to clipboard
    pyperclip.copy(proposed_input)

    # Ask user to change or not
    print("\nEnter new name to change it (name is in clipboard), otherwise press enter")

    user_input = input()

    name = proposed_input
    if len(user_input) > 0:
        name = user_input

    return name
