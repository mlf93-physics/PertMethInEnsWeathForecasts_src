import pprint as pp
import pyperclip
from config import MODEL, LICENCE, GLOBAL_PARAMS, NUMBA_CACHE


def ask_user(question: str):
    answer = input("\n" + question + " [y/n]").lower().strip()
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


def confirm_run_setup(args: dict):
    print("CONFIRM SETUP TO RUN:\n")
    print(f"Model: {MODEL}")
    print(f"Licence: {LICENCE}")
    print(f"Numba cache: {NUMBA_CACHE}")
    print()

    print("\nRun-time arguments:\n\n", pp.pformat(args))

    print("\nGlobal parameters\n\n", pp.pformat(GLOBAL_PARAMS.__dict__))

    confirm = ask_user("Please confirm the current setup to run")

    if not confirm:
        exit()
