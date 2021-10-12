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
