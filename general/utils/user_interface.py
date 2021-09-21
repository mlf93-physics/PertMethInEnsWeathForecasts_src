def ask_user(question):
    answer = input(question + " [y/n]").lower().strip()

    try:
        if answer[0] == "y":
            return True
        elif answer[0] == "n":
            return False
        else:
            print("Invalid Input")
            return ask_user()

    except Exception as error:
        print("Please enter valid inputs")
        print(error)
        return ask_user()
