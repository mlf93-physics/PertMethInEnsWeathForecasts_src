class InvalidArgument(Exception):
    """Exception raised for an invalid runtime argument

    Attributes:
        argument -- the invalid argument
        message -- explanation of the error
    """

    def __init__(self, message="", argument=None):
        self.argument = argument

        if len(message) == 0:
            self.message = "Argument is invalid for the current setup"
        else:
            self.message = message

        super().__init__(f"{self.message}. Argument: {self.argument}")
