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


class LicenceImplementationError(Exception):
    """Exception raised if an algorithm is not implemented for the current licence

    Attributes:
        licence -- the licence
        message -- explanation of the error
    """

    def __init__(self, message="", licence=None):
        self.licence = licence

        if len(message) == 0:
            self.message = (
                "This functionality is currently not available/implemented"
                + " for the current licence"
            )
        else:
            self.message = message

        super().__init__(f"{self.message}. Licence: {self.licence}")


class ModelError(Exception):
    """Exception raised for an invalid model

    Attributes:
        message -- explanation of the error
        model -- the invalid model
    """

    def __init__(self, message="", model=None):
        self.model = model

        if len(message) == 0:
            self.message = "Model is invalid"
        else:
            self.message = message

        super().__init__(f"{self.message}. Model: {self.model}")
