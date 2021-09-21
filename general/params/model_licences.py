class Model:
    def __init__(self, id: int = None, name: str = None):
        self.id = id
        self.name = name

    def __repr__(self):
        return self.name

    @property
    def home_dir_name(self):
        suffix = "_experiments"
        return self.name.lower() + suffix


class Models:
    """Class to hold the different model licenses"""

    SHELL_MODEL = Model(id=0, name="SHELL_MODEL")
    LORENTZ63 = Model(id=1, name="LORENTZ63")
