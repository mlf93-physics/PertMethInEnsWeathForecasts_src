class Experiment:
    def __init__(self, id: int = None, name: str = None):
        self.id = id
        self.name = name

    def __repr__(self):
        return self.name


class Experiments:
    NORMAL_PERTURBATION = Experiment(id=1, name="NORMAL_PERTURBATION")
    LORENTZ_BLOCK = Experiment(id=2, name="LORENTZ_BLOCK")
    BREEDING_VECTORS = Experiment(id=3, name="BREEDING_VECTORS")
    SINGULAR_VECTORS = Experiment(id=4, name="SINGULAR_VECTORS")
    LYAPUNOV_VECTORS = Experiment(id=5, name="LYAPUNOV_VECTORS")
