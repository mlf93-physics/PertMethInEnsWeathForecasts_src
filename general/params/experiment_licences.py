class Experiment:
    def __init__(self, id: int = None, name: str = None):
        self.id: int = id
        self.name: str = name

    def __repr__(self):
        return self.name


class Experiments:
    NORMAL_PERTURBATION = Experiment(id=1, name="NORMAL_PERTURBATION")
    LORENTZ_BLOCK = Experiment(id=2, name="LORENTZ_BLOCK")
    BREEDING_VECTORS = Experiment(id=3, name="BREEDING_VECTORS")
    BREEDING_EOF_VECTORS = Experiment(id=4, name="BREEDING_EOF_VECTORS")
    SINGULAR_VECTORS = Experiment(id=5, name="SINGULAR_VECTORS")
    LYAPUNOV_VECTORS = Experiment(id=6, name="LYAPUNOV_VECTORS")
    HYPER_DIFFUSIVITY = Experiment(id=7, name="HYPER_DIFFUSIVITY")
    COMPARISON = Experiment(id=8, name="COMPARISON")
    VERIFICATION = Experiment(id=10, name="VERIFICATION")
