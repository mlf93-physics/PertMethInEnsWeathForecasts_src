class Experiment:
    def __init__(self, id: int = None, name: str = None):
        self.id: int = id
        self.name: str = name

    def __repr__(self):
        return self.name


class Experiments:
    NORMAL_PERTURBATION = Experiment(id=1, name="NORMAL_PERTURBATION")
    TANGENT_LINEAR = Experiment(id=2, name="TANGENT_LINEAR")
    LORENTZ_BLOCK = Experiment(id=3, name="LORENTZ_BLOCK")
    BREEDING_VECTORS = Experiment(id=4, name="BREEDING_VECTORS")
    BREEDING_EOF_VECTORS = Experiment(id=5, name="BREEDING_EOF_VECTORS")
    SINGULAR_VECTORS = Experiment(id=6, name="SINGULAR_VECTORS")
    FINAL_SINGULAR_VECTORS = Experiment(id=7, name="FINAL_SINGULAR_VECTORS")
    LYAPUNOV_VECTORS = Experiment(id=8, name="LYAPUNOV_VECTORS")
    ADJ_LYAPUNOV_VECTORS = Experiment(id=9, name="ADJ_LYAPUNOV_VECTORS")
    HYPER_DIFFUSIVITY = Experiment(id=10, name="HYPER_DIFFUSIVITY")
    COMPARISON = Experiment(id=11, name="COMPARISON")
    VERIFICATION = Experiment(id=12, name="VERIFICATION")
