class photonic_problem:
    """Minimal interface for custom photonic problems."""

    def __init__(self):
        """Initialize placeholder problem."""
        self.n = None

    def setup_structure(self):
        """Prepare the underlying simulation structure."""
        pass

    def __call__(self):
        """Evaluate the photonic structure."""
        pass
