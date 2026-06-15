from enum import Enum


class PrimaryCategories(Enum):
    BBO = "black box optimisation"
    CO = "combinatorial optimisation"
    ML = "machine learning"
    PD = "pipeline design"
    PERFORMANCE = "performance"
    LOGISTICS = "logistics"
    OTHER = "other"


class Benchmark(Enum):
    PHOTONICS = "Photonics"
    AUTOML = "AutoML"
    KERNEL_TUNER = "KernelTuner"

    MATRIX_MULTIPLICATION = "MatrixMultiplication"

    ERDOS_MIN_OVERLAP = "ErdosMinOverlap"
    AUTO_CORRELATION = "AutoCorrelation"
    FOURIER_UNCERTAINTY = "FourierUncertaintyInequality"

    HEILBRONN_CONVEX_REGION = "HeilbronnConvexRegion"
    HEILBRONN_TRIANGLE = "HeilbronnTriangle"

    MIN_MAX_DISTANCE_RATIO = "MinMaxDistanceRatio"

    GRAPH_COLOURING = "GraphColouring"

    EUCLIDEAN_STEINER_TREE = "EuclideanSteinerTree"

    SUMS_VS_DIFFERENCES = "SumsVsDifferences"

    VRP = "VehicleRoutingProblem"
    TSP = "TravelingSalesmanProblem"

    KISSING_NUMBER_11D = "KissingNumber11D"
    CIRCLE_PACKING = "CirclePacking"
    UNIT_SQUARE_PACKING = "UnitSquarePacking"
    HEXAGONAL_PACKING = "HexagonalPacking"
    RECTANGLE_PACKING = "RectanglePacking"
    SPHERICAL_CODE = "SphericalCode"


class NoiseType(Enum):
    NOISELESS = "noiseless"
    NOISY = "noisy"


class ObjectiveType(Enum):
    SINGLE_OBJECTIVE = "single objective"
    MULTI_OBJECTIVE = "multi objective"


class VariableType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    INTEGER = "integer"
    MIXED = "mixed"


class StructureTag(Enum):
    GRAPH = "graph"
    ROUTING = "routing"
    PACKING = "packing"
    GEOMETRIC = "geometric"
    SPHERICAL = "spherical"
    TIME_SERIES = "time_series"
    MATRIX = "matrix"
    SIGNAL = "signal"
    MATHS = "mathematics"


class ComplexityTag(Enum):
    NP_HARD = "np_hard"
    NP_COMPLETE = "np_complete"
    UNKNOWN = "unknown"


# ============================================================
# OPTIONAL: HIGH-LEVEL DOMAIN TAGS (if still needed)
# ============================================================


class DomainTag(Enum):
    OPTIMIZATION = "optimization"
    MATHEMATICS = "mathematics"
    COMPUTATIONAL_GEOMETRY = "computational_geometry"
    OPERATIONS_RESEARCH = "operations_research"
    MACHINE_LEARNING = "machine_learning"
    SIGNAL_PROCESSING = "signal_processing"
