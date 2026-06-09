from typing import Any
# region Base Function
def check_key_words_exist_in(config: dict['str', 'Any']) -> bool:
    keys = ['tags', 'name', 'prompt', 'minimisation', 'evaluator', 'config']
    for key in keys:
        if key not in config:
            print(f'{key} not found in {config.keys()}')
            return False
    return True
#endregion


# region Analysis

def test_config_auto_correlation_1():
    from iohblade.benchmarks.analysis import AutoCorrIneq1
    acr1 = AutoCorrIneq1()
    config = acr1.get_config()
    assert check_key_words_exist_in(config)

def test_config_auto_correlation_2():
    from iohblade.benchmarks.analysis import  AutoCorrIneq2
    acr2 = AutoCorrIneq2()
    config = acr2.get_config()
    assert check_key_words_exist_in(config)

def test_config_auto_correlation_3():
    from iohblade.benchmarks.analysis import AutoCorrIneq3
    acr3 = AutoCorrIneq3()
    config = acr3.get_config()
    assert check_key_words_exist_in(config)
#endregion

#region BBOB

def test_config_BBOB():
    from iohblade.benchmarks.BBOB.bbob_sboxcost import BBOB_SBOX
    bbob = BBOB_SBOX()
    config = bbob.get_config()
    assert check_key_words_exist_in(config)

def test_config_MA_BBOB():
    from iohblade.benchmarks.BBOB.mabbob import MA_BBOB
    ma_bbob = MA_BBOB()
    config = ma_bbob.get_config()
    assert check_key_words_exist_in(config)

#endregion

#region Combinatorics
def test_config_erdos_overlap():
    from iohblade.benchmarks.combinatorics import ErdosMinOverlap
    eop = ErdosMinOverlap()
    config = eop.get_config()
    assert check_key_words_exist_in(config)

def test_config_euclidean_steiner_tree():
    from iohblade.benchmarks.combinatorics import EuclidianSteinerTree
    est = EuclidianSteinerTree(10)
    config = est.get_config()
    assert check_key_words_exist_in(config)

def test_graph_colouring():
    from iohblade.benchmarks.combinatorics import GraphColoring
    gc = GraphColoring('1')
    config = gc.get_config()
    assert check_key_words_exist_in(config)
#endregion

# region Fourier Uncertainty Inequality
def test_fourier():
    from iohblade.benchmarks.fourier import UncertaintyInequality
    uce = UncertaintyInequality()
    config = uce.get_config()
    assert check_key_words_exist_in(config)

# endregion

#region Geometry
def test_heilbronn_convex_region():
    from iohblade.benchmarks.geometry import HeilbronnConvexRegion
    hbc = HeilbronnConvexRegion(13)
    config = hbc.get_config()
    assert check_key_words_exist_in(config)

def test_heibronn_triangle():
    from iohblade.benchmarks.geometry import HeilbronnTriangle
    hbt = HeilbronnTriangle(10)
    config = hbt.get_config()
    assert check_key_words_exist_in(config)

def test_kissing_number_11D():
    from iohblade.benchmarks.geometry import KissingNumber11D
    kn = KissingNumber11D()
    config = kn.get_config()
    assert check_key_words_exist_in(config)

def test_min_max_distance_ration():
    from iohblade.benchmarks.geometry import MinMaxMinDistanceRatio
    mmd = MinMaxMinDistanceRatio(11, 3)
    config = mmd.get_config()
    assert check_key_words_exist_in(config)

def test_spherical_code():
    from iohblade.benchmarks.geometry import SphericalCode
    sc = SphericalCode()
    config = sc.get_config()
    assert check_key_words_exist_in(config)
#endregion

#region Kernel Tuner
def test_kernel_tuner():
    from iohblade.benchmarks.kerneltuner.kerneltuner import Kerneltuner
    kt = Kerneltuner()
    config = kt.get_config()
    assert check_key_words_exist_in(config)
#endregion

# region Logistics
def test_travelling_salesman_problem():
    from iohblade.benchmarks.logistics import TravelingSalesmanProblem
    tsp = TravelingSalesmanProblem('A-n53-k7')
    config = tsp.get_config()
    assert check_key_words_exist_in(config)

def test_vehicle_routing_problem():
    from iohblade.benchmarks.logistics import VehicleRoutingProblem
    vrp = VehicleRoutingProblem('A-n53-k7')
    config = vrp.get_config()
    assert check_key_words_exist_in(config)
#endregion

#region Matrix Multiplication
def test_mat_mul():
    from iohblade.benchmarks.matrix_multiplication import default_mat_mul_problems
    problem = default_mat_mul_problems()[0]
    config = problem.get_config()
    assert check_key_words_exist_in(config)
#endregion

#region Number Theory
def test_sums_vs_differences():
    from iohblade.benchmarks.number_theory import SumDifference
    sd = SumDifference()
    config = sd.get_config()
    assert check_key_words_exist_in(config)

#endregion

#region Packing
def test_circle_packing():
    from iohblade.benchmarks.packing import CirclePacking
    cp = CirclePacking()
    config = cp.get_config()
    assert check_key_words_exist_in(config)

def test_hexagonal_packing():
    from iohblade.benchmarks.packing import HexagonPacking
    hp = HexagonPacking(11)
    config = hp.get_config()
    assert check_key_words_exist_in(config)

def test_rectangle_packing():
    from iohblade.benchmarks.packing import RectanglePacking
    rp = RectanglePacking()
    config = rp.get_config()
    assert check_key_words_exist_in(config)

def test_unit_square_packing():
    from iohblade.benchmarks.packing import UnitSquarePacking
    usp = UnitSquarePacking(10)
    config = usp.get_config()
    assert check_key_words_exist_in(config)
#endregion

# region Photonics
# def test_photonics():
#     from iohblade.benchmarks.photonics.photonics import Photonics
#     p = Photonics(None)
#     assert check_key_words_exist_in(p.get_config())
#endregion