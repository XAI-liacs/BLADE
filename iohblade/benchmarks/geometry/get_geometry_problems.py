from iohblade.benchmarks.geometry import (
    HeilbronnConvexRegion,
    HeilbronnTriangle,
    KissingNumber11D,
    MinMaxMinDistanceRatio,
)


def get_heilbronn_convex_region_problems() -> list[HeilbronnConvexRegion]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Heilbronn Convex Region benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Heilbronn Convex Region 13 points.
            array[1] = Heilbronn Convex Region 14 points.

    """
    hcr1 = HeilbronnConvexRegion(13, 0.0309)
    hcr2 = HeilbronnConvexRegion(14, 0.0278)

    return [hcr1, hcr2]


def get_heilbronn_triangle_problems() -> list[HeilbronnTriangle]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Heilbronn Triangle benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Heilbronn Triangle 11 points.

    """
    ht1 = HeilbronnTriangle(11, 0.0365)

    return [ht1]


def get_kissing_number_11D_problems() -> list[KissingNumber11D]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Kissing Number 11D benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Kissing Number 11 Dimensions.

    """

    kn = KissingNumber11D()
    return [kn]


def get_min_max_dist_ratio_problem() -> list[MinMaxMinDistanceRatio]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Min Dist / Max Distance benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Min Max Distance Ratio 2 Dimensions.
            array[1] = Min Max Distance Ratio 3 Dimensions.
    """

    min_max_2D = MinMaxMinDistanceRatio(16, 2, 12.889)
    min_max_3D = MinMaxMinDistanceRatio(14, 3, 4.1658)

    return [min_max_2D, min_max_3D]
