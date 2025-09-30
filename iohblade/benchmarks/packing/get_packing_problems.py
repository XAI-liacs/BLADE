from . import HexagonPacking, RectanglePacking, UnitSquarePacking


def get_hexagon_packing_problems() -> list[HexagonPacking]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Hexagon Packing benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Hexagon Packing benchmark instance with 11 inner hexagons.
            array[1] = Hexagon Packing benchmark instance with 12 inner hexagons.
    """

    hp1 = HexagonPacking(n_hex=11, best_known=3.931)
    hp2 = HexagonPacking(n_hex=12, best_known=3.942)

    return [hp1, hp2]


def get_rectangle_packing_problems() -> list[RectanglePacking]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Rectangle Packing benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Rectangle Packing benchmark with rectangle perimeter = 4, packing 21 circles.
    """

    rp1 = RectanglePacking()

    return [rp1]


def get_square_packing_problems() -> list[UnitSquarePacking]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Unit Square Packing benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Unit Square Packing benchmark for Unit Square, packing 26 circles.
            array[1] = Unit Square Packing benchmark for Unit Square, packing 32 circles.
    """

    rp1 = UnitSquarePacking(n_circles=26, best_known=2.635)
    rp2 = UnitSquarePacking(n_circles=32, best_known=2.937)

    return [rp1, rp2]
