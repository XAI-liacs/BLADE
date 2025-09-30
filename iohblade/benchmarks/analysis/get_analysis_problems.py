from typing import Optional
from iohblade.benchmarks.analysis import AutoCorrIneq1, AutoCorrIneq2, AutoCorrIneq3
from .auto_correlation_base_spec import AutoCorrBaseSpec


def get_analysis_problems() -> list[AutoCorrBaseSpec]:
    """
    `get_x_problems` returns the whole set of said benchmark category. Here it returns Auto Correlation Inequality 1-3 benchamarks, as an array.

    Args:
        None

    Returns:
        An array of benchmark objects as follows:
            array[0] = Auto Correlation Inrquality 1
            array[1] = Auto Correlation Inrquality 2
            array[2] = Auto Correlation Inrquality 3

    """
    ac1 = AutoCorrIneq1()
    ac2 = AutoCorrIneq2()
    ac3 = AutoCorrIneq3()

    return [ac1, ac2, ac3]
