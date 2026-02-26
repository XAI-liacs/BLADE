# We try catch each problem here because we don't want to always load all dependencies, but still provide the names at top level.
try:
    from .automl.automl import AutoML
except Exception:
    AutoML = None
try:
    from .BBOB.bbob_sboxcost import BBOB_SBOX
except Exception:
    BBOB_SBOX = None
# try:
#     from .hlp import HLP
# except Exception:
#     HLP = None
try:
    from .kerneltuner.kerneltuner import Kerneltuner
except Exception:
    Kerneltuner = None
try:
    from .BBOB.mabbob import MA_BBOB
except Exception:
    MA_BBOB = None

try:
    from .photonics.photonics import Photonics
except Exception:
    Photonics = None
