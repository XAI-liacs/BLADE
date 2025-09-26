import pytest
from iohblade.misc.prepare_namespace import prepare_namespace, clean_local_namespace, _add_builtins_into

def test_prepare_namespace_imports_all():
    # Test 1: All libraries are available.
    test1 = {
        "code": """import numpy as np
import random
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math"]
    }

    soln = prepare_namespace(test1["code"], test1["allowed"])
    print(soln)
    expected_allowed_list = test1["namespace_keys"]
    _add_builtins_into(expected_allowed_list)
    for key in soln:
        assert key in expected_allowed_list
    
def test_prepare_namespace_rejects_not_white_listed():
    #Test 2 All libraries are not avaialble: Raise Error.
    test2 = {
        "code": """import numpy as np
import random
import scipy
import math""",
        "allowed" : ["numpy>=0.1.0"],
        "namespace_keys": ["np", "random", "math"]
    }
    with pytest.raises(ImportError, match=f"Import of scipy not allowed"):
        _ = prepare_namespace(test2["code"], test2["allowed"])

def test_clean_local_namespace_generates_purely_local_namespace():
    global_ns = {
        "imported_library_a" : object,
        "imported_library_b" : object,
        "imported_library_c" : object
    }

    # Simulating exec returns local_ns += global_ns
    local_ns = {
        "imported_library_a" : object,
        "imported_library_b" : object,
        "imported_library_c" : object,
        "instantiated_object": object,
        "local_list" : [1, 2, 34, 101]
    }

    local_ns2 = clean_local_namespace(local_ns, global_ns)
    for key in global_ns:
        assert key not in local_ns2
        assert key not in local_ns #Reference semantics.
