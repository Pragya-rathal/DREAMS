from importlib.util import module_from_spec, spec_from_file_location

import numpy as np


spec = spec_from_file_location("embeddings", "dreamsApp/app/utils/embeddings.py")
embeddings = module_from_spec(spec)
spec.loader.exec_module(embeddings)
l2_normalize = embeddings.l2_normalize


def test_l2_normalize_normal_vector():
    vec = np.array([3.0, 4.0])
    normalized = l2_normalize(vec)
    assert np.allclose(normalized, np.array([0.6, 0.8]))


def test_l2_normalize_zero_vector():
    vec = np.zeros(3)
    normalized = l2_normalize(vec)
    assert np.array_equal(normalized, vec)
