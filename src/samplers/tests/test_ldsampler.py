import mitsuba
import pytest
import enoki as ek
import numpy as np

from .utils import check_uniform_scalar_sampler, check_uniform_wavefront_sampler

def test01_ldsampler_scalar(variant_scalar_rgb):
    from mitsuba.core import xml

    sampler = xml.load_dict({
        "type" : "ldsampler",
        "sample_count" : 1024,
    })

    check_uniform_scalar_sampler(sampler)


def test02_ldsampler_wavefront(variant_gpu_rgb):
    from mitsuba.core import xml

    sampler = xml.load_dict({
        "type" : "ldsampler",
        "sample_count" : 1024,
    })

    check_uniform_wavefront_sampler(sampler)


def test03_ldsampler_deterministic_values(variant_scalar_rgb):
    from mitsuba.core import xml

    sampler = xml.load_dict({
        "type" : "ldsampler",
        "sample_count" : 1024,
    })

    sampler.seed(0)

    values_1d_dim0 = [0.1033020019, 0.0271301269, 0.8591613769, 0.3982238769, 0.6531066894,
                      0.9880676269, 0.7761535644, 0.2790832519, 0.4333801269, 0.3659973144]

    values_2d_dim0 = [[0.830387, 0.823217], [0.588199, 0.721654], [0.276676, 0.783178],
                      [0.762028, 0.23142], [0.92609, 0.457983], [0.430973, 0.378881],
                      [0.773746, 0.0322013], [0.552067, 0.835912], [0.311832, 0.748022],
                      [0.995426, 0.564428]]

    values_1d_dim1 = [0.9324035644, 0.5359191894, 0.2644348144, 0.8542785644, 0.2771301269,
                      0.4324035644, 0.2409973144, 0.6521301269, 0.8054504394, 0.9568176269]

    values_2d_dim1 = [[0.0237463, 0.282201], [0.469059, 0.211889], [0.818668, 0.881811],
                      [0.206363, 0.94724], [0.211246, 0.0947013], [0.655582, 0.841772],
                      [0.0305822, 0.966772], [0.245426, 0.814428], [0.82941, 0.574193],
                      [0.446598, 0.832006]]

    for v in values_1d_dim0:
        assert ek.allclose(sampler.next_1d(), v)

    for v in values_2d_dim0:
        assert ek.allclose(sampler.next_2d(), v)

    sampler.advance()

    for v in values_1d_dim1:
        assert ek.allclose(sampler.next_1d(), v)

    for v in values_2d_dim1:
        assert ek.allclose(sampler.next_2d(), v)
