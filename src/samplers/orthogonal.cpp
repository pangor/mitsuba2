#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-orthogonal:

Orthogonal Array sampler (:monosp:`orthogonal`)
-----------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel. This value has to be the square of a prime number. (Default: 4)
 * - strength
   - |int|
   - Orthogonal array's strength (Default: 2)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the substratum (Default: True)

This plugin implements the Orthogonal Array sampler generator introduced by
:cite:`jarosz19orthogonal`. It generalizes correlated multi-jittered sampling to higher dimensions
by using *orthogonal arrays (OAs)* of strength :math:`s`. The strength property of OAs tells that
projecting the generated samples to any combination of :math:`s` dimensions will always result in
a well stratified pattern. In other words, when :math:`s=2` (default value), the high-dimentional
samples are simultaneously stratified in all 2D projections as if they had been produced by
correlated multi-jittered sampling. By construction, samples produced by this generator are also
well stratified when projected on both 1D axis.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sampler_independent_25spp.jpg
   :caption: Independent sampler - 25 samples per pixel
.. subfigure:: ../../resources/data/docs/images/render/sampler_orthogonal_25spp.jpg
   :caption: Orthogonal Array sampler - 25 samples per pixel
.. subfigend::
   :label: fig-orthogonal-renders

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/sampler/orthogonal_1369_samples.svg
   :caption: 1369 samples projected onto the first two dimensions.
.. subfigure:: ../../resources/data/docs/images/sampler/orthogonal_49_samples_and_proj.svg
   :caption: 49 samples projected onto the first two dimensions and their
             projection on both 1D axis (top and right plot). The pattern is well stratified
             in both 2D and 1D projections. This is true for every pair of dimensions of the
             high-dimentional samples.
.. subfigend::
   :label: fig-orthogonal-pattern

 */

template <typename Float, typename Spectrum>
class OrthogonalSampler final : public PCG32Sampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PCG32Sampler, m_sample_count, m_base_seed, m_rng,
                    m_samples_per_wavefront, m_dimension_index,
                    current_sample_index, compute_per_sequence_seed)
    MTS_IMPORT_TYPES()

    OrthogonalSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);
        m_strength = props.int_("strength", 2);

        // Make sure m_resolution is a prime number
        auto is_prime = [](ScalarUInt32 x) {
            for (ScalarUInt32 i = 2; i <= x / 2; ++i)
                if (x % i == 0)
                    return false;
            return true;
        };

        m_resolution = 2;
        while (sqr(m_resolution) < m_sample_count || !is_prime(m_resolution))
            m_resolution++;

        if (m_sample_count != sqr(m_resolution))
            Log(Warn, "Sample count should be the square of a prime"
                "number, rounding to %i", sqr(m_resolution));

        m_sample_count = sqr(m_resolution);
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        OrthogonalSampler *sampler       = new OrthogonalSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_strength              = m_strength;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_resolution            = m_resolution;
        sampler->m_samples_per_wavefront = m_samples_per_wavefront;
        sampler->m_base_seed             = m_base_seed;
        return sampler;
    }

    void seed(uint64_t seed_offset, size_t wavefront_size) override {
        Base::seed(seed_offset, wavefront_size);
        m_permutation_seed = compute_per_sequence_seed(seed_offset);
    }

    Float next_1d(Mask active = true) override {
        if (unlikely(m_strength != 2)) {
            return bush(current_sample_index(),
                        m_dimension_index++,
                        m_resolution,
                        m_strength,
                        m_permutation_seed, active);
        } else {
            return bose(current_sample_index(),
                        m_dimension_index++,
                        m_resolution,
                        m_permutation_seed, active);
        }
    }

    Point2f next_2d(Mask active = true) override {
        Float f1 = next_1d(active),
              f2 = next_1d(active);
        return Point2f(f1, f2);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "OrthogonalSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  jitter = " << m_jitter << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()

private:
    /// Compute the digits of decimal value \ref i expressed in base \ref b
    std::vector<UInt32> to_base_s(UInt32 i, ScalarUInt32 b, ScalarUInt32 t) {
        std::vector<UInt32> digits(t);
        for (size_t ii = 0; ii < t; i /= b, ++ii)
            digits[ii] = i % b;
        return digits;
    }

    /// Evaluate polynomial with coefficients \ref x at location arg
    UInt32 eval_poly(const std::vector<UInt32> &coef, UInt32 x) {
        UInt32 res = 0;
        for (size_t l = coef.size(); l--; )
            res = (res * x) + coef[l];
        return res;
    }

    /// Bush construction technique for orthogonal arrays
    Float bush(UInt32 i,       // sample index
               ScalarUInt32 j, // dimension
               ScalarUInt32 s, // number of levels/stratas (has to be a prime number)
               ScalarUInt32 t, // strength of OA
               UInt32 p,       // pseudo-random permutation seed
               Mask active = true) {
        ScalarUInt32 N = enoki::pow(s, t);
        i = kensler_permute(i, N, p, active);
        auto i_digits = to_base_s(i, s, t);
        ScalarUInt32 stm = N / s;
        UInt32 phi = eval_poly(i_digits, j);

        // Multi-jitter flavor
        UInt32 stratum = kensler_permute(phi % s, s, p * (j + 1) * 0x51633e2d, active);
        UInt32 sub_stratum = kensler_permute((i / s) % stm, stm, p * (j + 1) * 0x68bc21eb, active);

        Float jitter = m_jitter ? m_rng.template next_float<Float>(active) : 0.5f;
        return (stratum + (sub_stratum + jitter) / stm) / s;
    }

    /// Bose construction technique for orthogonal arrays. It only support OA of strength == 2
    Float bose(UInt32 i,       // sample index
               ScalarUInt32 j, // dimension
               ScalarUInt32 s, // number of levels/stratas (has to be a prime number)
               UInt32 p,       // pseudo-random permutation seed
               Mask active = true) {

        // Permutes the sample index so that samples are obtained in random order
        i = kensler_permute(i % (s * s), s * s, p, active);

        // TODO could use enoki::divisor here
        UInt32 a_i0 = i / s;
        UInt32 a_i1 = i % s;

        UInt32 a_ij, a_ik;
        if (j == 0) {
            a_ij = a_i0;
            a_ik = a_i1;
        } else if (j == 1) {
            a_ij = a_i1;
            a_ik = a_i0;
        } else {
            UInt32 k = (j % 2) ? j - 1 : j + 1;
            a_ij = (a_i0 + (j - 1) * a_i1) % s;
            a_ik = (a_i0 + (k - 1) * a_i1) % s;
        }

        // Correlated multi-jitter flavor
        UInt32 stratum = kensler_permute(a_ij, s, p * (j + 1) * 0x51633e2d, active);
        UInt32 sub_stratum = kensler_permute(a_ik, s, p * (j + 1) * 0x68bc21eb, active);

        Float jitter = m_jitter ? m_rng.template next_float<Float>(active) : 0.5f;
        return (stratum + (sub_stratum + jitter) / s) / s;
    }

private:
    bool m_jitter;
    ScalarUInt32 m_strength;

    /// Stratification grid resolution
    ScalarUInt32 m_resolution;

    /// Per-sequence permutation seed
    UInt32 m_permutation_seed;
};

MTS_IMPLEMENT_CLASS_VARIANT(OrthogonalSampler, Sampler)
MTS_EXPORT_PLUGIN(OrthogonalSampler, "Orthogonal Array Sampler");
NAMESPACE_END(mitsuba)
