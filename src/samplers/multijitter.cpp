#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _sampler-multijitter:

Correlated Multi-Jittered sampler (:monosp:`multijitter`)
---------------------------------------------------------

.. pluginparameters::

 * - sample_count
   - |int|
   - Number of samples per pixel. This value might be slightly increased by the plugin to produce a
     strata grid with an aspect ratio close to one. (Default: 4)
 * - seed
   - |int|
   - Seed offset (Default: 0)
 * - jitter
   - |bool|
   - Adds additional random jitter withing the substratum (Default: True)

This plugin implements the methods introduced in Pixar's tech memo :cite:`kensler1967correlated`.

Unlike the previously described stratified sampler, multi-jittered sample patterns produce samples
that are well stratified in 2D but also well stratified when projected onto one dimension. This can
greatly reduce the variance of a Monte-Carlo estimator when the function to evaluate exhibits more
variation along one axis of the sampling domain than the other.

This sampler achieves this by first placing samples in a canonical arrangement that is stratified in
both 2D and 1D. It then shuffles the x-coordinate of the samples in every columns and the
y-coordinate in every rows. Fortunately, this process doesn't break the 2D and 1D stratification.
Kensler's method futher reduces sample clumpiness by correlating the shuffling applied to the
columns and the rows.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sampler_independent_16spp.jpg
   :caption: Independent sampler - 16 samples per pixel
.. subfigure:: ../../resources/data/docs/images/render/sampler_multijitter_16spp.jpg
   :caption: Correlated Multi-Jittered sampler - 16 samples per pixel
.. subfigend::
   :label: fig-multijitter-renders

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/sampler/multijitter_1024_samples.svg
   :caption: 1024 samples projected onto the first two dimensions.
.. subfigure:: ../../resources/data/docs/images/sampler/multijitter_64_samples_and_proj.svg
   :caption: 64 samples projected onto the first two dimensions and their projection on both 1D axis
             (top and right plot). As expected, the samples are well stratified both in 2D and 1D.
.. subfigend::
   :label: fig-multijitter-pattern

 */

template <typename Float, typename Spectrum>
class MultijitterSampler final : public PCG32Sampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PCG32Sampler, m_sample_count, m_base_seed, m_rng,
                    m_samples_per_wavefront, m_dimension_index,
                    current_sample_index, compute_per_sequence_seed)
    MTS_IMPORT_TYPES()

    MultijitterSampler(const Properties &props = Properties()) : Base(props) {
        m_jitter = props.bool_("jitter", true);

        // Find stratification grid resolution with aspect ratio close to 1
        m_resolution[1] = ScalarUInt32(sqrt(ScalarFloat(m_sample_count)));
        m_resolution[0] = (m_sample_count + m_resolution[1] - 1) / m_resolution[1];

        if (m_sample_count != hprod(m_resolution))
            Log(Warn, "Sample count rounded up to %i", hprod(m_resolution));

        m_sample_count = hprod(m_resolution);
        m_inv_sample_count = rcp(ScalarFloat(m_sample_count));
        m_inv_resolution   = rcp(ScalarPoint2f(m_resolution));
        m_resolution_x_div = m_resolution[0];
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        MultijitterSampler *sampler = new MultijitterSampler();
        sampler->m_jitter                = m_jitter;
        sampler->m_sample_count          = m_sample_count;
        sampler->m_inv_sample_count      = m_inv_sample_count;
        sampler->m_resolution            = m_resolution;
        sampler->m_inv_resolution        = m_inv_resolution;
        sampler->m_resolution_x_div      = m_resolution_x_div;
        sampler->m_samples_per_wavefront = m_samples_per_wavefront;
        sampler->m_base_seed             = m_base_seed;
        return sampler;
    }

    void seed(uint64_t seed_offset, size_t wavefront_size) override {
        Base::seed(seed_offset, wavefront_size);
        m_permutation_seed = compute_per_sequence_seed(seed_offset);
    }

    Float next_1d(Mask active = true) override {
        UInt32 sample_indices = current_sample_index();
        UInt32 perm_seed = m_permutation_seed + m_dimension_index++;

        Float p = kensler_permute(sample_indices, m_sample_count, perm_seed * 0x45fbe943, active);
        Float j = m_jitter ? m_rng.template next_float<Float>(active) : 0.5f;

        return (p + j) * m_inv_sample_count;
    }

    Point2f next_2d(Mask active = true) override {
        UInt32 sample_indices = current_sample_index();
        UInt32 perm_seed = m_permutation_seed + m_dimension_index++;

        UInt32 s = kensler_permute(sample_indices, m_sample_count, perm_seed * 0x51633e2d, active);

        UInt32 y = m_resolution_x_div(s);    // s / m_resolution.x()
        UInt32 x = s - y * m_resolution.x(); // s % m_resolution.x()

        UInt32 sx = kensler_permute(x, m_resolution.x(), perm_seed * 0x68bc21eb, active);
        UInt32 sy = kensler_permute(y, m_resolution.y(), perm_seed * 0x02e5be93, active);

        Float jx = 0.5f, jy = 0.5f;
        if (m_jitter) {
            jx = m_rng.template next_float<Float>(active);
            jy = m_rng.template next_float<Float>(active);
        }

        return Point2f((x + (sy + jx) * m_inv_resolution.y()) * m_inv_resolution.x(),
                       (y + (sx + jy) * m_inv_resolution.x()) * m_inv_resolution.y());
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MultijitterSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "  jitter = " << m_jitter << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_jitter;

    /// Stratification grid resolution and precomputed variables
    ScalarPoint2u m_resolution;
    ScalarPoint2f m_inv_resolution;
    ScalarFloat m_inv_sample_count;
    enoki::divisor<ScalarUInt32> m_resolution_x_div;

    /// Per-sequence permutation seed
    UInt32 m_permutation_seed;
};

MTS_IMPLEMENT_CLASS_VARIANT(MultijitterSampler, Sampler)
MTS_EXPORT_PLUGIN(MultijitterSampler, "Correlated Multi-Jittered Sampler");
NAMESPACE_END(mitsuba)