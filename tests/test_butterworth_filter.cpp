#include <gtest/gtest.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "robot_estimation/filtering_tools/butterworth_filter.hpp"

using namespace standard_filters;

class FilterTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
    }
    virtual void TearDown()
    {
    }
};

TEST_F(FilterTest, butterworth_filter_test)
{
    Eigen::Matrix<double, 100, 1> data, ref_data, fil_data;
    data.setZero();
    ref_data.setZero();
    fil_data.setZero();
    data << 0.0889, -0.2031, -0.4984, -0.8569, -0.1009, -0.1142, 1.1651, 1.3278,
        1.6593, 1.6178, 1.3627, 1.2840, 0.0185, -0.1340, -0.0704, -0.5002,
        -0.0792, -0.3037, 0.6825, 1.3819, 0.9942, 1.4237, 0.9620, 0.8490,
        0.8426, 0.5585, -0.3821, -0.9387, -0.0675, -0.2744, 0.5717, 0.4774,
        1.2927, 0.9265, 1.5102, 1.1783, 0.8277, 0.9717, 0.2912, -0.4925,
        -0.3619, -0.5082, -0.0539, 0.7740, 0.9200, 0.8711, 1.8019, 1.7010,
        0.8604, 0.5087, 0.1567, -0.1707, -0.3768, -0.7233, -0.0234, 0.2933,
        0.2463, 1.0669, 0.9228, 1.6637, 1.2746, 1.3070, 0.4898, 0.3850, -0.3204,
        -0.5693, -0.7699, -0.0330, 0.4634, 1.0577, 1.1010, 1.0944, 1.6018,
        1.0543, 0.9178, 0.8897, -0.0139, 0.1096, -0.6196, -0.2323, -0.0559,
        -0.1026, 0.6990, 1.6324, 1.0141, 1.3628, 1.0041, 0.5923, 0.2350, 0.3447,
        -0.3948, -0.1960, -0.5106, -0.0566, 0.9053, 0.5678, 0.8810, 1.6977,
        1.8656, 0.7604;
    ref_data << 0.0889, 0.0869, 0.0707, 0.0099, -0.1279, -0.3242, -0.4727,
        -0.4212, -0.0746, 0.5173, 1.1619, 1.6281, 1.7733, 1.5851, 1.1407,
        0.5716, 0.0432, -0.3038, -0.4156, -0.3138, -0.0342, 0.3836, 0.8470,
        1.2089, 1.3493, 1.2539, 1.0090, 0.7087, 0.3728, -0.0043, -0.3419,
        -0.4827, -0.3261, 0.0811, 0.5797, 1.0017, 1.2523, 1.3198, 1.2377,
        1.0472, 0.7677, 0.4010, -0.0109, -0.3470, -0.4522, -0.2392, 0.2299,
        0.7754, 1.2270, 1.4894, 1.5072, 1.2579, 0.8020, 0.2802, -0.1615,
        -0.4296, -0.4805, -0.3146, 0.0114, 0.4049, 0.7852, 1.1035, 1.3212,
        1.3850, 1.2438, 0.8959, 0.4106, -0.0931, -0.4730, -0.5917, -0.3738,
        0.1231, 0.7092, 1.1708, 1.3949, 1.3930, 1.2379, 0.9979, 0.7052, 0.3702,
        0.0250, -0.2530, -0.3719, -0.2782, 0.0285, 0.4925, 0.9819, 1.3209,
        1.3882, 1.1812, 0.8045, 0.4018, 0.0739, -0.1519, -0.2773, -0.2731,
        -0.0910, 0.2561, 0.6759, 1.0725;
    double sampling_freq = 200.;
    double nyquist_freq = sampling_freq / 2.;
    double cutoff_freq = 30.;
    double cutoff = cutoff_freq / nyquist_freq;
    const int filter_order = 5;

    typedef ButterworthFilter<Eigen::Matrix<double, 4, 1>, filter_order>
        FilterType;
    FilterType::VecToFilter state, measure, estimate;
    state.setConstant(data(0));
    Eigen::Ref<FilterType::VecToFilter> ref_estimate(estimate);

    FilterType myfilter;
    myfilter.initialize(state, cutoff);

    for (int i = 0; i < 100; i++)
    {
        measure.setConstant(data(i));
        myfilter.update(measure);
        myfilter.getData(ref_estimate);
        fil_data(i) = ref_estimate(0);
    }

    double error = (fil_data - ref_data).norm();
    EXPECT_NEAR(error, 0.0, 0.01);
}
