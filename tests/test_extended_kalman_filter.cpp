#include <gtest/gtest.h>
#include <stdlib.h>
#include <random>

#include "mim_estimation/io_tools/data_reader.hpp"
#include "mim_estimation/io_tools/non_rt_data_collector.hpp"
#include "simple_1d_ekf.hpp"

class TestEKF : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
    }
    virtual void TearDown()
    {
    }
};

/**
 * @brief generate_1d_noisy_sinus: for documentation purpose on the generated
 * data for the unittest
 * @param noise_gen: a random generator
 * @param noise: a gaussian noise
 * @param amplitude: sinus amplitude
 * @param pulsation: sinus pulsation
 * @param time: the current time in seconds
 * @return a noisy sinusoid
 */
double generate_1d_noisy_sinus(
    std::default_random_engine noise_gen,
    std::normal_distribution<double> noise, /**{0, 0.5}*/
    double amplitude,
    double pulsation,
    double time)
{
    return amplitude * sin(pulsation * time) + noise(noise_gen);
}

/**
 * @brief This test evaluates a simple 1d ekf for debugging purpose
 */
TEST_F(TestEKF, simple_1d_ekf_test)
{
    // create variables
    mim_estimation::io_tools::NonRtDataCollector dc;
    mim_estimation::io_tools::DataReader dr;
    Eigen::Matrix<double, 1, 1> input_pos;
    double output_pos, output_pos_ref;
    bool is_discrete = true;
    double dt = 0.0;
    int input_pose_id = 0;
    int output_pose_id = 0;
    double t = 0.0;

    // read the data from the file
    dr.read(std::string(CONFIG_PATH) + std::string("/simple_1d_ekf.dat"));

    // initialize the variables
    input_pos.setZero();
    output_pos = 0.0;
    output_pos_ref = 0.0;
    is_discrete = true;
    dt = 1.0 / dr.getFrequency();
    input_pose_id = dr.getIndex("input_pos");
    output_pose_id = dr.getIndex("state_pos");

    // create the class to test
    SimpleEKF ekf_1d(0.0, dt, is_discrete);

    // subscribe to the data_collector
    dc.addVariable(&t, "time", "s");
    dc.addVariable(input_pos.data(), "input_pos", "-");
    ekf_1d.subscribe_to_data_collector(dc);

    // start the data_collection and collect te initial state
    dc.startDataCollection();
    dc.updateDataCollection();

    // compute the ekf and check the output
    for (unsigned i = 1; i < dr.getNbRows(); ++i)
    {
        // simply keep track of the time
        t = double(i) * dt;
        // get the input signal from file (see generate_1d_noisy_sinus)
        input_pos(0, 0) = dr.getValue(i, input_pose_id);
        ekf_1d.update(input_pos, true);
        dc.updateDataCollection();
        output_pos = ekf_1d.getFilterState()(0, 0);
        output_pos_ref = dr.getValue(i, output_pose_id);
        ASSERT_NEAR(output_pos, output_pos_ref, 1e-6);
    }

    // stop collecting data and dump it
    dc.stopDataCollection();

    // this was used for debugging the test. I intentionaly leave it here.
    // in case the test FAILS one day, one can dump the results of the ekf.
    // dc.dump(std::string(TEST_CONFIG_PATH) + std::string("ekf_1d_2.out"));
}
