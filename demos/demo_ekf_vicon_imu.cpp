/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Simple demo that runs the EkfViconImu filter using a d-file as data.
 */

#include "mim_estimation/demos_and_tests/ekf_vicon_imu_simple.hpp"

using namespace mim_estimation;

///
/// \brief main, simple main that execute the estimation of the base position
/// from motion capture and IMU data saved from SL
/// \param argc: must be equal to 2
/// \param argv: contains the path to the data-file
/// \return 0 by default
///
int main(int argc, char **argv)
{
    /* *** get the arguments of main *** */
    if (argc != 3)
    {
        std::stringstream help_str;
        help_str << "Must have 2 argument:" << std::endl
                 << "1- The path to the config.yaml file" << std::endl
                 << "2- The path to the d-file";
        throw std::runtime_error(help_str.str());
    }
    std::string yaml_file, d_file;
    yaml_file = argv[1];
    d_file = argv[2];

    /* *** display the 2 inputs *** */
    std::cout << "Get the estimator parameters from " << yaml_file << std::endl;
    std::cout << "Get the data from " << d_file << std::endl;

    /* *** create the object to demonstrate *** */
    mim_estimation::test::EstimatorViconImuTest evi_test(yaml_file, d_file);
    evi_test.run();
    evi_test.display_all_statistics();
    evi_test.dump("/tmp/output_demo_estimator.dfile");

    return 0;
}
