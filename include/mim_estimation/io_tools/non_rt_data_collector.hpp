/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Collecting data in a non real-time inside a specific library
 * and dump it in the SL format.
 */

#pragma once

#include <deque>
#include <string>
#include "mim_estimation/io_tools/data_collector.hpp"

namespace mim_estimation
{
namespace io_tools
{
class NonRtDataCollector : public DataCollector
{
public:
    NonRtDataCollector()
    {
        stopDataCollection();
    }
    virtual ~NonRtDataCollector()
    {
    }

    //! virtual functions to add variables to data collection for different
    //! types
    virtual void addVariable(const double* data,
                             const std::string& name,
                             const std::string& units);

    virtual void addVariable(const float* data,
                             const std::string& name,
                             const std::string& units);

    virtual void addVariable(const int* data,
                             const std::string& name,
                             const std::string& units);

    virtual void addVariable(const bool* data,
                             const std::string& name,
                             const std::string& units);

    //! virtual function to update the collection with the recently added
    //! variables
    virtual void updateDataCollection();
    virtual void stopDataCollection();
    virtual void startDataCollection();

    //! virtual function to check whether data collection has completed:
    virtual bool isDataCollectionDone();

    //! convertion big endian and little endian
    float reverseFloat(const float inFloat);

    //! dump the data in SL format the
    void dump(std::string path);

private:
    // manage doubles
    std::deque<std::deque<double> > double_data_;
    std::deque<std::string> double_names_;
    std::deque<std::string> double_units_;
    std::deque<const double*> double_ptr_;
    // manage floats
    std::deque<std::deque<float> > float_data_;
    std::deque<std::string> float_names_;
    std::deque<std::string> float_units_;
    std::deque<const float*> float_ptr_;
    // manage integers
    std::deque<std::deque<int> > int_data_;
    std::deque<std::string> int_names_;
    std::deque<std::string> int_units_;
    std::deque<const int*> int_ptr_;
    // manage booleans
    std::deque<std::deque<bool> > bool_data_;
    std::deque<std::string> bool_names_;
    std::deque<std::string> bool_units_;
    std::deque<const bool*> bool_ptr_;
    bool running_;
};

}  // end namespace io_tools
}  // end namespace mim_estimation
