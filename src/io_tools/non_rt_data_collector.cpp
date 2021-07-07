/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implementation of the NonRtDataCollector class.
 */

#include "mim_estimation/io_tools/non_rt_data_collector.hpp"
#include <assert.h>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace mim_estimation
{
namespace io_tools
{
void NonRtDataCollector::addVariable(const double* data,
                                     const std::string& name,
                                     const std::string& units)
{
    if (running_) stopDataCollection();

    if (std::find(double_names_.begin(), double_names_.end(), name) ==
        double_names_.end())
    {
        double_ptr_.push_back(data);
        double_names_.push_back(name);
        double_units_.push_back(units);
        double_data_.push_back(std::deque<double>());
    }
}

void NonRtDataCollector::addVariable(const float* data,
                                     const std::string& name,
                                     const std::string& units)
{
    if (running_) stopDataCollection();

    if (std::find(float_names_.begin(), float_names_.end(), name) ==
        float_names_.end())
    {
        float_ptr_.push_back(data);
        float_names_.push_back(name);
        float_units_.push_back(units);
        float_data_.push_back(std::deque<float>());
    }
}

void NonRtDataCollector::addVariable(const int* data,
                                     const std::string& name,
                                     const std::string& units)
{
    if (running_) stopDataCollection();

    if (std::find(int_names_.begin(), int_names_.end(), name) ==
        int_names_.end())
    {
        int_ptr_.push_back(data);
        int_names_.push_back(name);
        int_units_.push_back(units);
        int_data_.push_back(std::deque<int>());
    }
}

void NonRtDataCollector::addVariable(const bool* data,
                                     const std::string& name,
                                     const std::string& units)
{
    if (running_) stopDataCollection();

    if (std::find(bool_names_.begin(), bool_names_.end(), name) ==
        bool_names_.end())
    {
        bool_ptr_.push_back(data);
        bool_names_.push_back(name);
        bool_units_.push_back(units);
        bool_data_.push_back(std::deque<bool>());
    }
}

//! virtual function to update the collection with the recently added variables
void NonRtDataCollector::updateDataCollection()
{
    if (!running_)
    {
        return;
    }
    unsigned index = 0;
    for (unsigned i = 0; i < double_ptr_.size(); ++i)
    {
        double_data_[i].push_back(*double_ptr_[i]);
    }
    index += double_ptr_.size();
    for (unsigned i = 0; i < float_ptr_.size(); ++i)
    {
        float_data_[i].push_back(double(*float_ptr_[i]));
    }
    index += float_ptr_.size();
    for (unsigned i = 0; i < int_ptr_.size(); ++i)
    {
        int_data_[i].push_back(double(*int_ptr_[i]));
    }
    index += int_ptr_.size();
    for (unsigned i = 0; i < bool_ptr_.size(); ++i)
    {
        bool_data_[i].push_back(double(*bool_ptr_[i]));
    }
}

void NonRtDataCollector::stopDataCollection()
{
    running_ = false;
}

void NonRtDataCollector::startDataCollection()
{
    running_ = true;
}

//! virtual function to check whether data collection has completed:
bool NonRtDataCollector::isDataCollectionDone()
{
    return !running_;
}

// dump the file in the SL format
void NonRtDataCollector::dump(std::string path)
{
    // std::cout << "compute the buffer size" << std:: endl;
    int buffer_size = 0;
    int nb_rows = 0;
    int nb_cols = 0;
    nb_cols = double_ptr_.size() + float_ptr_.size() + int_ptr_.size() +
              bool_ptr_.size();
    if (double_data_.size() > 0)
    {
        nb_rows = double_data_[0].size();
    }
    else if (float_data_.size() > 0)
    {
        nb_rows = float_data_[0].size();
    }
    else if (int_data_.size() > 0)
    {
        nb_rows = int_data_[0].size();
    }
    else if (bool_data_.size() > 0)
    {
        nb_rows = bool_data_[0].size();
    }
    else
    {
        std::cout << "nothing to dump, nothing done" << std::endl;
    }
    buffer_size = nb_cols * nb_rows;

    // std::cout << "create the buffer of data" << std:: endl;
    std::vector<float> buff(buffer_size);
    float* buff_ptr = &buff[0];

    // std::cout << "fill the buffer" << std:: endl;
    unsigned index = 0;
    for (unsigned i = 0; i < double_ptr_.size(); ++i)
    {
        for (unsigned j = 0; j < double_data_[i].size(); ++j)
        {
            buff_ptr[(index + i) + j * nb_cols] = float(double_data_[i][j]);
        }
    }
    index += double_ptr_.size();
    for (unsigned i = 0; i < float_ptr_.size(); ++i)
    {
        for (unsigned j = 0; j < float_data_[i].size(); ++j)
        {
            buff_ptr[(index + i) + j * nb_cols] = float(float_data_[i][j]);
        }
    }
    index += float_ptr_.size();
    for (unsigned i = 0; i < int_ptr_.size(); ++i)
    {
        for (unsigned j = 0; j < int_data_[i].size(); ++j)
        {
            buff_ptr[(index + i) + j * nb_cols] = float(int_data_[i][j]);
        }
    }
    index += int_ptr_.size();
    for (unsigned i = 0; i < bool_ptr_.size(); ++i)
    {
        for (unsigned j = 0; j < bool_data_[i].size(); ++j)
        {
            buff_ptr[(index + i) + j * nb_cols] = float(bool_data_[i][j]);
        }
    }
    index += bool_ptr_.size();
    assert((static_cast<int>(index)*nb_rows) == buffer_size);

    // std::cout << "convert the buffer to little endian" << std:: endl;
    for (int i = 0; i < buffer_size; ++i)
    {
        buff_ptr[i] = reverseFloat(buff_ptr[i]);
    }

    // std::cout << "open the file" << std:: endl;
    std::ofstream data_file;
    data_file.open(path.c_str(), std::ofstream::out);

    // std::cout << "dump the header" << std:: endl;
    double frequence = 0.001;
    data_file << buffer_size << " " << nb_cols << " " << nb_rows << " "
              << frequence << " ";
    data_file << std::endl;

    for (unsigned i = 0; i < double_ptr_.size(); ++i)
    {
        data_file << double_names_[i] << " " << double_units_[i] << " ";
    }
    for (unsigned i = 0; i < float_ptr_.size(); ++i)
    {
        data_file << float_names_[i] << " " << float_units_[i] << " ";
    }
    for (unsigned i = 0; i < int_ptr_.size(); ++i)
    {
        data_file << int_names_[i] << " " << int_units_[i] << " ";
    }
    for (unsigned i = 0; i < bool_ptr_.size(); ++i)
    {
        data_file << bool_names_[i] << " " << bool_units_[i] << " ";
    }
    // I am not sure why I should add 3 characters here
    data_file << std::endl;

    // std::cout << "dump the data" << std:: endl;
    data_file.write(reinterpret_cast<char*>(buff_ptr),
                    sizeof(float) * buffer_size);

    data_file.close();
}

float NonRtDataCollector::reverseFloat(const float inFloat)
{
    float retVal;
    char* floatToConvert = (char*)&inFloat;
    char* returnFloat = (char*)&retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

}  // end namespace io_tools
}  // end namespace mim_estimation
