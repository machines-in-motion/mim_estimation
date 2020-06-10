/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implementation of the DataReader class.
 */

#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "robot_estimation/io_tools/data_reader.hpp"

#define LLSB(x) ((x)&0xff) /*!< 32bit word byte/word swap macros */
#define LNLSB(x) (((x) >> 8) & 0xff)
#define LNMSB(x) (((x) >> 16) & 0xff)
#define LMSB(x) (((x) >> 24) & 0xff)

#define LONGSWAP(x) \
    ((LLSB(x) << 24) | (LNLSB(x) << 16) | (LNMSB(x) << 8) | (LMSB(x)))

namespace robot_estimation
{
namespace io_tools
{
void DataReader::read(const std::string& fname)
{
    /// open the file, and parse the parameters
    std::ifstream data_file;
    data_file.open(fname.c_str(), std::ifstream::in);
    if (!data_file.is_open()) throw std::runtime_error("data file not open");

    /// get the number of rows, columns, sampling frequency and calculate
    /// the buffer_size
    {
        std::string header;
        std::getline(data_file, header);
        std::istringstream header_stream(header);
        header_stream >> buffer_size_ >> nb_cols_ >> nb_rows_ >> frequency_;
        //    std::cout << "buffer size " << buffer_size_ << " with " <<
        //    nb_cols_
        //              << " cols and " << nb_rows_ << " rows\n" << "sampling
        //              freq "
        //              << frequency_ << std::endl;
    }

    var_names_.resize(nb_cols_);
    units_.resize(nb_cols_);

    /// loading the header
    {
        std::string header;
        std::getline(data_file, header);
        std::istringstream header_stream(header);
        for (int i = 0; i < nb_cols_; ++i)
        {
            header_stream >> var_names_[i] >> units_[i];
            //      std::cout << "varnames =" << var_names_[i]
            //                << " ; units = " << units_[i] << std::endl ;
        }
    }

    /// Read file into a buffer.
    //  std::cout << "Read file into a buffer and check if the matrix size "
    //            << "is correct." << std::endl;

    std::vector<float> buff(buffer_size_);
    float* buff_ptr = &buff[0];

    //  std::cout << "buffer created" << std::endl;
    data_file.read(reinterpret_cast<char*>(buff_ptr),
                   sizeof(float) * buffer_size_);

    /// Convert little-endian to big-endian
    //  std::cout << "convert little-endian to big-endian" << std::endl;
    for (int j = 0; j < nb_cols_; ++j)
    {
        for (int i = 0; i < nb_rows_; ++i)
        {
            auto aux = LONGSWAP(*((int*)&(buff_ptr[i + j * nb_rows_])));
            buff_ptr[i + j * nb_rows_] = *((float*)&aux);
        }
    }
    data_file.close();

    /// Transpose the data to get each column as a different stream of data.
    //  std::cout << "Transpose the data to get each column as a different "
    //            << "stream of data." << std::endl;
    Eigen::MatrixXf t =
        Eigen::Map<Eigen::MatrixXf>(buff_ptr, nb_cols_, nb_rows_).transpose();
    data_.resize(nb_rows_, nb_cols_);
    data_ = t.cast<double>();
}

int DataReader::getIndex(const std::string& name)
{
    for (int i = 0; i < var_names_.size(); ++i)
        if (var_names_[i] == name) return i;
    throw std::runtime_error("cannot find " + name + " in data");
}

void DataReader::getIndexes(const std::vector<std::string>& stream_names,
                            std::vector<int>& indexes)
{
    indexes.resize(stream_names.size());
    for (int i = 0; i < stream_names.size(); ++i)
    {
        indexes[i] = getIndex(stream_names[i]);
    }
}

void DataReader::fillPose(int row,
                          const std::vector<int>& index,
                          Eigen::Vector3d& pose,
                          Eigen::Quaterniond& orientation)
{
    assert(index.size() == 7);
    if (index.size() != 7)
    {
        std::stringstream error;
        error << "The number the index vector (" << index.size()
              << ") does not match the size of a translation + a quaternion "
                 "(3+4)";
        throw std::runtime_error(error.str());
    }
    pose = Eigen::Vector3d(
        data_(row, index[0]), data_(row, index[1]), data_(row, index[2]));
    orientation.x() = data_(row, index[3]);
    orientation.y() = data_(row, index[4]);
    orientation.z() = data_(row, index[5]);
    orientation.w() = data_(row, index[6]);
}

// void DataReader::fillTwist(int row,
//                            const std::vector<int>& index,
//                            geometry_utils::SpatialMotionVector& twist)
// {
//     assert(index.size() == 6);
//     if (index.size() != 6)
//     {
//         std::stringstream error;
//         error << "The number the index vector (" << index.size()
//               << ") does not match the size of a twist (6)";
//         throw std::runtime_error(error.str());
//     }
//     twist.v() = Eigen::Vector3d(
//         data_(row, index[0]), data_(row, index[1]), data_(row, index[2]));
//     twist.w() = Eigen::Vector3d(
//         data_(row, index[3]), data_(row, index[4]), data_(row, index[5]));
// }

void DataReader::fillVector(int row,
                            const std::vector<int>& index,
                            Eigen::Ref<Eigen::VectorXd> vec)
{
    assert(index.size() == vec.size());
    if (index.size() != vec.size())
    {
        std::stringstream error;
        error << "The number the index vector (" << index.size()
              << ") does not match the size of the given vector (" << vec.size()
              << ")";
        throw std::runtime_error(error.str());
    }
    for (int i = 0; i < vec.size(); ++i) vec(i) = data_(row, index[i]);
}

// void DataReader::fillWrench(int row,
//                             const std::vector<int>& index,
//                             geometry_utils::SpatialForceVector& wrench)
// {
//     assert(index.size() == 6);
//     if (index.size() != 6)
//     {
//         std::stringstream error;
//         error << "The number the index vector (" << index.size()
//               << ") does not match the size of a wrench (6)";
//         throw std::runtime_error(error.str());
//     }
//     wrench.f() = Eigen::Vector3d(
//         data_(row, index[0]), data_(row, index[1]), data_(row, index[2]));
//     wrench.n() = Eigen::Vector3d(
//         data_(row, index[3]), data_(row, index[4]), data_(row, index[5]));
// }

}  // end namespace io_tools
}  // end namespace robot_estimation
