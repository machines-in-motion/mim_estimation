/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Abstract interface DataCollector to collect data inside a specific
 * library
 */

#pragma once

#include <Eigen/Eigen>
#include <string>
#include <vector>

namespace mim_estimation
{
namespace io_tools
{
class DataCollector
{
public:
    DataCollector()
    {
    }
    virtual ~DataCollector()
    {
    }

    //! virtual functions to add variables to data collection for different
    //! types
    virtual void addVariable(const double* data,
                             const std::string& name,
                             const std::string& units) = 0;

    virtual void addVariable(const float* data,
                             const std::string& name,
                             const std::string& units) = 0;

    virtual void addVariable(const int* data,
                             const std::string& name,
                             const std::string& units) = 0;

    //! virtual function to update the collection with the recently added
    //! variables
    virtual void updateDataCollection() = 0;
    virtual void stopDataCollection() = 0;
    virtual void startDataCollection() = 0;

    //! virtual function to check whether data collection has completed:
    virtual bool isDataCollectionDone() = 0;

    /// helper functions to unroll eigen vectors into simpler types that can
    /// be handled by the addVariable functions

    //! arbitrary Eigen Vectors
    void addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                   const std::vector<std::string>& name,
                   const std::vector<std::string>& units);

    //! extension contains the list of strings to be appended to name for each
    //! elem of data
    void addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                   const std::string& name,
                   const std::vector<std::string>& extension,
                   const std::vector<std::string>& units);

    //! will just add 0,1,2..etc as extension to name
    void addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                   const std::string& name,
                   const std::string& unit);

    //! the actual recorded names will be name_x, name_y, name_z
    void addVector3d(const Eigen::Ref<const Eigen::Vector3d>& data,
                     const std::string& name,
                     const std::string& units);

    //! the actual recorded names will be name_x, name_y, name_z plus extension
    void addVector3d(const Eigen::Ref<const Eigen::Vector3d>& data,
                     const std::string& name,
                     const std::string& units,
                     const std::string& extension);

    //! the actual recorded names will be name_q0, name_q1, name_q2, name_q3
    //! with units "-"
    void addQuaternion(const Eigen::Ref<const Eigen::Vector4d>& data,
                       const std::string& name);

    //! the actual recorded names will be name_q0, name_q1, name_q2, name_q3
    //! with units "-"
    void addQuaternion(const Eigen::Quaterniond& data,
                       const std::string& name);

    //! the actual recorded names will have extension x,y,z,a,b,g
    void addVector6d(const Eigen::Ref<const Eigen::Matrix<double, 6, 1> >& data,
                     const std::string& name,
                     const std::string& units);

    //! arbitrary Eigen matrices (will just add matrix indes as extension to
    //! name, eg 11, 12, 21, 22
    void addMatrix(const Eigen::Ref<const Eigen::MatrixXd>& data,
                   const std::string& name,
                   const std::string& unit);

    //! the recorded names will have extensions _1_1, _2_1, ... _4_4
    //  void addMatrix4x4(const Eigen::Ref<const Eigen::Matrix<double, 4, 4> >&
    //  data,
    //                    const std::string& name,
    //                    const std::string& units);
};

}  // end namespace io_tools
}  // end namespace mim_estimation
