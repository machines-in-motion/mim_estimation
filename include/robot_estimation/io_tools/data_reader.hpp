/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Parser to load SL file format (clmcplot)
 * It loads the data and provide a usueful API for accessing the fields
 * Implement some function to access the data from the SL d-file.
 */

#pragma once

#include <Eigen/Eigen>

namespace robot_estimation
{
namespace io_tools
{
class DataReader
{
public:
    DataReader()
    {
    }
    ~DataReader()
    {
    }

    ///
    /// \brief read, reads a clmcplot file and fills the internal data
    /// structure. \param fname: name of the d-file to read.
    ///
    void read(const std::string& fname);

    ///
    /// \brief findIndex, find the column index corresponding to the field name.
    /// \param name: name of the field.
    /// \return the column index corresponding to the field name.
    ///
    int getIndex(const std::string& stream_name);

    ///
    /// \brief getIndexes:, find stream column indexes corresponding to the
    /// names. \param names: list of the stream name. \param indexes: the output
    /// list of indexes.
    ///
    void getIndexes(const std::vector<std::string>& stream_names,
                    std::vector<int>& indexes);

    ///
    /// \brief fillPose, fill an existing pose object with the data.
    /// \param row: row index homogenous to time/sampling_period.
    /// \param indexes: the indexes of the data streams to copy in order:
    /// [x, y, z, qw, qx, qy, qz].
    /// \param pose: the object to fill.
    ///
    void fillPose(int row,
                  const std::vector<int>& indexes,
                  Eigen::Vector3d& pose,
                  Eigen::Quaterniond& orientation);

    // ///
    // /// \brief fillTwist, fill an existing twist object with the data.
    // /// \param row: row index homogenous to time/sampling_period.
    // /// \param indexes: the indexes of the data streams to copy in order:
    // /// [v_x, v_y, v_z, w_x, w_y, w_z].
    // /// \param twist: the object to fill.
    // ///
    // void fillTwist(int row,
    //                const std::vector<int>& indexes,
    //                geometry_utils::SpatialMotionVector& twist);

    ///
    /// \brief fillVector, fill an existing vector from the data
    /// \param row: row index homogenous to time/sampling_period.
    /// \param indexes: the indexes of the data streams to copy.
    /// \param vec: the vector with the correct dimension that is filled
    ///
    void fillVector(int row,
                    const std::vector<int>& indexes,
                    Eigen::Ref<Eigen::VectorXd> vec);

    // ///
    // /// \brief fillWrench, fill an existing wrench from the data
    // /// \param row: row index homogenous to time/sampling_period.
    // /// \param indexes: the indexes of the data streams to copy in order:
    // /// [f_x, f_y, f_z, tau_x, tau_y, tau_z]
    // /// \param wrench
    // ///
    // void fillWrench(int row,
    //                 const std::vector<int>& index,
    //                 geometry_utils::SpatialForceVector& wrench);

    ///
    /// \brief getNbRows, get the number of rows equivalent to the number of
    /// time step saved in the d-file \return the number of time step saved
    ///
    inline int getNbRows()
    {
        return data_.rows();
    }

    ///
    /// \brief getValue, get the value of a specific stream at a specific time
    /// \param row: row index homogenous to time/sampling_period.
    /// \param index: the index of the data streams to copy from.
    /// \return the value
    ///
    inline double getValue(int row, int index)
    {
        return data_(row, index);
    }

    ///
    /// \brief getFrequence, get the getFrequence of the read data
    /// \return the frequence
    ///
    inline double getFrequency()
    {
        return frequency_;
    }

private:
    Eigen::MatrixXd data_;
    std::vector<std::string> var_names_;
    std::vector<std::string> units_;
    int nb_rows_;
    int nb_cols_;
    int buffer_size_;
    double frequency_;
};

}  // end namespace io_tools
}  // end namespace robot_estimation
