/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implementation of the DataCollector class.
 */

#include "robot_estimation/io_tools/data_collector.hpp"

#include <sstream>

namespace robot_estimation
{
namespace io_tools
{
void DataCollector::addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                              const std::vector<std::string>& name,
                              const std::vector<std::string>& units)
{
    assert(data.size() <= name.size());
    assert(data.size() <= units.size());

    for (int i = 0; i < data.size(); ++i)
    {
        addVariable((double*)&(data.data()[i]), name[i], units[i]);
    }
}

void DataCollector::addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                              const std::string& name,
                              const std::vector<std::string>& extension,
                              const std::vector<std::string>& units)
{
    std::vector<std::string> full_names;
    for (int i = 0; i < (int)extension.size(); ++i)
    {
        std::stringstream ss;
        ss << name << '_' << extension[i];
        full_names.push_back(ss.str());
    }
    addVector(data, full_names, units);
}

void DataCollector::addVector(const Eigen::Ref<const Eigen::VectorXd>& data,
                              const std::string& name,
                              const std::string& unit)
{
    std::vector<std::string> full_names;
    for (int i = 0; i < (int)data.size(); ++i)
    {
        std::stringstream ss;
        ss << name << '_' << i;
        full_names.push_back(ss.str());
    }
    addVector(data, full_names, std::vector<std::string>(data.size(), unit));
}

void DataCollector::addVector3d(const Eigen::Ref<const Eigen::Vector3d>& data,
                                const std::string& name,
                                const std::string& units)
{
    std::vector<std::string> varnames;
    varnames.push_back(name + "_x");
    varnames.push_back(name + "_y");
    varnames.push_back(name + "_z");
    std::vector<std::string> un;
    un.resize(3, units);
    addVector(data, varnames, un);
}

void DataCollector::addVector3d(const Eigen::Ref<const Eigen::Vector3d>& data,
                                const std::string& name,
                                const std::string& units,
                                const std::string& extension)
{
    std::vector<std::string> varnames;
    varnames.push_back(name + "_x" + extension);
    varnames.push_back(name + "_y" + extension);
    varnames.push_back(name + "_z" + extension);
    std::vector<std::string> un;
    un.resize(3, units);
    addVector(data, varnames, un);
}

void DataCollector::addQuaternion(const Eigen::Ref<const Eigen::Vector4d>& data,
                                  const std::string& name)
{
    std::vector<std::string> varnames;
    varnames.push_back(name + "_x");
    varnames.push_back(name + "_y");
    varnames.push_back(name + "_z");
    varnames.push_back(name + "_w");
    std::vector<std::string> un;
    un.resize(4, "-");
    addVector(data, varnames, un);
}

void DataCollector::addQuaternion(const Eigen::Quaterniond& data,
                                  const std::string& name)
{
    addQuaternion(data.coeffs(), name);
}

void DataCollector::addVector6d(
    const Eigen::Ref<const Eigen::Matrix<double, 6, 1> >& data,
    const std::string& name,
    const std::string& units)
{
    std::vector<std::string> varnames;
    varnames.push_back(name + "_x");
    varnames.push_back(name + "_y");
    varnames.push_back(name + "_z");
    varnames.push_back(name + "_a");
    varnames.push_back(name + "_b");
    varnames.push_back(name + "_g");
    std::vector<std::string> un;
    un.resize(6, units);
    addVector(data, varnames, un);
}

void DataCollector::addMatrix(const Eigen::Ref<const Eigen::MatrixXd>& data,
                              const std::string& name,
                              const std::string& unit)
{
    std::vector<std::string> full_names;
    for (int i = 0; i < (int)data.cols(); ++i)
    {
        for (int j = 0; j < (int)data.rows(); ++j)
        {
            std::stringstream ss;
            ss << name << '_' << j + 1 << '_' << i + 1;
            addVariable((double*)&(data.data()[i * (int)data.rows() + j]),
                        ss.str(),
                        unit);
        }
    }
}

}  // end namespace io_tools
}  // end namespace robot_estimation
