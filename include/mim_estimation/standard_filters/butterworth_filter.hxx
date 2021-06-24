/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements the ButterworthFilter temaplated class.
 */

#pragma once

#include <cmath>
#include <complex>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mim_estimation/standard_filters/butterworth_filter.hpp"

namespace mim_estimation
{
namespace standard_filters
{
template <typename _Scalar, int _Rows, int _Order>
ButterworthFilter<Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
                  _Order>::ButterworthFilter()
    : initialized_filter_(false)
{
}

template <typename _Scalar, int _Rows, int _Order>
ButterworthFilter<Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
                  _Order>::~ButterworthFilter()
{
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::initialize(const ButterworthFilter::VecToFilter& current,
                        double cutoff)
{
    if (_Order >= 1 && (cutoff <= 1. && cutoff >= 0.))
    {
        cutoff_ = cutoff;
        zeros_coeffs_vec_.setZero();
        poles_coeffs_vec_.setZero();
        measurements_mat_.setZero();
        states_mat_.setZero();

        initialized_filter_ = ButterworthFilter<
            Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
            _Order>::compute_filter_coefficients();
    }
    else
    {
        throw std::runtime_error("bad initialization of butterworth filter.");
    }

    if (initialized_filter_)
    {
        for (int i = 0; i <= _Order; i++)
        {
            measurements_mat_.col(i) = current;
            states_mat_.col(i) = current;
        }
    }
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::getData(ButterworthFilter::VecToFilter& current_estimate)
{
    current_estimate = states_mat_.col(0);
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::getData(Eigen::Ref<ButterworthFilter::VecToFilter>
                         current_estimate)
{
    current_estimate = states_mat_.col(0);
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::update(const ButterworthFilter::VecToFilter& new_measurement)
{
    if (initialized_filter_)
    {
        measurements_mat_.col(0) = new_measurement;
        states_mat_.col(0) = measurements_mat_ * zeros_coeffs_vec_ -
                             states_mat_.block(0, 1, _Rows, _Order) *
                                 poles_coeffs_vec_.block(1, 0, _Order, 1);
        for (int i = _Order; i > 0; i--)
        {
            measurements_mat_.col(i) = measurements_mat_.col(i - 1);
            states_mat_.col(i) = states_mat_.col(i - 1);
        }
    }
}

template <typename _Scalar, int _Rows, int _Order>
bool ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::compute_filter_coefficients()
{
    //! prewarp band edges to s plane with normalized sample frequency (T=2Hz)
    double T = 2;
    double W = 2 / T * tan(M_PI * cutoff_ / T);

    //! generate continuous time poles, zeros and gain for butterworth filter
    std::vector<std::complex<double> > Cpoles, Czeros;
    double C = 1, Cgain = pow(C, _Order);
    std::complex<double> imag_num(0., 1.);

    for (int j = 1; j <= _Order; j++)
    {
        std::complex<double> pole(M_PI * (2. * j + _Order - 1.) /
                                  (2. * _Order));
        Cpoles.push_back(std::exp(imag_num * pole));
    }
    if ((_Order % 2) == 1)
    {
        Cpoles.at((_Order - 1) / 2) = (std::complex<double>)(-1.);
    }

    //! continuous time frequency transform
    std::vector<std::complex<double> > CTpoles, CTzeros;
    double CTgain;
    ButterworthFilter<
        Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
        _Order>::CT_transform(Czeros,
                              Cpoles,
                              Cgain,
                              W,
                              true,
                              CTzeros,
                              CTpoles,
                              CTgain);

    //! discrete time frequency transform: bilinear transform to convert poles
    //! to the z plane
    std::vector<std::complex<double> > DTpoles, DTzeros;
    double DTgain;
    ButterworthFilter<
        Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
        _Order>::DT_transform(CTzeros,
                              CTpoles,
                              CTgain,
                              T,
                              DTzeros,
                              DTpoles,
                              DTgain);

    //! convert to the correct polynomial output form
    ButterworthFilter<
        Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
        _Order>::poly_coeffs(DTpoles, poles_coeffs_vec_);
    ButterworthFilter<
        Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
        _Order>::poly_coeffs(DTzeros, zeros_coeffs_vec_);
    zeros_coeffs_vec_.array() *= DTgain;

    return true;
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::CT_transform(const std::vector<std::complex<double> >& Czeros,
                          const std::vector<std::complex<double> >& Cpoles,
                          double Cgain,
                          double omega,
                          bool flag,
                          std::vector<std::complex<double> >& CTzeros,
                          std::vector<std::complex<double> >& CTpoles,
                          double& CTgain)
{
    int num_zeros = Czeros.size(), num_poles = Cpoles.size();
    CTzeros.resize(num_zeros);
    CTpoles.resize(num_poles);
    double C = 1.;

    CTgain = Cgain * pow(C / omega, (double)(num_zeros - num_poles));
    for (int i = 0; i < num_zeros; i++)
    {
        CTzeros.at(i) = Czeros.at(i) * omega / C;
    }
    for (int i = 0; i < num_poles; i++)
    {
        CTpoles.at(i) = Cpoles.at(i) * omega / C;
    }
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::DT_transform(const std::vector<std::complex<double> >& CTzeros,
                          const std::vector<std::complex<double> >& CTpoles,
                          double CTgain,
                          double T,
                          std::vector<std::complex<double> >& DTzeros,
                          std::vector<std::complex<double> >& DTpoles,
                          double& DTgain)
{
    int num_zeros = CTzeros.size(), num_poles = CTpoles.size();
    DTzeros.resize(num_zeros);
    DTpoles.resize(num_poles);

    std::complex<double> complex_gain = CTgain;
    for (int i = 0; i < num_zeros; i++)
    {
        complex_gain *= ((2. - CTzeros.at(i) * T) / T);
    }
    for (int i = 0; i < num_poles; i++)
    {
        complex_gain /= ((2. - CTpoles.at(i) * T) / T);
        DTpoles.at(i) = (2. + CTpoles.at(i) * T) / (2. - CTpoles.at(i) * T);
    }

    DTgain = complex_gain.real();
    if (num_zeros == 0)
    {
        for (int i = 0; i < num_poles; i++)
        {
            DTzeros.push_back(-1.);
        }
    }
    else
    {
    }
}

template <typename _Scalar, int _Rows, int _Order>
void ButterworthFilter<
    Eigen::Matrix<_Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1>,
    _Order>::poly_coeffs(const std::vector<std::complex<double> >& roots,
                         Eigen::Matrix<double, _Order + 1, 1>& coeffs)
{
    std::vector<std::complex<double> > coefficients, delta_coefficients;
    coefficients.resize(_Order + 1);
    delta_coefficients.resize(_Order);

    coefficients.at(0) = (std::complex<double>)(1.);
    for (int i = 1; i <= _Order; i++)
    {
        coefficients.at(i) = (std::complex<double>)(0.);
    }
    for (int i = 0; i < _Order; i++)
    {
        for (int j = 1; j <= i + 1; j++)
        {
            delta_coefficients.at(j - 1) = roots.at(i) * coefficients.at(j - 1);
        }
        for (int j = 1; j <= i + 1; j++)
        {
            coefficients.at(j) -= delta_coefficients.at(j - 1);
        }
    }
    for (int i = 0; i < coefficients.size(); i++)
    {
        coeffs(i) = coefficients.at(i).real();
    }
}

}  // namespace standard_filters
}  // namespace mim_estimation
