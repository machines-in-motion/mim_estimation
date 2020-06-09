/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft
 * 
 * @brief Implements the Butterworth filters. An Introduction to the math can be
 * found here: https://en.wikipedia.org/wiki/Butterworth_filter
 */

#pragma once

#include <vector>
#include <iostream>
#include <Eigen/Dense>

namespace standard_filters{

  /*! 
   *  General filter template
   */
  template<class classType, int _Order>
  class ButterworthFilter
  {
  };

  /*! 
   *  Specialization to an Eigen::Vector. This Filter class implements a butterworth low pass
   *  filter able to filter a vector of measurements. Only Eigen::Vectors or Eigen::Matrices
   *  with column dimension 1 are valid.
   *  Usage example: 
   *                    typedef Filter<Eigen::Matrix<double,6,1> > FilterType;
   *   INSTANTIATE      FilterType myfilter;
   *   INITIALIZE       myfilter.initialize(initial_state, filter_order, cuttoff_frequency);
   *   UPDATE           myfilter.update(measurement);
   *   RETRIEVE EST     myfilter.getData(estimate);
   */
  template<typename _Scalar, int _Rows, int _Order>
  class ButterworthFilter<Eigen::Matrix< _Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1 >, _Order>
  {
    public:
      //! Input and Output type definition (VecToFilter)
      typedef Eigen::Matrix< _Scalar, _Rows, 1, Eigen::ColMajor, _Rows, 1 > VecToFilter;

    public:
      ButterworthFilter();
      ~ButterworthFilter();

      /*!
       *  Function to initialize several parameters of the filter
       *  @param[in] current_state	initial state of the system of type VecToFilter
       *  @param[in] filter_order	order of the butterworth filter (integer >= 1)
       *  @param[in] cutoff_frequency	cuttoff frequency for the filter as percentage of the nyquist frequency.
       *                                For example, if sampling frequency is 200Hz; then the Nyquist frequency
       *                                is half of the sampling frequency, namely 100Hz. cut_off frequency is a 
       *                                double element of [0-1], such that 0 corresponds to 0*Nyquist frequency
       *                                and 1 corresponds to a cutoff_frequency equal to the Nyquist frequency.
       */
      void initialize(const VecToFilter& current_state, double cutoff_frequency);

      /*!
       *  Function to update the filtered state based on the new raw measurement
       *  @param[in] new_measurement	new raw measurement of type VecToFilter
       */
      void update(const VecToFilter& new_measurement);

      /*!
       *  Function to ask for the filtered estimate
       *  @param[out] current_state	filtered estimate of state of type VecToFilter
       */
      void getData(VecToFilter& current_state);
      void getData(Eigen::Ref<ButterworthFilter::VecToFilter> current_state);

    private:

      /*!
       *  Function to compute the coefficients of a Butterworth filter without reading them from a table.
       *  It only needs to be done once at initialization.
       *  return                        true if successfull computing filter coefficients
       */
      bool compute_filter_coefficients();

      //! Auxiliar function to compute the coefficients of a Butterworth filter
      void CT_transform(const std::vector<std::complex<double> >& Czeros,
                        const std::vector<std::complex<double> >& Cpoles,
                        double Cgain, double omega, bool flag,
                        std::vector<std::complex<double> >& CTzeros,
                        std::vector<std::complex<double> >& CTpoles,
                        double& CTgain);

      //! Auxiliar function to compute the coefficients of a Butterworth filter
      void DT_transform(const std::vector<std::complex<double> >& CTzeros,
                        const std::vector<std::complex<double> >& CTpoles,
                        double CTgain, double T,
                        std::vector<std::complex<double> >& DTzeros,
                        std::vector<std::complex<double> >& DTpoles,
                        double& DTgain);

      //! Auxiliar function to compute the coefficients of a Butterworth filter
      void poly_coeffs(const std::vector<std::complex<double> >& roots,
                       Eigen::Matrix<double, _Order+1, 1>& coeffs);

      //! storage of butterworth filter coefficients
      Eigen::Matrix<double, _Order+1, 1> zeros_coeffs_vec_, poles_coeffs_vec_;

      //! logic variable
      bool initialized_filter_;

      //! order for the butterworth filter (Should be >=1)
      //int order_;

      //! cutoff frequency for the butterworth filter ([0-1] percentage of the nyquist frequency)
      double cutoff_;

      //! matrices to store past estimates and measurements for filtering
      Eigen::Matrix<double, _Rows, _Order+1> measurements_mat_, states_mat_;
  };
}

#include "robot_estimation/filtering_tools/butterworth_filter.hxx"
