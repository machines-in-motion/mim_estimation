/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft
 * 
 * @brief Implements the EKF templated class.
 */

#include <fstream>

// used here only to help the IDE (qtcreator) find the different variable
// this include does nothing
#include "robot_estimation/filtering_tools/ekf.hpp"

#ifndef EKF_HPP_
#define EKF_HPP_

template <class S, class M>
EKF<S, M>::EKF(bool numerical_jac, bool is_discrete,
               double dt, unsigned lin_order) :
  numerical_jac_(numerical_jac),
  is_discrete_(is_discrete),
  dt_(dt),
  lin_order_(lin_order)
{

  // Van loan discretization variables
  VL_mat_.resize(2*N,2*N);
  exp_VL_mat_.resize(2*N,2*N);
  // discrete process variable
  proc_jac_disc_.resize(N,N);
  proc_noise_disc_.resize(N,N);
  // continuous process variables
  proc_jac_.resize(N,N);
  noise_jac_.resize(N,N);
  proc_noise_.resize(N,N);

  // measurement variables
  meas_jac_.resize(K,N);
  meas_noise_disc_.resize(K,K);
  meas_noise_.resize(K,K);
  // kalman filter measurement gain
  gain_mat_.resize(N,K);

  VL_mat_.setZero();
  exp_VL_mat_.setZero();
  proc_jac_disc_.setZero();
  proc_jac_.setZero();
  noise_jac_.setZero();
  proc_noise_disc_.setZero();
  proc_noise_.setZero();
  meas_jac_.setZero();
  meas_noise_disc_.setZero();
  meas_noise_.setZero();
  gain_mat_.setZero();

}

template <class S, class M>
void EKF<S, M>::updateFilter(bool update_on) {

  // set to true if you want the check activated (!Non RT safe)
  // check_jacobians(false);

  // compute the jacobians
  if(numerical_jac_) {
    computeJacobians(proc_jac_, meas_jac_, 1e-12);
  } else {
    formProcessJacobian(); // proc_jac_
    formMeasJacobian(); // meas_jac_
  }

  // compute the covariances
  formProcessNoise(); // proc_noise_
  formMeasNoise(); // meas_noise_

  // get the actual measurement data
  formActualMeas(); // meas_actual_

  // discretize process jacobian and noise matrix using Van Loan's method
  // (http://wolfweb.unr.edu/~fadali/EE782/NumEvalQphi.pdf):
  if(is_discrete_) {
    proc_jac_disc_ = proc_jac_;
    proc_noise_disc_ = proc_noise_;
  } else {
    if(lin_order_ == 2){// order 2 linearization of the process dynamics
      // in the continuous time system case we compute noise_jac_
      formNoiseJacobian();// noise_jac_
      VL_mat_.block(0,0,N,N) = - proc_jac_;
      VL_mat_.block(0,N,N,N) = noise_jac_ * proc_noise_ * noise_jac_.transpose();
      VL_mat_.block(N,N,N,N) = proc_jac_.transpose();
      exp_VL_mat_ = Eigen::MatrixXd::Identity(2 * N, 2 * N) + dt_*VL_mat_ +
                    0.5 * dt_ * dt_ * VL_mat_ * VL_mat_;
      proc_jac_disc_ = exp_VL_mat_.block(N,N,N,N).transpose();
      proc_noise_disc_ = proc_jac_disc_ * exp_VL_mat_.block(0,N,N,N);

    }else{// order 1 linearization of the process  dynamic
      proc_jac_disc_ = Eigen::MatrixXd::Identity(N, N) + dt_ * proc_jac_;
      proc_noise_disc_ = proc_noise_ / dt_;
    }
  }

  // assume measurements are discrete, else need to divide by dt:
  // meas_noise_disc_ = meas_noise_/dt_;
  meas_noise_disc_ = meas_noise_;

  // predict:
  state_pre_ = processModel(state_post_);
  state_pre_.state_cov =
      proc_jac_disc_ * state_post_.state_cov * proc_jac_disc_.transpose() +
      proc_noise_disc_;

  // update:
  if(update_on) {

    meas_pred_ = measModel(state_pre_);
    meas_pred_.meas_cov =
        meas_jac_ * state_pre_.state_cov * meas_jac_.transpose() +
        meas_noise_disc_;

    ldlt.compute(meas_pred_.meas_cov);
    gain_mat_ = (ldlt.solve(meas_jac_)).transpose();
    gain_mat_ = state_pre_.state_cov * gain_mat_;

    state_post_ = state_pre_.getState() + gain_mat_ * (meas_actual_-meas_pred_);
    state_post_.state_cov = state_pre_.state_cov -
                            gain_mat_ * meas_jac_ * state_pre_.state_cov;

  } else {

    state_post_ = state_pre_;
    state_post_.state_cov = state_pre_.state_cov;

  }
}

template <class S, class M>
void EKF<S, M>::computeJacobians(Eigen::MatrixXd &G_t,
                                 Eigen::MatrixXd &H_t, double ds) {
  // reset the matrices
  G_t.setZero();
  H_t.setZero();
  // Compute G_t:
  Eigen::Matrix<double,N,1> dx, diff_x;
  S x_m, x_p, x_mdot, x_pdot;
  // Compute H_t:
  M s_m, s_p;
  Eigen::Matrix<double,K,1> diff_s;

  for(int i=0; i<N; ++i) {

    dx.setZero();

    // minus:
    dx(i) = -ds;

    x_m = state_post_ + dx;

    x_mdot = processModel(x_m);
    s_m = measModel(x_m);

    // plus:
    dx(i) = ds;

    x_p = state_post_ + dx;

    x_pdot = processModel(x_p);
    s_p = measModel(x_p);

    // central finite difference:
    diff_x = x_pdot - x_mdot;
    diff_s = s_p - s_m;

    G_t.block(0,i,N,1) = diff_x/(2*ds);
    H_t.block(0,i,K,1) = diff_s/(2*ds);

  }
}

template <class S, class M>
bool EKF<S, M>::check_jacobians(bool evaluate)
{
  bool ok = true;
  if (evaluate){
    computeJacobians(proc_jac_, meas_jac_, 1e-12);
    Eigen::MatrixXd proc_jac_fd = proc_jac_;
    Eigen::MatrixXd meas_jac_fd = meas_jac_;
    formProcessJacobian(); // proc_jac_
    formMeasJacobian(); // meas_jac_
    double error_proc_jac =
        (proc_jac_fd -
         Eigen::MatrixXd::Identity(N, N) - dt_ * proc_jac_).norm();
    double meas_proc_jac = (meas_jac_fd - meas_jac_).norm();
    assert(error_proc_jac < 1e-3 && "The process jacobian is wrong");
    assert(meas_proc_jac < 1e-3 && "The measure jacobian is wrong");
    ok &= error_proc_jac < 1e-3;
    ok &= meas_proc_jac < 1e-3;
  }
  return ok;
}

template <class S, class M>
void EKF<S, M>::printDebug(void) {

  std::cout << "State (pre):" << std::endl;
  std::cout << state_pre_.getState() << std::endl << std::endl;

  std::cout << "State (post):" << std::endl;
  std::cout << state_post_.getState() << std::endl << std::endl;

  std::cout << "State Cov (pre):" << std::endl;
  std::cout << state_pre_.state_cov << std::endl << std::endl;

  std::cout << "State Cov (post):" << std::endl;
  std::cout << state_post_.state_cov << std::endl << std::endl;

  std::cout << "Process Jacobian (continuous):" << std::endl;
  std::cout << proc_jac_ << std::endl << std::endl;

  std::cout << "Process Jacobian (discrete):" << std::endl;
  std::cout << proc_jac_disc_ << std::endl << std::endl;

  std::cout << "Process Noise (continuous):" << std::endl;
  std::cout << proc_noise_ << std::endl << std::endl;

  std::cout << "Process Noise (discrete):" << std::endl;
  std::cout << proc_noise_disc_ << std::endl << std::endl;

  std::cout << "Noise Jacobian (continuous):" << std::endl;
  std::cout << noise_jac_ << std::endl << std::endl;

  std::cout << "Measurement Jacobian (cont/disc):" << std::endl;
  std::cout << meas_jac_ << std::endl << std::endl;

  std::cout << "Measurement Noise (continuous):" << std::endl;
  std::cout << meas_noise_ << std::endl << std::endl;

  std::cout << "Measurement Noise (discrete):" << std::endl;
  std::cout << meas_noise_disc_ << std::endl << std::endl;

  std::cout << "Gain Matrix:" << std::endl;
  std::cout << gain_mat_ << std::endl << std::endl;

  std::cout << "Meas (Actual):" << std::endl;
  std::cout << meas_actual_.getMeas() << std::endl << std::endl;

  std::cout << "Meas (Pred):" << std::endl;
  std::cout << meas_pred_.getMeas() << std::endl << std::endl;

  std::cout << "Delx:" << std::endl;
  std::cout << gain_mat_*(meas_actual_-meas_pred_) << std::endl << std::endl;

}

#endif // EKF_HPP_
