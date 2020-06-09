/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck Gesellschaft
 * 
 * @brief Implements the Unscented Kalman Filter (UKF).
 * 
 * The theory is described here.
 * https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
 * 
 * @todo Explain here the math a little at least or add ref.
 */

#include "Eigen/Eigen"
#include <iostream>

template <class S, class M> 
class UKF {

public:

  UKF(bool numerical_jac, bool is_discrete, bool update_on, double dt);
  ~UKF(){}

  static const int N = S::state_dim;
  static const int P = S::noise_dim;
  static const int K = M::meas_dim;
  static const int L = 2*N+1; // number of Sigma pts

  virtual void initialize(void) = 0;
  virtual void updateControls(void){}
  void update(void);
  virtual Eigen::Matrix<double,N,1> getFilterState(void) = 0;
  void printDebug(void);

 protected:

  S state_pre_, state_post_;
  M meas_actual_, meas_pred_;

  virtual Eigen::Matrix<double,N,1> processModel(S &s) = 0;
  virtual Eigen::Matrix<double,K,1> measModel(S &s) = 0;
  virtual void formProcessJacobian(void) = 0;
  virtual void formProcessNoise(void) = 0;
  virtual void formNoiseJacobian(void) = 0;
  virtual void formMeasNoise(void) = 0;
  virtual void formActualMeas(void) = 0;

  Eigen::MatrixXd proc_jac_;
  Eigen::MatrixXd proc_jac_disc_;

  Eigen::MatrixXd noise_jac_;

  Eigen::MatrixXd proc_noise_;
  Eigen::MatrixXd proc_noise_disc_;

  Eigen::MatrixXd meas_jac_;

  Eigen::MatrixXd meas_noise_;
  Eigen::MatrixXd meas_noise_disc_;

  Eigen::MatrixXd state_weights_;
  Eigen::MatrixXd cov_weights_;

  Eigen::MatrixXd spoints_mat_;
  Eigen::MatrixXd spoints_proc_;
  Eigen::MatrixXd spoints_meas_;
  Eigen::MatrixXd cross_cov_;
  Eigen::MatrixXd gain_mat_;

  Eigen::LDLT<Eigen::MatrixXd> ldlt;

  double alpha_, beta_, kappa_, lambda_;

  bool numerical_jac_, is_discrete_, update_on_;
  double dt_;
};

#include "robot_estimation/filtering_tools/ukf.hxx"
