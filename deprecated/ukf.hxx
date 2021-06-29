/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implements the UKF templated class.
 */

#pragma once

#include "mim_estimation/standard_filters/ukf.hpp"

namespace mim_estimation
{
namespace standard_filters
{
template <class S, class M>
UKF<S, M>::UKF(bool numerical_jac, bool is_discrete, bool update_on, double dt)
{
    numerical_jac_ = numerical_jac;
    is_discrete_ = is_discrete;
    update_on_ = update_on;
    dt_ = dt;

    state_pre_.state_cov.resize(N, N);
    state_post_.state_cov.resize(N, N);
    meas_pred_.meas_cov.resize(K, K);
    meas_actual_.meas_cov.resize(K, K);
    proc_jac_.resize(N, N);
    proc_jac_disc_.resize(N, N);
    noise_jac_.resize(N, P);
    proc_noise_disc_.resize(N, N);
    proc_noise_.resize(P, P);
    meas_jac_.resize(K, N);
    meas_noise_disc_.resize(K, K);
    meas_noise_.resize(K, K);
    state_weights_.resize(L, 1);
    cov_weights_.resize(L, 1);
    spoints_mat_.resize(N + P, 2 * (N + P) + 1);
    cross_cov_.resize(N, K);
    gain_mat_.resize(N, K);

    state_pre_.state_cov.setZero();
    state_post_.state_cov.setZero();
    meas_pred_.meas_cov.setZero();
    meas_actual_.meas_cov.setZero();
    proc_jac_.setZero();
    proc_jac_disc_.setZero();
    noise_jac_.setZero();
    proc_noise_disc_.setZero();
    proc_noise_.setZero();
    meas_jac_.setZero();
    meas_noise_disc_.setZero();
    meas_noise_.setZero();
    state_weights_.setZero();
    cov_weights_.setZero();
    spoints_mat_.setZero();
    cross_cov_.setZero();
    gain_mat_.setZero();

    alpha_ = 1e-3;
    beta_ = 2.0;
    kappa_ = 0.0;
    lambda_ = alpha_ * alpha_ * (N + kappa_) - N;
    state_weights_(0) = (lambda_ / (N + lambda_));
    cov_weights_(0) = (lambda_ / (N + lambda_)) + (1 - alpha_ * alpha_ + beta_);
    for (int i = 1; i < L; ++i)
    {
        state_weights_(i) = (1.0 / (2.0 * (N + lambda_)));
        cov_weights_(i) = state_weights_(i);
    }
}

template <class S, class M>
void UKF<S, M>::update(void)
{
    updateControls();
    form_process_noise();
    form_measurement_noise();
    form_actual_measurement();

    // discretize:
    if (!is_discrete_)
    {
        form_process_jacobian();
        form_noise_jacobian();
        proc_noise_disc_ = dt_ * proc_jac_ * noise_jac_ * proc_noise_ *
                           noise_jac_.transpose() * proc_jac_.transpose();
        meas_noise_disc_ = meas_noise_ / dt_;
    }

    // predict:

    // form sigma pts:
    ldlt.compute((N + lambda_) * state_post_.state_cov);
    Eigen::Matrix<double, N, 1> state_vec = state_post_.getState();
    spoints_mat_ = state_vec.replicate(1, L);
    spoints_mat_.block<N, N>(0, 1) += ldlt.matrixL();
    spoints_mat_.block<N, N>(0, N + 1) -= ldlt.matrixL();

    spoints_proc_ = Eigen::MatrixXd::Zero(N, L);
    S tmp_state;
    for (int i = 0; i < L; ++i)
    {
        tmp_state = spoints_mat_.col(i);
        spoints_proc_.col(i) = process_model(tmp_state);
    }
    state_pre_ = spoints_proc_ * state_weights_;
    state_vec = state_pre_.getState();
    state_pre_.state_cov =
        (spoints_proc_ - state_vec.replicate(1, L)) *
            cov_weights_.asDiagonal() *
            (spoints_proc_ - state_vec.replicate(1, L)).transpose() +
        proc_noise_disc_;

    // update:
    if (update_on_)
    {
        spoints_meas_ = Eigen::MatrixXd::Zero(K, L);
        for (int i = 0; i < L; ++i)
        {
            tmp_state = spoints_proc_.col(i);
            spoints_meas_.col(i) = measurement_model(tmp_state);
        }
        meas_pred_ = spoints_meas_ * state_weights_;
        Eigen::Matrix<double, K, 1> meas_vec = meas_pred_.get_measurement();
        meas_pred_.meas_cov =
            (spoints_meas_ - meas_vec.replicate(1, L)) *
                cov_weights_.asDiagonal() *
                (spoints_meas_ - meas_vec.replicate(1, L)).transpose() +
            meas_noise_disc_;
        cross_cov_ = (spoints_proc_ - state_vec.replicate(1, L)) *
                     cov_weights_.asDiagonal() *
                     (spoints_meas_ - meas_vec.replicate(1, L)).transpose();

        ldlt.compute(meas_pred_.meas_cov);
        gain_mat_ = (ldlt.solve(cross_cov_.transpose())).transpose();

        state_post_ =
            state_pre_.getState() + gain_mat_ * (meas_actual_ - meas_pred_);

        state_post_.state_cov =
            state_pre_.state_cov -
            gain_mat_ * meas_pred_.meas_cov * gain_mat_.transpose();
    }
    else
    {
        state_post_ = state_pre_;
        state_post_.state_cov = state_pre_.state_cov;
    }
}

template <class S, class M>
void UKF<S, M>::print_debug(void)
{
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

    std::cout << "Sigma Points:" << std::endl;
    std::cout << spoints_mat_ << std::endl << std::endl;

    std::cout << "Sigma Points Proc:" << std::endl;
    std::cout << spoints_proc_ << std::endl << std::endl;

    std::cout << "Sigma Points Meas:" << std::endl;
    std::cout << spoints_meas_ << std::endl << std::endl;

    std::cout << "State Weights:" << std::endl;
    std::cout << state_weights_ << std::endl << std::endl;

    std::cout << "Cov Weights:" << std::endl;
    std::cout << cov_weights_ << std::endl << std::endl;

    std::cout << "Cross Cov:" << std::endl;
    std::cout << cross_cov_ << std::endl << std::endl;

    std::cout << "Gain Matrix:" << std::endl;
    std::cout << gain_mat_ << std::endl << std::endl;

    std::cout << "Meas (Actual):" << std::endl;
    std::cout << meas_actual_.get_measurement() << std::endl << std::endl;

    std::cout << "Meas (Pred):" << std::endl;
    std::cout << meas_pred_.get_measurement() << std::endl << std::endl;

    std::cout << "Delx:" << std::endl;
    std::cout << gain_mat_ * (meas_actual_ - meas_pred_) << std::endl
              << std::endl;
}

}  // namespace standard_filters
}  // namespace mim_estimation
