/**
 * @file
 * @license BSD 3-clause
 * @copyright Copyright (c) 2020, New York University and Max Planck
 * Gesellschaft
 *
 * @brief Implementation of the Extended Kalman Filter (EKF).
 */

#pragma once

#include <iostream>
#include "Eigen/Eigen"

namespace mim_estimation
{
namespace standard_filters
{
/**
 *  \brief This class implements an extended Kalman Filter (EKF).
 *
 *  This class implements a templated Extended Kalman Filter that receives
 *  a State class "S" and a measurement class
 *  "M". Both templated class must have this following operator: -, +, =, <<.
 *  This operator must be eigen compatible.
 *  To achieve this in a compact way I strongly suggest inheriting directly from
 *  the class Eigen::Vector<N> (see Eigen documentation
 *  https://eigen.tuxfamily.org/dox/TopicCustomizing_InheritingMatrix.html).
 *
 *  The main sources of reading comes from
 *    - [1] "S. Thrun, W. Burgard, and D. Fox, Probabilistic Robotics, 2000"
 * (Chapter 3)
 *    - [2] "http://wolfweb.unr.edu/~fadali/EE782/NumEvalQphi.pdf"
 *    - [3] "https://en.wikipedia.org/wiki/Extended_Kalman_filter"
 *
 *  The notation used in this documentation is the one from
 *  "http://wolfweb.unr.edu/~fadali/EE782/NumEvalQphi.pdf".
 *
 *  \f{align}
 *    x_t = g(u_t, x_{t-1}) + \epsilon_t \\
 *    z_t = h(x_t) + \delta_t
 *  \f}
 *  With \f$ x_t \f$ the system state, \f$ u_t \f$ the system control,
 *  \f$ g \f$ the process evolution, \f$ \epsilon_t \approx N(0, R_t) \f$ the
 *  process Gaussian noise,
 *  \f$ h \f$ the measurement evolution and \f$ \delta_t \approx N(0, Q_t) \f$
 *  the measurement Gaussian noise.
 *  The first equation depicts the "process" in the code.
 *  The second equation represents the "measurement" in the code.
 *
 *  Using a taylor expansion of the previous equation we get:
 *  \f{align}
 *    x_t = g(u_t, \mu_{t-1}) + G_t \; * \; (x_{t-1} - \mu_{t-1}) + \epsilon_t
 * \\ z_t = h(\mu_t) + H_t \; * \; (x_{t} - \mu_{t}) + \delta_t \f} With \f$
 * \mu_{t} \f$ the average of the estimated state of covariance \f$ \Sigma_t
 * \f$. The equation of the EKF routine are defined with the "update_filter"
 *  function.
 *
 *  The mapping between the variable names and this piece of documentation is
 *  done next to each variable.
 *
 */
template <class S, class M>
class EKF
{
public:
    /**
     * @brief EKF, constructor of the class
     * @param numerical_jac: do we compute the Jacobians analytically or not?
     * @param is_discrete: is the process discrete?
     * @param dt: the sampling period of the EKF.
     */
    EKF(bool numerical_jac,
        bool is_discrete,
        double dt,
        unsigned lin_order = 2);

    /** @brief Default destructor. */
    ~EKF()
    {
    }

    static const int N = S::state_dim; /*!< State dimension */
    static const int P = S::noise_dim; /*!< Noise dimension */
    static const int K = M::meas_dim;  /*!< Measurement dimension */

    /**
     * @brief updateFilter: compute a pass of the extended Kalman filter
     * @param update_on: is the measurement new?
     *
     * From here on we only have access to \f$ \mu_{t-1} \f$ and
     * \f$ \Sigma_{t-1} \f$ the previous estimated state along time.
     * First of all we need to check if the system is a discrete one
     * (see constructor). If the system is continuous we need to build a
     * discrete approximation of it. In order to do so we use the Van Loan's
     * method (see [2] slide 15).
     * - 1/ Compute the dynamic matrix of the continuous process \f$ M \f$
     * \f{align}
     *  M = \left[ \left.
     *  \frac{-G_t}{0}
     *  \right|
     *  \frac{E_t R_t E_t^T}{G_t^T}
     *  \right]
     * \f}
     * With \f$ E_t \f$ being the jacobian of the noise. In [2], \f$ E_t \f$ is
     * also considered as the control jacobian in a way.
     * - 2/ Obtain the matrix M exponential
     * \f{align}
     *  e^{M \Delta t} =
     *  \left[ \left.
     *  \frac{ e^{-G_t \Delta t} }{0}
     *  \right|
     *  \frac{ e^{-G_t \Delta t} F_{t, disc} }{ e^{G_t^T \Delta t} }
     *  \right]
     * \f}
     * - 3/ Transpose the lower right corner to compute the discretized process
     *      noise covariance.
     * \f{align}
     *  R_{t, disc} = e^{G_t \Delta t} = (e^{G_t^T \Delta t})^T
     * \f}
     * - 4/ Compute the discretized process jacobian as follow:
     * \f{align}
     *  F_{t, disc} = R_{t, disc} \;\;\; e^{M \Delta t}[\text{upper right
     * corner}] \f} In the following we will discard the \f$ {disc} \f$
     * subscript in the expression of the process Jacobian and noise covariance
     * for sake of clarity. We assume that the system is either discrete or
     * already discretized.
     *
     * When the system has been discretized (or is discrete) we compute the EKF
     * with the following procedure.
     * First we need to predict what is supposed to be
     * the current estimated state \f$ \tilde{\mu}_{t} \f$ and its Covariance
     * \f$ \tilde{\Sigma}_{t} \f$ according to the model and the current
     * control: \f{align}
     *    \tilde{\mu}_{t} &= g(u_t, \mu_{t-1}) \\
     *    \tilde{\Sigma}_{t} &= G_t \Sigma_{t-1} G_t^T + R_t
     * \f}
     * The second step consist in introducing the actual measurement when
     * one occur.
     * \f{align}
     *    K_{t} &= \tilde{\Sigma}_{t} H_t^T (H_t \tilde{\Sigma}_t H_t^T +
     * Q_t)^{-1} \\
     *    \mu_{t} &= \tilde{\mu}_{t} + K_t (z_t - h(\tilde{\mu}_t)) \\
     *    \Sigma_{t} &= (I - K_t H_t) \tilde{\Sigma}_{t}
     * \f}
     */
    void updateFilter(bool update_on);

    /**
     * @brief getFilterState
     * @return the current state estimate \f$ \mu_{t} \f$
     */
    virtual Eigen::Matrix<double, N, 1> getFilterState(void) = 0;

    /**
     * @brief printDebug, print all the protected method of the class
     * for debugging purposes
     */
    void printDebug(void);

protected:
    S state_pre_;  /*!< Predicted state (\f$ \tilde{\mu} \f$, \f$ \tilde{\Sigma}
                      \f$)*/
    S state_post_; /*!< State of the filter. One variable is used for the
                        previous (\f$ \mu_{t-1} \f$, \f$ \Sigma_{t-1} \f$) and
                        current state (\f$ \mu_{t} \f$, \f$ \Sigma_{t} \f$)*/
    M meas_actual_; /*!< Current measurement (\f$ z_{t} \f$) */
    M meas_pred_;   /*!< Predicted measurement (\f$ h(\tilde{\mu}_t)\f$ ) */

    /**
     * \brief processModel, a model of the continuous/discrete state dynamics
     * \f$ g(u_t, x_{t-1}) \f$ \param s: \f$ [x_{t-1}^T \;\; u_t^T]^T \f$
     * \return the predicted state (Max: should not return anything and update
     * state_pre_?)
     */
    virtual Eigen::Matrix<double, N, 1> processModel(S &s) = 0;

    /**
     * @brief formProcessJacobian, compute the Jacobian of the process \f$ G_t
     * \f$
     */
    virtual void formProcessJacobian(void) = 0;

    /**
     * @brief formProcessNoise, compute the covariance of the process noise \f$
     * R_t \f$
     */
    virtual void formProcessNoise(void) = 0;

    /**
     * @brief formNoiseJacobian, compute the Jacobian of the process noise \f$
     * E_t \f$
     */
    virtual void formNoiseJacobian(void) = 0;

    /**
     * @brief measModel, a model of the measurement dynamics \f$ h(x_t) \f$
     * @param s: a state \f$ x_t \f$
     * @return the predicted measurement (Max: should not return anything and
     * update meas_pred_?)
     */
    virtual Eigen::Matrix<double, K, 1> measModel(S &s) = 0;

    /**
     * @brief formMeasJacobian, compute the Jacobian of the measurement \f$ H_t
     * \f$
     */
    virtual void formMeasJacobian(void) = 0;

    /**
     * @brief formMeasJacobian, compute the measure covariance \f$ Q_t \f$
     */
    virtual void formMeasNoise(void) = 0;

    /**
     * @brief formMeasJacobian, compute the measurement from sensors \f$ z_t \f$
     */
    virtual void formActualMeas(void) = 0;

    /**
     * @brief computeJacobians, compute the finite differentiation
     * @param[in, out] G_t: the process jacobian to compute
     * @param[in, out] H_t: the measurement jacobian
     * @param[in, out] ds: the finite differentiation delta \f$ \frac{f(x+ds) -
     * f(x-ds)}{2ds} \f$
     */
    void computeJacobians(Eigen::MatrixXd &G_t,
                          Eigen::MatrixXd &H_t,
                          double ds);

    /**
     * @brief check_the_jacobians, this function computes the jacobian from the
     * discrete model and from the continuous jacobian discretized
     * and compare them
     * @param[in] evaluate: true, the function is computing the check/false
     * it does nothing
     * @return a boolean that says if everything is alright.
     */
    bool check_jacobians(bool evaluate);

    /*!< for Van Loan discretization, \f$ M \f$ */
    Eigen::MatrixXd VL_mat_;
    /*!< for Van Loan discretization, \f$ e^{M dt} \f$ */
    Eigen::MatrixXd exp_VL_mat_;

    /*!< discretized/discrete process Jacobian, \f$ G_{t,disc}/G_t \f$ */
    Eigen::MatrixXd proc_jac_disc_;
    /*!< user input process Jacobian, \f$ G_t \f$ */
    Eigen::MatrixXd proc_jac_;
    /*!< Jacobian of the process noise, \f$ E_t \f$ */
    Eigen::MatrixXd noise_jac_;
    /*!< Jacobian of the measurement, \f$ H_t \f$ */
    Eigen::MatrixXd meas_jac_;
    /*!< user input covariance of the process noise \f$ R_t \f$ */
    Eigen::MatrixXd proc_noise_;
    /*!< discretized/discrete covariance of the process noise \f$ R_{t,disc}/R_t
     * \f$ */
    Eigen::MatrixXd proc_noise_disc_;
    /*!< user input covariance of the measure noise \f$ Q_t \f$ */
    Eigen::MatrixXd meas_noise_;
    /*!< covariance of the process noise \f$ Q_t \f$ assuming a discrete
     * measurement */
    Eigen::MatrixXd meas_noise_disc_;
    /*!< Kalman filter gain matrix \f$ K_t \f$ */
    Eigen::MatrixXd gain_mat_;

    /*!< LDLT decomposition to invert some matrix */
    Eigen::LDLT<Eigen::MatrixXd> ldlt;

    /*!< Do we compute the Jacobian numerically or analytically? */
    bool numerical_jac_;
    /*!< Is the process discrete? */
    bool is_discrete_;
    /*!< sampling period */
    double dt_;
    /*!< order of the process dynamic linearization */
    unsigned lin_order_;
};

}  // namespace standard_filters
}  // namespace mim_estimation

#include "mim_estimation/standard_filters/ekf.hxx"
