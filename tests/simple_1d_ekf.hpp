#include <Eigen/Core>
#include "mim_estimation/filtering_tools/ekf.hpp"
#include "mim_estimation/io_tools/non_rt_data_collector.hpp"

using namespace mim_estimation::standard_filters;

class SimpleEKFState : public Eigen::Matrix<double, 1, 1>
{
public:
    static const int state_dim = 1;
    static const int noise_dim = 1;

    SimpleEKFState(void) : Eigen::Matrix<double, 1, 1>(), pos_(this->head<1>())
    {
        state_cov.setZero();
    }

    // This constructor allows you to construct SimpleEKFState from Eigen
    // expressions
    template <typename OtherDerived>
    SimpleEKFState(const Eigen::MatrixBase<OtherDerived>& other)
        : Eigen::Matrix<double, 1, 1>(other), pos_(this->head<1>())
    {
        state_cov.setZero();
    }

    // This method allows you to assign Eigen expressions to SimpleEKFState
    template <typename OtherDerived>
    SimpleEKFState& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
        this->Eigen::Matrix<double, 1, 1>::operator=(other);
        return *this;
    }

    Eigen::Matrix<double, 1, 1> getState()
    {
        return *this;
    }

    Eigen::Ref<Eigen::Matrix<double, 1, 1> > pos_;
    Eigen::Matrix<double, 1, 1> state_cov;
};

class SimpleEKFMeasure : public Eigen::Matrix<double, 1, 1>
{
public:
    static const int meas_dim = 1;

    SimpleEKFMeasure(void)
        : Eigen::Matrix<double, 1, 1>(), meas_pos_(this->head<1>())
    {
    }

    // This constructor allows you to construct SimpleEKFMeasure from Eigen
    // expressions
    template <typename OtherDerived>
    SimpleEKFMeasure(const Eigen::MatrixBase<OtherDerived>& other)
        : Eigen::Matrix<double, 1, 1>(other), meas_pos_(this->head<1>())
    {
    }

    // This method allows you to assign Eigen expressions to SimpleEKFMeasure
    template <typename OtherDerived>
    SimpleEKFMeasure& operator=(const Eigen::MatrixBase<OtherDerived>& other)
    {
        this->Eigen::Matrix<double, 1, 1>::operator=(other);
        return *this;
    }

    Eigen::Ref<Eigen::Matrix<double, 1, 1> > meas_pos_;
    Eigen::Matrix<double, 1, 1> meas_cov;
};

class SimpleEKF : public EKF<SimpleEKFState, SimpleEKFMeasure>
{
public:
    SimpleEKF(double init_pos, double dt, bool is_discrete)
        : EKF(false, is_discrete, dt)
    {
        init_var_ = 0.1;
        // initialize the covariances
        Eigen::Matrix<double, N, N> init_cov =
            init_var_ * Eigen::Matrix<double, N, N>::Identity();
        state_pre_.state_cov = init_cov;
        state_post_.state_cov = init_cov;
        meas_pred_.meas_cov = init_cov;
        meas_actual_.meas_cov = init_cov;
        // initialize the state
        state_post_.pos_(0, 0) = init_pos;
        state_pre_ = state_post_;
        // set the measurement
        meas_vel_(0, 0) = 0.0;
        // covariances gains
        q_proc_noise_ = 0.01;
        q_meas_noise_ = 0.01;
    }

    ~SimpleEKF()
    {
    }

    void update(Eigen::Matrix<double, 1, 1> pos, bool new_measured_value)
    {
        // store the measurements
        meas_pos_ = pos;
        // update the filter state
        updateFilter(new_measured_value);
    }

    Eigen::Matrix<double, N, 1> getFilterState(void)
    {
        return state_post_;
    }

    void subscribe_to_data_collector(
        mim_estimation::io_tools::DataCollector& data_collector)
    {
        data_collector.addVariable(state_post_.pos_.data(), "state_pos", "m");
        data_collector.addVariable(
            state_pre_.pos_.data(), "state_pred_pos", "m");
        data_collector.addVariable(gain_mat_.data(), "gain", "-");
        data_collector.addMatrix(state_post_.state_cov, "state_cov", "m");
        data_collector.addVariable(meas_actual_.data(), "meas_actual", "-");
        data_collector.addVariable(meas_pred_.data(), "meas_pred", "-");
    }

private:
    /**
     * @brief processModel
     * * \f{align}
     *  \dot{pos} = (meas_vel - meas_vel_bias)
     *  \dot{bias} = 0.0
     * \f}
     *
     * @param s: current state
     * @return out: the predicted state
     */
    Eigen::Matrix<double, N, 1> processModel(SimpleEKFState& s)
    {
        Eigen::Matrix<double, N, 1> out;
        out = s.pos_ + dt_ * meas_vel_;
        return out;
    }

    /**
     * @brief formProcessJacobian
     */
    void formProcessJacobian(void)
    {
        proc_jac_ << 0;
    }

    void formProcessNoise(void)
    {
        proc_noise_ << q_proc_noise_ * q_proc_noise_;
    }

    void formNoiseJacobian(void)
    {
        noise_jac_ << 1;
    }

    /**
     * @brief measModel
     * @param s: current state
     * @return out: the predicted measurement
     */
    Eigen::Matrix<double, K, 1> measModel(SimpleEKFState& s)
    {
        Eigen::Matrix<double, K, 1> out;
        out = s.pos_;
        return out;
    }

    void formMeasJacobian(void)
    {
        meas_jac_ << 1;
    }

    void formMeasNoise(void)
    {
        meas_noise_ << q_meas_noise_ * q_meas_noise_;
    }

    void formActualMeas(void)
    {
        meas_actual_ = meas_pos_;
    }

    Eigen::Matrix<double, 1, 1> vel_;
    Eigen::Matrix<double, 1, 1> meas_pos_;
    Eigen::Matrix<double, 1, 1> meas_vel_;
    double q_proc_noise_;
    double q_meas_noise_;
    double init_var_;
};
