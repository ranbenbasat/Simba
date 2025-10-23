#ifndef SIMBA_H
#define SIMBA_H

#include "defs.h"

/*
* Implementation of the Simba algorithm from the �Better than Optimal: Improving Adaptive Stochastic Quantization Using Shared Randomness� paper (ACM SIGMETRICS 2026)
*/
class Simba
{
    const double* X; // sorted input entries

    vector<double> pts_cumsum; // cumulative sum of input entries
    double sos; // sum of squares in X
    int d; // number of input entries
    void preprocess(); // preprocess cumsum and sos
    double MSE(int changed_h, int changed_q); // compute MSE given changed Q[h][q]. Compute from scratch if both are -1
    void calcThresholds(int changed_h, int changed_q); // compute thresholds given changed Q[h][q]. Compute from scratch if both are -1
    void calcBetas(); // compute betas from scratch
    void updateBetas(int h, int q, double oldval, double newval); // update betas given changed Q[h][q], old threshold value oldval, new threshold value newval
    double calc_MSEi(int i) const; // compute MSEi
    double lastMSE;
    int ell;
    int s;
    double minX, maxX;
    vector<vector<double>>Q;
    double* cur_cost;
    vector<int> new_threshold_indices;
    double* threshold_values;
    double* MSEi; // MSE contribution of each interval
    double* betai; // beta of each interval
    int iters;
    int bin_iters;
    double bin_iters_increase_threshold; // threshold to increase bin iters
    double stopping_threshold;           // termination threshold for Simba. If the current iteration's MSE improvement is repeatedly less than stopping_threshold*initialMSE, stop.
    bool debug;
    vector<vector<double>> _calcQuantizationValues(double* initial_levels = NULL, string log_cost_fn = "");
    double min_required_improvement;
public:
    vector<vector<double>> calcQuantizationValues(double* X, size_t d, int s, int ell, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, bool debug, double* initial_levels = NULL, string log_cost_fn = "");
};

#endif //SIMBA_H
