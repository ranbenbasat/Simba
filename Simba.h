#ifndef SIMBA_H
#define SIMBA_H

#include "defs.h"

class Simba
{
	const double* svec;

	vector<double> pts_cumsum;
	double sos;
	int d;

	void preprocess();

	double cost(int changed_H, int changed_X);

	void calcThresholds(int changed_H, int changed_X);

	void calcBetas();

	void updateBetas(int j, int q, double oldval, double newval);

	double calc_MSEi(int i) const;
	double lastMSE;

	int ell;
	int s;
	double minD, maxD;

	vector<vector<double>>levels;

	double* cur_cost;

	vector<int> new_threshold_indices;
	double* threshold_values;
	double* MSEi;
	double* betai;

	int iters;
	int bin_iters;
	double bin_iters_increase_threshold;
	double stopping_threshold;
	bool debug;

	vector<vector<double>> calcQuantizationValuesSR(double* initial_levels = NULL, string log_cost_fn = "");

	double min_required_improvement;

public:

	vector<vector<double>> calcQuantizationValues(double* svec_p, size_t d, int s, int ell, int iters, int bin_iters, double bin_iters_increase_threshold, double stopping_threshold, int m, bool debug, string quantype, double* initial_levels = NULL, string log_cost_fn = "");

};

#endif //SIMBA_H
