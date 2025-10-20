#ifndef DEFS_H
#define DEFS_H

#define CEIL(x) ((int) (x + 0.9999999999999));

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <limits>
#include <fstream>
#include <list>
#include <chrono>
#include <functional>
#include <numeric>
#include <cmath>

using namespace std;

double sq_vnmse(vector<double>& svec, vector<double>& sqv, vector<double> *W);




void newCalcGrid(double* svec, int d, int M, string type, double* grid);
double calc_SR_vNMSE(const std::vector<double>& vec,
	const std::vector<std::vector<double>>& sqv,
	double snorm = -1.0);

std::vector<double> calcThresholds(const std::vector<std::vector<double>>& sqv);
double calcNormSquared(const std::vector<double>& vec);


#endif // !DEFS_H#pragma once
