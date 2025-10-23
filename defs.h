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

const double one_plus_eps = 1.0000000001;

double sq_vnmse(vector<double>& X, vector<double>& Q, vector<double>* W); // Used for ASQ methods (without shared randomness)

double calc_SR_vNMSE(const std::vector<double>& X, // Used for AUQ methods (with shared randomness)
    const std::vector<std::vector<double>>& Q,
    double snorm = -1.0);

#endif // !DEFS_H
