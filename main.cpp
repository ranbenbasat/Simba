#include <stdio.h>              
#include <stdlib.h>             
#include <fstream>
#include "defs.h"
#include "ApproxQUIVER.h"
#include "ExactQUIVER.h"
#include <random>
#include "Simba.h"



int main()
{
    {
        default_random_engine generator;
        generator.seed(42);


        int n = 1 << 21;

        int s = 64;
        int M = -1;
        int M_Simba_for_QUIVER_Initialization = 1000;
        int M_Simba = -1;
        int ell = 32;
        int iters = 10000000;


        bool run_ExactQUIVER = false;
        bool run_WeightedExactQUIVER = false;
        bool run_AccelQUIVER = false;

        bool run_ApproxQUIVER = false;
        bool run_WeightedApproxQUIVER = false;


        bool run_KoveSimba = true;

        string log_cost_fn = "";

        auto start = chrono::high_resolution_clock::now();
        auto stop = chrono::high_resolution_clock::now();

        bool debug = false;
        bool truncate = false;
        double T_Truncate = 3.097;
        int bin_iters = 2;
        double bin_iters_increase_threshold = .99;
        double stopping_threshold = .999;

        if (truncate) {
            bin_iters = 4;
            bin_iters_increase_threshold = .999;
            stopping_threshold = .99999;
        }
        string quantype = "Histogram";

        normal_distribution<double> distribution(0.0, 1);
        vector<double> vec(n);
        vector<double> svec;
        int idx = -1;
        double norm = 0;

        for (int i = 0; i < n; ++i) {
            double number = distribution(generator);
            norm += number * number;
            if (!truncate || (number < T_Truncate && number > -T_Truncate)) {
                vec[++idx] = number;
            }
        }
        if (truncate) {
            n = idx + 1;
        }
        vec.resize(idx + 1);
        svec.resize(idx + 1);
        cout << "norm = " << norm << endl;


        start = chrono::high_resolution_clock::now();
        partial_sort_copy(vec.begin(), vec.end(), svec.begin(), svec.end());
        stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "partial_sort_copy time: " << duration.count() / 1000 << " ms" << endl;


        double eps = 1;

        double vnmse;

        vector<double> W(n, 1);
        for (int i = 0; i < n; ++i) {
            W[i] = 1; // to compare with unweighted version
        }
        if (run_ExactQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<false, false> eq(svec.data(), (uint32_t)n, nullptr);
            auto quant_values = eq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "ExactQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "ExactQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_WeightedExactQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<true, false> eq(svec.data(), (uint32_t)n, W.data());
            auto quant_values = eq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, &W);

            cout << "WeightedExactQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "WeightedExactQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_AccelQUIVER) {
            start = chrono::high_resolution_clock::now();
            ExactQUIVER<false, true> aeq(svec.data(), (uint32_t)n, nullptr);
            auto quant_values = aeq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "AccelQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "AccelQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_ApproxQUIVER) {
            start = chrono::high_resolution_clock::now();
            ApproxQUIVER<false> taq(svec.data(), (uint32_t)n, nullptr, M);
            auto quant_values = taq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, nullptr);

            cout << "ApproxQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "ApproxQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_WeightedApproxQUIVER) {
            start = chrono::high_resolution_clock::now();
            ApproxQUIVER<true> taq(svec.data(), (uint32_t)n, W.data(), M);
            auto quant_values = taq.calcQuantizationValues(s);
            stop = chrono::high_resolution_clock::now();

            vnmse = sq_vnmse(svec, quant_values, &W);

            cout << "WeightedApproxQUIVER: ";
            cout << "[";
            for (int i = 0; i < s; ++i) {
                cout << quant_values[i] << ", ";
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "WeightedApproxQUIVER time: " << duration.count() / 1000 << " ms" << endl;
        }
        if (run_KoveSimba) {
            start = chrono::high_resolution_clock::now();
            Simba As;
            vector<double> initial_levels;
            if (M_Simba_for_QUIVER_Initialization == -1) {
                ExactQUIVER<false, true> eq(svec.data(), (uint32_t)n, nullptr);
                initial_levels = eq.calcQuantizationValues(s);
            }
            else {
                ApproxQUIVER<false> taq(svec.data(), (uint32_t)n, nullptr, M_Simba_for_QUIVER_Initialization);
                initial_levels = taq.calcQuantizationValues(s);
                vnmse = sq_vnmse(svec, initial_levels, nullptr);
                //cout << "Initial levels vnmse = " << vnmse << endl;
            }
            auto quant_values_array = As.calcQuantizationValues(svec.data(), svec.size(), s, ell, iters, bin_iters, bin_iters_increase_threshold, stopping_threshold, M_Simba, debug, quantype, initial_levels.data(), log_cost_fn);
            stop = chrono::high_resolution_clock::now();

            std::vector<std::vector<double>> quant_values;
            for (int i = 0; i < ell; ++i) {
                std::vector<double> v;
                for (int j = 0; j < s; ++j) {
                    v.push_back(quant_values_array[i][j]);
                }
                quant_values.push_back(v);
            }

            if (truncate) {
                vnmse = calc_SR_vNMSE(svec, quant_values, norm);
            }
            else {
                vnmse = calc_SR_vNMSE(svec, quant_values);
            }

            cout << "No histogram QUIVER_initialized KoveSimba: ";
            cout << "[";
            for (auto v : quant_values) {
                cout << "[";
                for (auto u : v) {
                    cout << u << ", ";
                }
                cout << "]" << endl;
            }

            cout << "], vnmse = " << vnmse << endl;
            auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
            cout << "No histogram QUIVER_initialized KoveSimba time: " << duration.count() / 1000 << " ms" << endl;

            //exit(1);
        }   
    }

    // Check for memory leaks: send all reports to STDOUT
    // _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    // _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    // _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    // _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
    // _CrtDumpMemoryLeaks();

    return 0;
}


