#include <iostream>
#include <fstream>
#include <cstdlib>
#include <complex>
#include <time.h>
#include <cstring>
#include <CL/cl.hpp>

#include "traits.h"
#include "settings.h"
#include "operations.hpp"
#include "fdn_gpu_c.hpp"
#include "cl_info.hpp"

int main(int argc, char *argv[])
{
    // Переменные
    cl_float2 *signal = new cl_float2[POLAR*NP*CHP];
    cl_float2 *coef = new cl_float2[DNP*CHP];
    cl_float2 *dn_out = new cl_float2[POLAR*DNP*(NP + NF - 1)];
    cl_float2 *dn_out_k = new cl_float2[POLAR*DNP*(NP + NF - 1)];
    int *del_c = new int[DNP*CHP];
    float *del_f = new float[DNP*CHP];
    float *hfir = new float[NPOLINOM*NF];

#ifdef INFO
    OpenCL_Info();
#endif

#if MODEL_TEST == 1 && TIME_TEST == 0
#ifdef QT_DEBUG
#if RECTANGLE_TILE == 1 || RECTANGLE_TILE_WITH_REGISTER == 1
    const char nameSignRe[] = "./tb/rectangle/signal/signal_re.txt";
    const char nameSignIm[] = "./tb/rectangle/signal/signal_im.txt";
    const char nameCoefRe[] = "./tb/rectangle/coef/coef_re.txt";
    const char nameCoefIm[] = "./tb/rectangle/coef/coef_im.txt";
    const char nameDelayCeil[] = "./tb/rectangle/delay_ceil/delay_ceil.txt";
    const char nameDelayFrac[] = "./tb/rectangle/delay_frac/delay_frac.txt";
    const char nameDnRe[] = "./tb/rectangle/dn/dn_re.txt";
    const char nameDnIm[] = "./tb/rectangle/dn/dn_im.txt";
#elif QUADRATE_TILE == 1
    const char nameSignRe[] = "./tb/quadrat/signal/signal_re.txt";
    const char nameSignIm[] = "./tb/quadrat/signal/signal_im.txt";
    const char nameCoefRe[] = "./tb/quadrat/coef/coef_re.txt";
    const char nameCoefIm[] = "./tb/quadrat/coef/coef_im.txt";
    const char nameDelayCeil[] = "./tb/quadrat/delay_ceil/delay_ceil.txt";
    const char nameDelayFrac[] = "./tb/quadrat/delay_frac/delay_frac.txt";
    const char nameDnRe[] = "./tb/quadrat/dn/dn_re.txt";
    const char nameDnIm[] = "./tb/quadrat/dn/dn_im.txt";
#endif
    const char nameHfir[] = "./tb/h_fir/h_fir.txt";
#else
#if RECTANGLE_TILE == 1 || RECTANGLE_TILE_WITH_REGISTER == 1
    const char nameSignRe[] = "./../tb/rectangle/signal/signal_re.txt";
    const char nameSignIm[] = "./../tb/rectangle/signal/signal_im.txt";
    const char nameCoefRe[] = "./../tb/rectangle/coef/coef_re.txt";
    const char nameCoefIm[] = "./../tb/rectangle/coef/coef_im.txt";
    const char nameDelayCeil[] = "./../tb/rectangle/delay_ceil/delay_ceil.txt";
    const char nameDelayFrac[] = "./../tb/rectangle/delay_frac/delay_frac.txt";
    const char nameDnRe[] = "./../tb/rectangle/dn/dn_re.txt";
    const char nameDnIm[] = "./../tb/rectangle/dn/dn_im.txt";
#elif QUADRATE_TILE == 1
    const char nameSignRe[] = "./../tb/quadrat/signal/signal_re.txt";
    const char nameSignIm[] = "./../tb/quadrat/signal/signal_im.txt";
    const char nameCoefRe[] = "./../tb/quadrat/coef/coef_re.txt";
    const char nameCoefIm[] = "./../tb/quadrat/coef/coef_im.txt";
    const char nameDelayCeil[] = "./../tb/quadrat/delay_ceil/delay_ceil.txt";
    const char nameDelayFrac[] = "./../tb/quadrat/delay_frac/delay_frac.txt";
    const char nameDnRe[] = "./../tb/quadrat/dn/dn_re.txt";
    const char nameDnIm[] = "./../tb/quadrat/dn/dn_im.txt";
#endif
    const char nameHfir[] = "./../tb/h_fir/h_fir.txt";
#endif
    // Заполняем входными данными и эталонами
    signal = readFromFileComp(nameSignRe, nameSignIm, POLAR*NP*CHP);
    coef = readFromFileComp(nameCoefRe, nameCoefIm, DNP*CHP);
    readFromFile(nameDelayCeil, del_c, DNP*CHP);
    readFromFile(nameDelayFrac, del_f, DNP*CHP);
    readFromFile(nameHfir, hfir, NPOLINOM*NF);
#else
    rand_mas(signal, POLAR*NP*CHP);
    rand_mas(coef, DNP*CHP);
    rand_mas(del_c, DNP*CHP);
    rand_mas(del_f, DNP*CHP);
    rand_mas(hfir, NPOLINOM*NF);
#endif

#if TIME_TEST == 0
    // Обнуляем массив
    for(int i=0; i<POLAR*DNP*(NP + NF - 1); i++)
        dn_out[i] = {0, 0};
    // Вычисляем FDN
    fdn(dn_out, signal, hfir, coef, del_f, del_c,
        POLAR, DNP, CHP, NF, NPOLINOM, NP, DMAX, cl_float2{0,0});
#endif
    // Обнуляем массив
    for(int i=0; i<POLAR*DNP*(NP + NF - 1); i++)
        dn_out_k[i] = {0, 0};
    // Запускаем вычисления
    fdn_gpu(dn_out_k, 0, 0, signal, coef, del_c, del_f, hfir);

#if TIME_TEST == 0

    // Сравниваем полученные данные
    int err_kern = compare(dn_out, dn_out_k, POLAR*DN*(NP + NF - 1));

    if(err_kern == 0)
        std::cout << "Data from kernel OK!" << std::endl;
#endif

    delete[] signal;
    delete[] coef;
    delete[] dn_out;
    delete[] dn_out_k;
    delete[] del_c;
    delete[] del_f;
    delete[] hfir;

    return 0;

}
