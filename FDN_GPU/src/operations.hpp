#pragma once
//#define __CL_HAS_ANON_STRUCT__
#include <CL/cl.hpp>
#include <type_traits>
#include <iostream>
#include <ctype.h>
#include <iomanip>
#include <fstream>


template <typename T>
void printTable(const T *mas, const int row, const int col);

/*
Создание перегруженных вариантов функций при помощи при помощи возвращаемого типа
или дополнительного неиспользуемого параметра в виде указателя на nullptr:
https://ru.cppreference.com/w/cpp/types/enable_if
*/

//--------------------------------------------------------------------------------//
//                                cl_float2 traits                                //
//--------------------------------------------------------------------------------//
// SFINAE to cl_float2
template<typename T>
struct is_float2 {
	static constexpr bool value = false;
};

template<>
	struct is_float2<cl_float2> {
	static constexpr bool value = true;
};
//--------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------//
//                             Arithmetic operations                              //
//--------------------------------------------------------------------------------//
// Complex additional for cl_float2
template<typename T>
typename std::enable_if_t<is_float2<T>::value, T> add(const T& lhs, const T& rhs)
{
	T out;
	out.x = lhs.x + rhs.x;
	out.y = lhs.y + rhs.y;
	return out;
}

// Complex subtraction for cl_float2
template<typename T>
typename std::enable_if_t<is_float2<T>::value, T> sub(const T& lhs, const T& rhs)
{
    T out;
    out.x = lhs.x - rhs.x;
    out.y = lhs.y - rhs.y;
    return out;
}

// Complex multiplication for cl_float2
template<typename T>
typename std::enable_if_t<is_float2<T>::value, T> mult(const T& lhs, const T& rhs)
{
	T out;
	out.x = lhs.x * rhs.x - lhs.y * rhs.y;
	out.y = lhs.x * rhs.y + lhs.y * rhs.x;
	return out;
}

// Additional for arithmetic types
template<typename T>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> add(const T& lhs, const T& rhs)
{
	T out;
	out = lhs + rhs;
	return out;
}

// Subtraction for arithmetic types
template<typename T>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> sub(const T& lhs, const T& rhs)
{
    T out;
    out = lhs - rhs;
    return out;
}

// Multiplication for arithmetic types
template<typename T>
typename std::enable_if_t<std::is_arithmetic<T>::value, T> mult(const T& lhs, const T& rhs)
{
	T out;
	out = lhs * rhs;
	return out;
}

cl_float2 mult(cl_float2 lhs, float rhs) {
    cl_float2 out;
    out.x = lhs.x*rhs;
    out.y = lhs.y*rhs;

    return out;
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                              Matrix manipulations                              //
//--------------------------------------------------------------------------------//
// Matrix multiplications with delay
template<typename T>
void matrixMult(const int Npoint, const int Nch, const int Ndn, const int Dmax,
                const int Pol, const T *x, const int *D, const T *w, T *y, T zero)
{
    int npd = 0;
    T xd;

    for (int dn = 0; dn < Ndn; dn++) {
        for (int np = 0; np < Npoint + Dmax; np++) {
			T acc = T{ 0 };
            for (int ch = 0; ch < Nch; ch++) {
                npd = np - D[ch*Ndn + dn];
                xd = (npd >= 0 && npd < Npoint) ? x[Pol*Npoint*Nch + npd*Nch + ch] : zero;
                acc = add(acc, mult(xd, w[ch*Ndn + dn]));
            }
            y[Pol*Ndn*Npoint + np*Ndn + dn] = acc;
		}
	}
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                                  Convolution                                   //
//--------------------------------------------------------------------------------//
template <typename T, typename Th>
void conv(T* y, const T *x, const Th *h, const int Np, const int NFil,
          const int Ndn, const int Pol, const int Mf)
{
    int L = Np + NFil - 1; // Количество отсчетов на выходе фильтра
    for (int dn = 0; dn < Ndn; ++dn) {
        for (int n = 0; n < Np + NFil - 1; ++n) {
            for (int m = 0; m < NFil; ++m) {
                if (n - m >= 0 && n - m < Np) {
                    y[Pol*Ndn*L + dn + n*Ndn] = add(y[Pol*Ndn*L + dn + n*Ndn],
                            mult(x[Pol*Ndn*Np + dn + (n-m)*Ndn], h[Mf*NFil + m]));
                }
            }
        }
    }
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                                   Multipl                                      //
//--------------------------------------------------------------------------------//
template <typename T>
void Multipl(T* w, const float* delay, const int Ndn, const int Nch)
{
    // Цикл поэлементного перемножения матриц
    for(int i=0;i<Ndn*Nch;i++)
    {
        w[i] = mult(w[i], delay[i]);
    }
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                                       FDN                                      //
//--------------------------------------------------------------------------------//
template <typename T, typename Th>
void fdn(T* y, const T *x, const Th *h, const T *w, const float *delay, const int *D,
         const int Npol, const int Ndn, const int Nch, const int Nfil, const int Nm,
         const int Npoint, const int Dmax, const T zero)
{
    // Массив для промежуточных значений вычислений
    T *mas_mul = new T[Npol*Ndn*Npoint+Dmax*128];
    T *inw = new T[Ndn*Nch];
    for(int i=0; i<Ndn*Nch; i++)
    {
        inw[i] = w[i];
    }
    // Цикл по порядку полинома
    for(int m=0;m<Nm;m++)
    {
        for(int p=0;p<Npol;p++)
        {
            // Перемножаем входные данные на матрицу коэффициентов
            matrixMult(Npoint, Nch, Ndn, Dmax, p, x, D, inw, mas_mul, zero);

            // Свертка с коэффициентами
            conv(y, mas_mul, h, Npoint, Nfil, Ndn, p, m);
        }
        // Домножаем коэффициенты на задержки
        Multipl(inw, delay, Ndn, Nch);
    }
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                              Display operations                                //
//--------------------------------------------------------------------------------//
void printVal(const cl_float2 val, int prec = 3)
{
    std::cout.precision(prec);
    std::cout << "(" << std::setw(prec+3) <<
                 val.x <<
//                 val.s[0] <<
                 std::flush << ", "
              << std::setw(prec+3) << val.y
//              << std::setw(prec+3) << val.s[1]
              << ")"<< std::flush;
}

void printVal(const int val, int prec = 3)
{
    std::cout.precision(prec);
    std::cout << std::setw(prec+3) << val;
}

void printVal(const float val, int prec = 3)
{
    std::cout.precision(prec);
    std::cout << std::fixed << std::setw(prec+3) << val;
}

template <typename T>
void printTable(const T *mas, const int row, const int col)
{
    std::cout << "-------------------------------------------------------------------------------" << std::endl;
    for(int r = 0;r < row; r++)
    {
        for(int c = 0; c < col; c++)
        {
            printVal(mas[r + c*row], 3);
            std::cout << "|";
        }
        std::cout << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
    }
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                    Функция чтения из файла входных данных                      //
//--------------------------------------------------------------------------------//
template<typename T>
void readFromFile(const char* fileName, T* mas, const int len)
{
    std::ifstream fin(fileName);
    if (!fin)
	{
        std::cout << "File " << fileName << " do not open!" << std::endl;
		exit(1);
	}
    else
        {
            // Запись данных в массив
            for(int i = 0; i < len; i++)
            {
                fin >> mas[i];
            }
            fin.close();
        }
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                     Заполнение комплексных чисел из файла                      //
//--------------------------------------------------------------------------------//
cl_float2 *readFromFileComp(const char* fileNameRe, const char* fileNameIm,
                            const int len)
{
    cl_float2 *mas = new cl_float2[len];
    float *masRe = new float[len];
    float *masIm = new float[len];

    readFromFile(fileNameRe, masRe, len);
    readFromFile(fileNameIm, masIm, len);

    for(int i=0; i<len;i++)
        mas[i] = cl_float2{masRe[i], masIm[i]};

    return mas;
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                                 Compare data                                   //
//--------------------------------------------------------------------------------//
template<typename T>
int compare(const T *mas1, const T *mas2, const int len)
{
    T subst;

    for(int i=0; i<len; i++)
    {
        subst = sub(mas1[i], mas2[i]);
        if(subst.x > 0.0001 || subst.y > 0.0001)
        {
            float re = subst.x;
            float im = subst.y;
            std::cout << "Error in " << i << " number "
                      << "re = " << re << " im = " << im << std::endl;
            return 1;
        }
    }
    return 0;
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                         Запись кода кернела в строку                           //
//--------------------------------------------------------------------------------//
std::string fileRead(const char *fileName) {
    std::ifstream srcfile(fileName);
    srcfile.unsetf(std::ios::skipws); // Отключение пропуска пробельных символов
    std::string src((std::istreambuf_iterator<char>(srcfile)),
                    (std::istreambuf_iterator<char>()));
    return src;
}
//--------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------//
//                                    Random                                      //
//--------------------------------------------------------------------------------//
void rand_mas(cl_float2 *mas, int len)
{
    for(int i=0; i<len; i++)
    {
        mas[i].x = (static_cast<float>(std::rand()))/RAND_MAX;
        mas[i].y = (static_cast<float>(std::rand()))/RAND_MAX;
    }
}

void rand_mas(float *mas, int len)
{
    for(int i=0; i<len; i++)
    {
        mas[i] = (static_cast<float>(std::rand()))/RAND_MAX;
    }
}

void rand_mas(int *mas, int len)
{
    for(int i=0; i<len; i++)
    {
        mas[i] = std::rand()%4;
    }
}
//--------------------------------------------------------------------------------//


