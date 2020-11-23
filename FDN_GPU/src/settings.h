#ifndef SETTINGS_H
#define SETTINGS_H

#define MODEL_TEST 1
//#define QT_DEBUG
#define INFO
//#define POL2
#define TIME_TEST 1
#define RECTANGLE_TILE 1
#define RECTANGLE_TILE_WITH_REGISTER 0
#define QUADRATE_TILE 0

#define LOCAL_H_LOCAL_SIGNAL 0
#define LOCAL_SIGNAL 1

#if TIME_TEST == 1

    #define TS_N 64 // Количество потоков (должно быть кратно ДН)
    #define TS_CH 18 // Количество потоков (должно быть кратно ДН)
	#define TS_DN 18 // Количество потоков (должно быть кратно ДН)
	
	#define WPT_N 8 // Количество регистров внутренней памяти по данным
	#define RTS_N TS_N/WPT_N // Уменьшенный размер окна по сигналу
	
	#define TS 18 // Количество потоков (должно быть кратно ДН)

    #define N 10000 // Количество отсчётов входных данных
#ifdef POL2
    #define CH 64 // Количество поляризаций
#else
    #define CH 64*2 // Количество каналов
#endif
    #define DN 18 // Количество диаграмм
    #define NF 48 // Порядок фильтра
    #define THREADS 256 // Размер окна для свертки

#else
    
	#define TS_N 6 // Количество потоков (должно быть кратно ДН)
	#define TS_CH 4 // Количество потоков (должно быть кратно ДН)
	#define TS_DN 2 // Количество потоков (должно быть кратно ДН)

	#define WPT_N 2 // Количество регистров внутренней памяти по данным
	#define RTS_N TS_N/WPT_N // Уменьшенный размер окна по сигналу
	
	#define TS 4 // Количество потоков

    #define N 50000 // Количество отсчётов входных данных
    #define CH 4 // Количество каналов
    #define DN 2 // Количество диаграмм
    #define NF 10 // Порядок фильтра
    #define THREADS 256 // Размер окна для свертки

#endif

#ifdef POL2
    #define POLAR 2 // Количество поляризаций
#else
    #define POLAR 1 // Количество поляризаций
#endif

#define NPOLINOM 5 // Порядок полинома
#define DMAX 3 // Максимальное значение целоочисленной задержки

#define DIVUP(a, b) (((a) + (b) - 1) / (b))

#if RECTANGLE_TILE == 1 || RECTANGLE_TILE_WITH_REGISTER == 1
	#define NP (DIVUP(N, TS_N)*TS_N)
	#define CHP (DIVUP(CH, TS_CH)*TS_CH)
	#define DNP (DIVUP(DN, TS_DN)*TS_DN)
#elif QUADRATE_TILE == 1
	#define NP (DIVUP(N, TS)*TS)
	#define CHP (DIVUP(CH, TS)*TS)
	#define DNP (DIVUP(DN, TS)*TS)
#endif

#endif
