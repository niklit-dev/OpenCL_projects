#if CPLX

T mult_conv(T lhs, Th rhs) {
	T out;
	out.x = lhs.x*rhs;
	out.y = lhs.y*rhs;
	return out;
}

#else

T mult_conv(T lhs, Th rhs) {
	T out;
	out = lhs*rhs;
	return out;
}

#endif
//=====================================================================================================
#if LOCAL_H_LOCAL_SIGNAL == 1

__kernel void convolve(      __global T *y,    // Output signal
					   const __global T *x,    // Input signal
					   const __global Th *h,   // Coeficients FIR
					   const          int Pol, // Polarization
					   const          int M    // Number polinom
					   ) 
{
	int padding = NF - 1;   // Увеличение размера выходного массива
    int L = (NP + NF - 1)*DNP; // Общее количество отсчетов
	// Global
	int gx = get_group_id(0)*get_local_size(0)*DNP+get_local_id(0)*DNP; // Координата по х
	int gy = get_group_id(1);     // Координата по у		
	
	__local T localMem[THREADS + 2*(NF - 1)];	
	__local Th localH[NF];
	
	int locx = get_local_id(0);
	if(locx < NF)
		localH[locx] = h[M*NF + locx];

	for(int i = 0; i<NF; i++)
	{
		localH[i] = h[M*NF + i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Закачиваем часть сигнала в локальную память
	for(int sub_x = get_local_id(0), y = gx; sub_x < THREADS + 2*(NF - 1); sub_x+=get_local_size(0), y += get_local_size(0)*DNP) 
	{
		int idx = y - padding*DNP;
		localMem[sub_x] = (idx >= 0 && idx < NP*DNP) ? x[Pol*DNP*NP + idx + get_group_id(1)] : (T)(0);
	}	

        barrier(CLK_LOCAL_MEM_FENCE);
	
	T accum = (T)(0);
	// Локаоьное положение по х
	int lx = get_local_id(0) + padding;
	
	// Цикл по локальной памяти
	#pragma unroll
	for(int f = 0; f<NF; f++)
	{
		accum += mult_conv(localMem[lx - f], localH[f]);		
	}

	y[Pol*DNP*(NP + NF - 1) + gx + get_group_id(1)] += (T)accum;	
}

#endif
//=====================================================================================================
#if LOCAL_SIGNAL == 1

__kernel void convolve(      __global T *y, 
					   const __global T *x, 
					   const __global Th *h,
					   const          int Pol, 
					   const          int M) 
{
	int padding = NF - 1;   // Увеличение размера выходного массива
    int L = (NP + NF - 1)*DNP; // Общее количество отсчетов
	// Global
	int gx = get_group_id(0)*get_local_size(0)*DNP+get_local_id(0)*DNP; // Координата по х
	int gy = get_group_id(1);     // Координата по у		
	
	__local T localMem[THREADS + 2*(NF - 1)];
	
	// Закачиваем часть сигнала в локальную память
	for(int sub_x = get_local_id(0), y = gx; sub_x < THREADS + 2*(NF - 1); sub_x+=get_local_size(0), y += get_local_size(0)*DNP) 
	{
		int idx = y - padding*DNP;
		localMem[sub_x] = (idx >= 0 && idx < NP*DNP) ? x[Pol*DNP*NP + idx + get_group_id(1)] : (T)(0);
	}	

    barrier(CLK_LOCAL_MEM_FENCE);

	T accum = (T)(0);
	// Локаоьное положение по х
	int lx = get_local_id(0) + padding;
	
	// Цикл по локальной памяти
	#pragma unroll
	for(int f = 0; f<NF; f++)
	{
		accum += mult_conv(localMem[lx - f], h[M*NF + f]);
	}

	y[Pol*DNP*(NP + NF - 1) + gx + get_group_id(1)] += (T)accum;
	
}

#endif
