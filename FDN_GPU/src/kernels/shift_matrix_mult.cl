#if CPLX

T mult(T lhs, T rhs) {
    T out;
    out.x = lhs.x*rhs.x - lhs.y*rhs.y;
    out.y = lhs.x*rhs.y + lhs.y*rhs.x;
    return out;
}

T add(T lhs, T rhs) {
    T out;
    out.x = lhs.x + rhs.x;
    out.y = lhs.y + rhs.y;
    return out;
}

#else

T mult(T lhs, T rhs) {
	T out;
	out = lhs * rhs;
	return out;
}

T add(T lhs, T rhs) {
	T out;
	out = lhs + rhs;
	return out;
}

#endif
//=====================================================================================================
#if RECTANGLE_TILE == 1
	
	
__kernel void shift_matrix_mult (const __global T* S,   // Input signal
								 const __global int* D,	// Integer delay
								 const __global T* W,	// Turning coefficients
								       __global T* Y,   // Output diagrams
								 const          int Pol // Polarization
						        )    
{    	
	int lx = get_local_id(0); // Local address on the tile to the side of the diagrams (max TS_DN)
	int ly = get_local_id(1); // Local address on the tile in the direction of the signal (max TS_N)
	int gx = get_local_size(0)*get_group_id(0) + lx; // The global address on the diagrams (max DNP)
	int gy = get_local_size(1)*get_group_id(1) + ly; // The global address on the signal (max NP)
	
	// Local memory
	__local T Ssub[TS_N + DMAX][TS_CH]; // Signal
	__local T Wsub[TS_CH][TS_DN]; // Turning coefficients
	__local int Dsub[TS_CH][TS_DN]; // Integer delay
	
	// Initializing the accumulator register
	T acc = (T)(0);
	// Number of tile that fit in the receivers
	const int numTiles = CHP/TS_CH;	
	
	// Loop through the numbers tiles
	for (int t = 0; t < numTiles; ++t) {			
		// Write the signal into local memory for one tile
		// Loop on the receivers
		for(int sub_x = lx, x = gx; sub_x < TS_CH; sub_x+=get_local_size(0), x += get_local_size(0)) 
		{
			// Signal loop for a single receiver
			for(int sub_y = ly, y = gy; sub_y < TS_N+DMAX; sub_y+=get_local_size(1), y += get_local_size(1)) 
			{
				bool cond_y = y - DMAX >= 0 && y - DMAX < NP; // Checking for signal excess
				Ssub[sub_y][sub_x] = (cond_y) ? S[Pol*NP*CHP + (y - DMAX)*CHP + TS_CH*t + sub_x] : (T)(0);
			}
		}

		// Write the coefficients and delays into local memory for one tile
		for(int sub_y = ly, y = gy; sub_y < TS_CH; sub_y+=get_local_size(1), y += get_local_size(1)) // Loop on the receivers
		{
			bool cond_y = y >= 0 && y < NP; // Checking for signal excess
			Wsub[sub_y][lx] = (cond_y) ? W[gx + (TS_CH*t + sub_y)*DNP] : (T)(0);
			Dsub[sub_y][lx] = (cond_y) ? D[gx + (TS_CH*t + sub_y)*DNP] : (0);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);		
		// Calculation diagrams for this tile
		#pragma unroll
		for (int k = 0; k < TS_CH; ++k) {
			acc = add(acc, mult(Ssub[ly + DMAX - Dsub[k][lx]][k], Wsub[k][lx]));

		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	Y[Pol*NP*DNP + gy*get_global_size(0) + gx] = acc;
}

#endif
//=====================================================================================================
#if RECTANGLE_TILE_WITH_REGISTER == 1

__kernel void shift_matrix_mult (const __global T* S,	// Input signal
						         const __global int* D, // Integer delay
						         const __global T* W,	// Turning coefficients
								       __global T* Y,	// Output diagrams
						         const          int Pol // Polarization
						        )    
{    	
	int lx = get_local_id(0); // Local address on the tile to the direction of the diagrams (max TS_DN)
	int ly = get_local_id(1); // Local address on the tile in the direction of the signal (max TS_N/WPT_N)
	int gx = get_local_size(0)*get_group_id(0) + lx; // The global address on the diagrams (max DNP)
	int gy = get_local_size(1)*get_group_id(1) + ly; // The global address on the signal (max NP/WPT_N)

	// Local memory
	__local T Ssub[TS_N + DMAX][TS_DN];
	__local T Wsub[TS_CH][TS_DN];
	__local int Dsub[TS_CH][TS_DN];
	
	// Initializing the array accumulator registers
	T acc[WPT_N];
	// Number of tile that fit in the receivers
	const int numTiles = CHP/TS_CH;	
	// Zeroing an array
	for(int w = 0; w < WPT_N; w++)
	{
		acc[w] = (T)(0);
	}
	
	// Loop through the numbers tiles
	for (int t = 0; t < numTiles; ++t) 
	{	
		// Write the signal into local memory for one tile
		// Loop on the receivers
		for(int sub_x = lx, x = gx; sub_x < TS_CH; sub_x+=get_local_size(0), x += get_local_size(0))
		{
			bool cond_x = x >= 0 && x < DNP; // Checking for diagrams excess
			// Signal loop for a single receiver
			for(int sub_y = ly, y = gy; sub_y < TS_N+DMAX; sub_y+=get_local_size(1), y += get_local_size(1))
			{
				bool cond_y = y - DMAX >= 0 && y - DMAX < NP; // Checking for signal excess
				Ssub[sub_y][sub_x] = (cond_y && cond_x) ? S[Pol*NP*CHP + (y - DMAX)*CHP + TS_CH*t + sub_x] : (T)(0);
			}
		}
		
		// Write the coefficients and delays into local memory for one tile
		for(int sub_y = ly, y = gy; sub_y < TS_CH; sub_y+=get_local_size(1), y += get_local_size(1)) // каналы
		{
			bool cond_y = y >= 0 && y < NP; // Checking for signal excess
			Wsub[sub_y][lx] = (cond_y) ? W[gx + (TS_CH*t + sub_y)*DNP] : (T)(0);
			Dsub[sub_y][lx] = (cond_y) ? D[gx + (TS_CH*t + sub_y)*DNP] : (0);
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);		
		
		// Calculation diagrams for this tile
		for (int k = 0; k < TS_CH; ++k) {
			for(int w = 0; w < WPT_N; w++)
			{
				acc[w] = add(acc[w], mult(Ssub[ly + w*RTS_N + DMAX - Dsub[k][lx]][k], Wsub[k][lx]));
			}
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// Output diagrams 
	for(int w = 0; w < WPT_N; w++)
	{
		Y[Pol*NP*DNP + (gy + w*RTS_N)*get_global_size(0) + gx] = acc[w];
	}
}

#endif
//=====================================================================================================
#if QUADRATE_TILE == 1

__kernel void shift_matrix_mult (const __global T* S,	 // Input signal
                                 const __global int* D,	 // Integer delay
                                 const __global T* W,	 // Turning coefficients
                                       __global T* Y,	 // Output diagrams
						         const          int Pol) // Polarization
{    	
	int lx = get_local_id(0); // Local address on the tile to the direction of the diagrams (max TS)
	int ly = get_local_id(1); // Local address on the tile in the direction of the signal (max TS)
	int gx = get_local_size(0)*get_group_id(0) + lx; // The global address on the diagrams (max DNP)
	int gy = get_local_size(1)*get_group_id(1) + ly; // The global address on the signal (max NP)

	// Local memory 
	__local T Ssub[TS + DMAX][TS];
	__local T Wsub[TS][TS];
	__local int Dsub[TS][TS];
	
	// Initializing the array accumulator registers
	T acc = (T)(0);
	// Number of tile that fit in the receivers
	const int numTiles = CHP/TS;
	
	// Loop through the numbers tiles
	for (int t = 0; t < numTiles; ++t) {	
		// Write the signal into local memory for one tile
		// Loop on the receivers
		for (int sub_x = lx, x = gx; sub_x < TS; sub_x += get_local_size(0), x += get_local_size(0)) {
			bool cond_x = x >= 0 && x < DNP; // Checking for diagrams excess
			// Signal loop for a single receiver
			for (int sub_y = ly, y = gy; sub_y < TS + DMAX; sub_y += get_local_size(1), y += get_local_size(1)) {
				bool cond_y = y - DMAX >= 0 && y - DMAX < NP; // Checking for signal excess
				Ssub[sub_y][sub_x] = (cond_y && cond_x) ? S[Pol*NP*CHP + (y - DMAX)*CHP + TS*t + sub_x] : (T)(0);
			}
		}
		
		// Write the coefficients and delays into local memory for one tile
		Wsub[ly][lx] = W[gx + (TS*t + ly)*DNP];
		Dsub[ly][lx] = D[gx + (TS*t + ly)*DNP];
		
		barrier(CLK_LOCAL_MEM_FENCE);

		// Calculation diagrams for this tile
		for (int k = 0; k < TS; ++k) {
			acc = add(acc, mult(Ssub[ly + DMAX - Dsub[k][lx]][k], Wsub[k][lx]));

		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	Y[Pol*NP*DNP + gy*get_global_size(0) + gx] = acc;
}

#endif
