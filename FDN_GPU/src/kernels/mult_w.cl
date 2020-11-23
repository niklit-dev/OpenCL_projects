__kernel void mult_w(global T *w, global const float *d)
{
	int gx = get_group_id(0);
	T win = w[gx];
	w[gx] = mult_conv(win, d[gx]);
}