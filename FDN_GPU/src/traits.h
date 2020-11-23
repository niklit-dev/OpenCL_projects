#pragma once
#include <CL/cl.hpp>


// Type name
template<typename T> struct dtype_traits {
	static const char * getName() { return "Unsupported"; }
};
 
template<>
struct dtype_traits<float> {
    static const char* getName() { return "float"; }
};

template<>
struct dtype_traits<int> {
	static const char* getName() { return "int"; }
};

template<>
struct dtype_traits<cl_float2> {
	static const char* getName() { return "float2"; }
};


// Complex flag
template<typename T>
static bool iscplx() {
    return false;
}

template<>
//static bool iscplx<cl_float2>() {
bool iscplx<cl_float2>() {
    return true;
}

template<>
//static bool iscplx<float>() {
bool iscplx<float>() {
    return false;
}

template<>
//static bool iscplx<int>() {
bool iscplx<int>() {
    return false;
}
