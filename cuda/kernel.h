#pragma once
#include <string>

extern "C" void __declspec(dllexport) runFilter3(unsigned char* filter, unsigned char divisor_,
	unsigned char offset_, const char* cFileame, const unsigned char compare = 0);

extern "C" void __declspec(dllexport) runFilter5(unsigned char* filter, unsigned char divisor_,
	unsigned char offset_, const char* cFileame, const unsigned char compare = 0);