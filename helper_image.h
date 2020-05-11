/////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////

// These are helper functions for the SDK samples (image,bitmap)
#ifndef HELPER_IMAGE_H
#define HELPER_IMAGE_H

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>

#include <assert.h>
#include <math.h>

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

// namespace unnamed (internal)
namespace 
{
    //! size of PGM file header 
    const unsigned int PGMHeaderSize = 0x40;
}

#ifdef _WIN32
    #ifndef FOPEN
	#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
    #endif
    #ifndef FOPEN_FAIL
	#define FOPEN_FAIL(result) (result != 0)
    #endif
    #ifndef SSCANF
	#define SSCANF sscanf_s
    #endif
#else
    #ifndef FOPEN
	#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
    #endif
    #ifndef FOPEN_FAIL
	#define FOPEN_FAIL(result) (result == NULL)
    #endif
    #ifndef SSCANF
	#define SSCANF sscanf
    #endif
#endif

inline bool
__loadPPM( const char* file, unsigned char** data, 
         unsigned int *w, unsigned int *h, unsigned int *channels ) 
{
    FILE *fp = NULL;
    if( FOPEN_FAIL(FOPEN(fp, file, "rb")) ) 
    {
        std::cerr << "__LoadPPM() : Failed to open file: " << file << std::endl;
        return false;
    }

    // check header
    char header[PGMHeaderSize];
    if (fgets( header, PGMHeaderSize, fp) == NULL) {
       std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
       return false;
    }
    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else {
        std::cerr << "__LoadPPM() : File is not a PPM or PGM image" << std::endl;
        *channels = 0;
        return false;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    while(i < 3) 
    {
        if (fgets(header, PGMHeaderSize, fp) == NULL) {
            std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
            return false;
        }
        if(header[0] == '#') 
            continue;

        if(i == 0) 
        {
            i += SSCANF( header, "%u %u %u", &width, &height, &maxval);
        }
        else if (i == 1) 
        {
            i += SSCANF( header, "%u %u", &height, &maxval);
        }
        else if (i == 2) 
        {
            i += SSCANF(header, "%u", &maxval);
        }
    }

    // check if given handle for the data is initialized
    if( NULL != *data) 
    {
        if (*w != width || *h != height) 
        {
            std::cerr << "__LoadPPM() : Invalid image dimensions." << std::endl;
        }
    } 
    else 
    {
        *data = (unsigned char*) malloc( sizeof( unsigned char) * width * height * *channels);
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread( *data, sizeof(unsigned char), width * height * *channels, fp) == 0) {
        std::cerr << "__LoadPPM() read data returned error." << std::endl;
    }
    fclose(fp);

    return true;
}

inline bool
__savePPM( const char* file, unsigned char *data, 
         unsigned int w, unsigned int h, unsigned int channels) 
{
    assert( NULL != data);
    assert( w > 0);
    assert( h > 0);

    std::fstream fh( file, std::fstream::out | std::fstream::binary );
    if( fh.bad()) 
    {
        std::cerr << "__savePPM() : Opening file failed." << std::endl;
        return false;
    }

    if (channels == 1)
    {
        fh << "P5\n";
    }
    else if (channels == 3) {
        fh << "P6\n";
    }
    else {
        std::cerr << "__savePPM() : Invalid number of channels." << std::endl;
        return false;
    }

    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) 
    {
        fh << data[i];
    }
    fh.flush();

    if( fh.bad()) 
    {
        std::cerr << "__savePPM() : Writing data failed." << std::endl;
        return false;
    } 
    fh.close();

    return true;
}

#endif // HELPER_IMAGE_H
