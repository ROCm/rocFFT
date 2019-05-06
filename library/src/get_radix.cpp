/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "radix_table.h"
#include "rocfft.h"
#include <stddef.h>
#include <vector>

std::vector<size_t> GetRadices(size_t length)
{

    std::vector<size_t> radices;

    // get number of items in this table
    std::vector<SpecRecord> specRecord  = GetRecord();
    size_t                  tableLength = specRecord.size();

    // printf("tableLength=%d\n", tableLength);
    for(int i = 0; i < tableLength; i++)
    {

        if(length == specRecord[i].length)
        { // if find the matched size

            size_t numPasses = specRecord[i].numPasses;
            // printf("numPasses=%d, table item %d \n", numPasses, i);
            for(int j = 0; j < numPasses; j++)
            {
                radices.push_back((specRecord[i].radices)[j]);
            }
            break;
        }
    }

    // if not in the table, then generate the radice order with the algorithm.
    if(radices.size() == 0)
    {
        size_t R = length;

        // Possible radices
        size_t cRad[]   = {13, 11, 10, 8, 7, 6, 5, 4, 3, 2, 1}; // Must be in descending order
        size_t cRadSize = (sizeof(cRad) / sizeof(cRad[0]));

        size_t workGroupSize;
        size_t numTrans;
        // need to know workGroupSize and numTrans
        DetermineSizes(length, workGroupSize, numTrans);
        size_t cnPerWI = (numTrans * length) / workGroupSize;

        // Generate the radix and pass objects
        while(true)
        {
            size_t rad;

            // Picks the radices in descending order (biggest radix first) performance
            // purpose
            for(size_t r = 0; r < cRadSize; r++)
            {

                rad = cRad[r];
                if((rad > cnPerWI) || (cnPerWI % rad))
                    continue;

                if(!(R % rad)) // if not a multiple of rad, then exit
                    break;
            }

            assert((cnPerWI % rad) == 0);

            R /= rad;
            radices.push_back(rad);

            assert(R >= 1);
            if(R == 1)
                break;

        } // end while
    } // end if(radices == empty)

    return radices;
}

// get working group size and number of transforms
void GetWGSAndNT(size_t length, size_t& workGroupSize, size_t& numTransforms)
{
    workGroupSize = 0;
    numTransforms = 0;

    // get number of items in this table
    std::vector<SpecRecord> specRecord  = GetRecord();
    size_t                  tableLength = specRecord.size();

    // printf("tableLength=%d\n", tableLength);
    for(int i = 0; i < tableLength; i++)
    {

        if(length == specRecord[i].length)
        { // if find the matched size

            workGroupSize = specRecord[i].workGroupSize;
            numTransforms = specRecord[i].numTransforms;
            break;
        }
    }
    // if not in the predefined table, then use the algorithm to determine
    if(workGroupSize == 0)
    {
        DetermineSizes(length, workGroupSize, numTransforms);
    }
}
