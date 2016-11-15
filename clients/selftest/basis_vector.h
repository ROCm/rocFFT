/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef BASIS_VECTOR_H
#define BASIS_VECTOR_H
#include <math.h>
#include <iostream>

#define __HIPCC__
#include <hip/hip_runtime_api.h>

enum FftBasisChoice
{
	FBC_RANDOM,
	FBC_ALL,
	FBC_EVEN,
	FBC_ODD,
	FBC_PRIME,
	FBC_FIBONACCI,
};




// A class to create basis vector mix (complex to complex)

template <typename T, typename CT>
class FftBasisVectorMixComplex
{
	size_t B; // no. of transforms
	size_t N[3];
	size_t L;

	CT *mix;//time domain as input
	CT *ref;//freq domain
	CT *otp;//freq domain produced by the library to be tested against ref

	CT *dev; // device buffer

	FftBasisChoice mfbc;

public:
	FftBasisVectorMixComplex(size_t dim, size_t *length, size_t batch, FftBasisChoice fbc) :
		mfbc(fbc),
		mix(NULL),
		ref(NULL),
		otp(NULL),
		dev(NULL)
	{
		B = batch;

		N[0] = 1;
		N[1] = 1;
		N[2] = 1;

		L = 1;
		for(size_t i=0; i<dim; i++)
		{
			N[i] = length[i];
			L *= N[i];
		}


		mix = (CT *)malloc(L * B * sizeof(CT));
		ref = (CT *)malloc(L * B * sizeof(CT));
		otp = (CT *)malloc(L * B * sizeof(CT));

		hipMalloc(&dev, L * B * sizeof(CT));
	}

	void RawPtrs(CT **in, CT **out, CT **devb)
	{
		*in = mix;
		*out = otp;
		*devb = dev;
	}

	void Generate(CT **data, CT **mag, int dir = -1)
	{
		srand((unsigned int)time(NULL));

		memset(mix, 0, L * B *sizeof(CT));
		memset(ref, 0, L * B *sizeof(CT));

		// Create a Q-point data (in total) for all dimensions put together
		// in essence creating a mix of Q number of basis vectors
		typedef size_t BvSpot[3];
		BvSpot *bvSpot;
		const size_t Q = L < 16 ? L : 16;
		bvSpot = (BvSpot *)malloc(Q * sizeof(BvSpot));

		assert(Q <= 128);

		// generate unique indices so basis vectors are not repeated
		for(size_t q=0; q<Q; q++)
		{
			size_t a, b, c;
			bool unique = true;

			do
			{
				a = (rand()%(N[0]));
				b = (rand()%(N[1]));
				c = (rand()%(N[2]));

				unique = true;
				for(size_t r=0; r<q; r++)
				{
					if( (bvSpot[r][0] == a) && (bvSpot[r][1] == b) && (bvSpot[r][2] == c) )
						unique = false;
				}
			} while(!unique);

			bvSpot[q][0] = a;
			bvSpot[q][1] = b;
			bvSpot[q][2] = c;
		}


		for(size_t h=0; h<B; h++)
		{
			for(size_t q=0; q<Q; q++)
			{
				T c = (T)(rand()%10);//amplitude
				T d = (T)(rand()%10);//amplitude

				ref[h*L + bvSpot[q][2]*N[1]*N[0] + bvSpot[q][1]*N[0] + bvSpot[q][0]][0] = c;
				ref[h*L + bvSpot[q][2]*N[1]*N[0] + bvSpot[q][1]*N[0] + bvSpot[q][0]][1] = d;

				for(size_t idx2=0; idx2<N[2]; idx2++)
				{
					for(size_t idx1=0; idx1<N[1]; idx1++)
					{
						for(size_t idx0=0; idx0<N[0]; idx0++)
						{
							const double TWO_PI = 6.283185307179586476925286766559;
							double theta0 = TWO_PI * (((double)idx0)/((double)N[0])) * ((double)bvSpot[q][0]);
							double theta1 = TWO_PI * (((double)idx1)/((double)N[1])) * ((double)bvSpot[q][1]);
							double theta2 = TWO_PI * (((double)idx2)/((double)N[2])) * ((double)bvSpot[q][2]);

							double dr = (dir == -1) ? 1.0 : -1.0;

							T a = (T)cos(dr*(theta0 + theta1 + theta2));
							T b = (T)sin(dr*(theta0 + theta1 + theta2));

							mix[h*L + idx2*N[1]*N[0] + idx1*N[0] + idx0][0] += (a*c - b*d);
							mix[h*L + idx2*N[1]*N[0] + idx1*N[0] + idx0][1] += (b*c + a*d);
						}
					}
				}
			}
		}


		free(bvSpot);


		*data = mix;
		*mag = ref;
	}

	~FftBasisVectorMixComplex()
	{
		if(mix)
			free(mix);

		if(ref)
			free(ref);

		if(otp)
			free(otp);

		if(dev)
			hipFree(dev);
	}
};


#endif // BASIS_VECTOR_H

