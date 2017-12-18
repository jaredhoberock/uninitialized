// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>

#include "uninitialized.hpp"

struct non_trivial_constructor
{
  __device__ non_trivial_constructor()
    : x(13)
  {
    printf("inside non_trivial_constructor with x %d and thread %d\n", x, threadIdx.x);
  }

  int x;
};

struct non_trivial_destructor
{
  __device__ non_trivial_destructor()
  {
    printf("inside non_trivial_destructor with x %d and thread %d\n", x, threadIdx.x);
  }

  int x;
};

struct trivial
{
  int x;
};

__global__ void kernel()
{
  // allow types with trivial constructors & destructors, but disallow initialization
  __shared__ trivial triv;
  
  __syncthreads();
  if(threadIdx.x == 0)
  {
    triv.x = 42;
  }
  __syncthreads();

  printf("Trivial types are allowed to be declared shared. Thread %2d sees triv's value is %d\n", threadIdx.x, triv.x);

  // disallow types with non-trivial constructor
  // __shared__ non_trivial_constructor y; // Error!

  // disallow types with non-trivial destructor
  // __shared__ non_trivial_destructor z; // Error!

  // wrap __shared__ variables with uninitialized, which has trivial constructor and destructor

  __shared__ uninitialized<non_trivial_constructor> y; // OK!

  // user explicitly constructs the wrapped object
  if(threadIdx.x == 0)
  {
    y.construct();
  }
  __syncthreads();

  __shared__ uninitialized<non_trivial_destructor> z; // OK!

  if(threadIdx.x == 0)
  {
    // access to underlying type
    // the explicit get() is unfortunate
    z.get().x = 13;
  }
  __syncthreads();

  // user explicitly destroys the wrapped object
  if(threadIdx.x == 0)
  {
    z.destroy();
  }
  __syncthreads();
}

int main()
{
  kernel<<<1,32>>>();
  cudaThreadSynchronize();
  return 0;
}

