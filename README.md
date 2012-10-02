`uninitialized`
==============

Occasionally, it is desirable to explicitly manage the construction and destruction of stack-allocated variables.
Doing so with explicit use of untyped buffers, placement new, and delete can be cumbersome. This library solves those
problems.

Demo
====

The following code demonstrates how to use `uninitialized` from within a CUDA kernel
on variables in `__shared__` memory to explicitly control which CUDA thread is responsible for managing the
lifetime of the variable:

```
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
```

