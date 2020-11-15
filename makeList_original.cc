// $Smake: g++ -Wall -O3 -o %F %f

#include <iostream>
#include <ctime>
#include <limits.h>

//----------------------------------------------------------------------------
// Returns the number of seconds since some fixed arbitrary time in the past

double wtime( void )
{
    timespec ts;
    clock_gettime( CLOCK_MONOTONIC, &ts );
    return double( ts.tv_sec + ts.tv_nsec / 1.0e9 );
}

// Efficient variant b
int* merge(int* a, int lo, int m, int hi)
{
    int i, j, k;
    int* b = new int [m-lo+1];

    i=0; j=lo;
    // copy first half of array a to auxiliary array b
    while (j<=m)
        b[i++]=a[j++];

    i=0; k=lo;
    // copy back next-greatest element at each time
    while (k<j && j<=hi)
        if (b[i]<=a[j])
            a[k++]=b[i++];
        else
            a[k++]=a[j++];

    // copy back remaining elements of first half (if any)
    while (k<j)
        a[k++]=b[i++];
    delete [] b;
    return a;
}

int* mergesort(int* a, int lo, int hi)
{
    if (lo<hi)
    {
        int m=lo+(hi-lo)/2;
        mergesort(a, lo, m);
        mergesort(a, m+1, hi);
        return merge(a, lo, m, hi);
    }
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    // get length of list from command line
    int n = argc > 1 ? atoi( argv[1] ) : 0;
    if ( n <= 0 )
    {
        std::cerr << "usage: " << argv[0] << " N   (N is a positive integer)"
                  << std::endl;
        exit( EXIT_FAILURE );
    }

    // create list of random integers
    srandom( (unsigned int) time( NULL ) );
    int* a = new int [n];
    for ( int i = 0; i < n; i++ )
    {
	     a[i] = int( random() % INT_MAX );
    }

    for(int j=0; j< n; j++)
    {
      printf("Integer value before is %d\n" , a[j]);
    }
    // DO SOMETHING INTERESTING WITH THE LIST
    a = mergesort(a, 0, n-1);

    for(int j=0; j< n; j++)
    {
      printf("Integer value after is %d\n" , a[j]);
    }

    for(int j=0; j < n - 1; j++)
    {
      if(a[j] > a[j+1])
      {
        printf("Not sorted\n");
      }
    }



    // all done
    delete [] a;
    return 0;
}
