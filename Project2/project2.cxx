// $Smake: g++ -Wall -O3 -o %F %f -lcblas -latlas -lhdf5 -fopenmp

#include <iostream>
#include <ctime>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <hdf5.h>
#include <math.h>
#include <omp.h>
#include "wtime.c"
extern "C" {
  #include <cblas.h>
}

/* Check return values from HDF5 routines */
#define CHKERR(status,name) if ( status ) \
     fprintf( stderr, "Warning: nonzero status (%d) in %s\n", status, name )

/*----------------------------------------------------------------------------
 * Usage
 */

void usage( char* program_name)
{
    fprintf( stderr, "Usage: %s [-v] [-e tol] [-m maxiter] input_file\n", program_name);
}

/*----------------------------------------------------------------------------
 * Read List
 */
void readList(char* fname, char* name, double** a, int* num)
{
  hid_t file_id, dataset_id, file_dataspace_id, dataspace_id;
  herr_t status;
  hsize_t* dims;
  int rank;
  int ndims;
  hsize_t num_elem;

  /* Open existing HDF5 file */
  file_id = H5Fopen( fname, H5F_ACC_RDONLY, H5P_DEFAULT );

  /* Open existing first dataset */
  dataset_id = H5Dopen( file_id, name, H5P_DEFAULT );

  /* Determine dataset parameters */
  file_dataspace_id = H5Dget_space( dataset_id );
  rank = H5Sget_simple_extent_ndims( file_dataspace_id );
  dims = (hsize_t*) malloc( rank * sizeof( hsize_t ) );
  ndims = H5Sget_simple_extent_dims( file_dataspace_id, dims, NULL );
  if ( ndims != rank )
  {
      fprintf( stderr, "Warning: expected dataspace to be dimension " );
      fprintf( stderr, "%d but appears to be %d\n", rank, ndims );
  }

  /* Allocate matrix */
  num_elem = H5Sget_simple_extent_npoints( file_dataspace_id );
  *a = (double*) malloc( num_elem * sizeof( double) );
  *num = num_elem;

  /* Create dataspace */
  dataspace_id = H5Screate_simple( rank, dims, NULL );

  /* Read matrix data from file */
  status = H5Dread( dataset_id, H5T_NATIVE_DOUBLE, dataspace_id,
                      file_dataspace_id, H5P_DEFAULT, *a );

  CHKERR( status, "H5Dread()" );

  /* Close resources */
  status = H5Sclose( dataspace_id ); CHKERR( status, "H5Sclose()" );
  status = H5Sclose( file_dataspace_id ); CHKERR( status, "H5Sclose()" );
  status = H5Dclose( dataset_id ); CHKERR( status, "H5Dclose()" );
  status = H5Fclose( file_id ); CHKERR( status, "H5Fclose()" );
  free( dims );
}

void PowerMethodLoops(char* nameOfFile, double* A, double* x, int size, double read_time, double tolerance, int maxiter)
{
  double t1;
  double t2;
  double compute_time = 0.0;
  double norm = 0.0;

  #pragma omp parallel for default(shared) reduction(+:norm)
  for (int i =0; i < size ; i ++)
  {
      norm += x [i] * x [i];
  }
  norm = sqrt ( norm );

  #pragma omp parallel for default(shared)
  for(int i=0; i<size; i++)
  {
    x[i] = (x[i] / norm);
  }

  double lambda_new = 0.0;
  double lambda_old = lambda_new + 2 * tolerance;
  double delta = fabs(lambda_new - lambda_old);
  int numiter = 0;
  double* y = new double[size];

  t1 = wtime();

  while (delta >= tolerance && numiter <= maxiter)
  {
    numiter++;

    #pragma omp parallel for default(shared)
    for(int i=0; i< size; i++)
    {
      y[i] = 0.0;
      double temp = 0.0;
      #pragma omp parallel for default(shared) reduction(+:temp)
      for(int j=0; j< size; j++)
      {
        temp += A[((i)*(size)+j)]*x[j];
      }
      y[i] = temp;
    }

    lambda_old = lambda_new;
    double dot_product = 0.0;

    #pragma omp parallel for default(shared) reduction(+:dot_product)
    for (int i=0; i<size; i++)
    {
      dot_product += x[i] * y[i];
    }

    lambda_new = dot_product;

    double norm2 = 0.0;

    #pragma omp parallel for default(shared) reduction(+:norm2)
    for (int i =0; i < size ; i ++)
    {
      norm2 += y [ i ] * y [ i ];
    }
    norm2 = sqrt ( norm2 );

    #pragma omp parallel for default(shared)
    for(int j=0; j<size; j++)
    {
       y[j] = y[j] / norm2;
       x[j] = y[j];
    }

    delta = fabs(lambda_new - lambda_old);
  }

  t2 = wtime();
  compute_time = t2 - t1;

    printf("%s Using Loops: \n", nameOfFile);

  if (numiter > maxiter)
  {
    printf("*** WARNING ****: maximum number of iterations exceeded, file=%s\n", nameOfFile);
  }

  printf("eigenvalue = %f found in %d iterations\n", lambda_new, numiter);
  printf("elapsed read time = %f seconds\n", read_time);
  printf("elapsed compute time = %f\n", compute_time);
}

void PowerMethodLoopsOpenMP(char* nameOfFile, double* A, double* x, int size, double read_time, double tolerance, int maxiter)
{
  double t1;
  double t2;
  double compute_time = 0.0;

  double norm = 0.0;
  double lambda_new = 0.0;
  double lambda_old = lambda_new + 2 * tolerance;
  double delta = fabs(lambda_new - lambda_old);
  int numiter = 0;
  double* y = new double[size];
  double dot_product = 0.0;

  #pragma omp parallel default(none) shared(x, A, size, tolerance, maxiter, y, norm, lambda_new, lambda_old, numiter, delta, dot_product)
  {
    // determine thread number and total number of threads
    const int num_threads = omp_get_max_threads();
    const int rank = omp_get_thread_num();
    int chunk = size/num_threads;
    int firstRow = rank*chunk;
    int lastRow = rank*chunk + chunk -1 ;
    int length = lastRow - firstRow + 1;

    for (int i =firstRow; i <= lastRow ; i ++)
    {
      #pragma omp critical (norm)
      {
        norm += x [i] * x [i];
      }
    }
    #pragma omp barrier

    for(int i=firstRow; i<=lastRow; i++)
    {
      x[i] = x[i] / sqrt(norm);
    }
    #pragma omp barrier

    while (delta >= tolerance && numiter <= maxiter)
    {
      #pragma omp master
      {
        numiter++;
      }

      for(int i=firstRow; i <= lastRow; i++)
      {
        y[i] = 0.0;
        for(int j=0; j< size; j++)
        {
          y[i] += A[((i)*(size)+j)]*x[j];
        }
      }

      #pragma omp master
      {
        lambda_old = lambda_new;
      }
      #pragma omp barrier

      for (int i=firstRow; i<= lastRow; i++)
      {
        #pragma omp critical (dot_product)
        {
          dot_product += x[i] * y[i];
        }
      }
      #pragma omp barrier

      #pragma omp master
      {
        lambda_new = dot_product;
        norm = 0.0;
      }
      #pragma omp barrier

      for (int i =firstRow; i <= lastRow ; i ++)
      {
        #pragma omp critical (norm)
        {
          norm += y [i] * y [i];
        }
      }
      #pragma omp barrier

      for(int j=firstRow; j<= lastRow; j++)
      {
         y[j] = y[j] / sqrt(norm);
         x[j] = y[j];
      }

      #pragma omp master
      {
        delta = fabs(lambda_new - lambda_old);
      }
      #pragma omp barrier
    }
  }

  printf("%s Using Loops: \n", nameOfFile);
  if (numiter > maxiter)
  {
    printf("*** WARNING ****: maximum number of iterations exceeded, file=%s\n", nameOfFile);
  }

  printf("eigenvalue = %f found in %d iterations\n", lambda_new, numiter);
  printf("elapsed read time = %f seconds\n", read_time);
  printf("elapsed compute time = %f\n", compute_time);

}


//----------------------------------------------------------------------------
void PowerMethodCBLAS(char* nameOfFile, double* A, double* x, int size, double read_time, double tolerance, int maxiter)
{
  double t1;
  double t2;
  double compute_time = 0.0;

  cblas_dscal(size, (1/cblas_dnrm2(size, x, 1)), x, 1);

  double lambda_new = 0.0;
  double lambda_old = lambda_new + 2 * tolerance;
  double delta = fabs(lambda_new - lambda_old);
  int numiter = 0;
  double* y = new double[size];

  t1 = wtime();
  while (delta >= tolerance && numiter <= maxiter)
  {
    numiter++;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1, A, size, x, 1, 0, y, 1);
    lambda_old = lambda_new;
    lambda_new = cblas_ddot(size, x, 1, y, 1);

    cblas_dscal(size, 1/cblas_dnrm2(size, y, 1), y, 1);
    cblas_dcopy(size, y, 1, x, 1);
    delta = fabs(lambda_new - lambda_old);

  }

  t2 = wtime();
  compute_time = t2 - t1;

  printf("%s Using CBLAS: \n", nameOfFile);
  if (numiter > maxiter)
  {
    printf("*** WARNING ****: maximum number of iterations exceeded, file=%s\n", nameOfFile);
  }

  printf("eigenvalue = %f found in %d iterations\n", lambda_new, numiter);
  printf("elapsed read time = %f seconds\n", read_time);
  printf("elapsed compute time = %f\n", compute_time);

}

int main( int argc, char* argv[] )
{
    char* nameOfFile;
    double* A;
    int num;
    double t1;
    double t2;

    int v = 0;
    double tolerance = 1e-6;
    int maxiter = 1000;
    int c;

    while ((c = getopt (argc, argv, "ve:m:")) != -1)
    {
      if(c== 'v')
      {
        v=1;
      }
      else if(c == 'e')
      {
        if(atoi(optarg)!=0)
        {
          tolerance = atof(optarg);
        }
        else
        {
          usage( argv[0] );
          return EXIT_FAILURE;
        }
      }
      else if(c == 'm')
      {
        if(atoi(optarg)!=0)
        {
          maxiter = atoi(optarg);
        }
        else
        {
          usage( argv[0] );
          return EXIT_FAILURE;
        }
      }
    }

    if(v>0)
    {
        printf("tolerance = %f\n", tolerance);
        printf("maximum number of iterations = %d\n", maxiter);
    }

    if ( argc < 2 )
    {
        usage( argv[0] );
        return EXIT_FAILURE;
    }
    nameOfFile = argv[argc-1];

    double read_time;

    t1 = wtime();
    readList( nameOfFile, "/A/value", &A, &num);
    t2 = wtime();
    read_time =  t2 - t1;

    int size = sqrt(num);
    double* x = new double[size];

    #pragma omp parallel for default(shared)
    for(int i=0; i<size; i++)
    {
      x[i] = 1;
    }

    //PowerMethodLoops(nameOfFile, A, x, size, read_time, tolerance, maxiter);
    //PowerMethodCBLAS(nameOfFile, A, x, size, read_time, tolerance, maxiter);

    PowerMethodLoopsOpenMP(nameOfFile, A, x, size, read_time, tolerance, maxiter);

  }
