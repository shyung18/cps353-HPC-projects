// $Smake: g++ -Wall -O3 -o %F %f -lcblas -latlas -lhdf5
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <limits.h>
#include <hdf5.h>

/* Check return values from HDF5 routines */
#define CHKERR(status,name) if ( status ) \
     fprintf( stderr, "Warning: nonzero status (%d) in %s\n", status, name )

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

/*----------------------------------------------------------------------------
 * Usage
 */
void usage( char* program_name)
{
    fprintf( stderr, "Usage: %s [-v] input_file output_file\n", program_name);
}

/*----------------------------------------------------------------------------
 * Read List
 */
void readList(char* fname, char* name, int** a, int* num)
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
  *a = (int*) malloc( num_elem * sizeof( int) );
  *num = num_elem;

  /* Create dataspace */
  dataspace_id = H5Screate_simple( rank, dims, NULL );

  /* Read matrix data from file */
  status = H5Dread( dataset_id, H5T_NATIVE_INT, dataspace_id,
                      file_dataspace_id, H5P_DEFAULT, *a );

  CHKERR( status, "H5Dread()" );

  /* Close resources */
  status = H5Sclose( dataspace_id ); CHKERR( status, "H5Sclose()" );
  status = H5Sclose( file_dataspace_id ); CHKERR( status, "H5Sclose()" );
  status = H5Dclose( dataset_id ); CHKERR( status, "H5Dclose()" );
  status = H5Fclose( file_id ); CHKERR( status, "H5Fclose()" );
  free( dims );
}

/*----------------------------------------------------------------------------
 * Write List
 */
void writeList( char* fname, char* name, int* a, int num)
{
  hid_t   file_id, group_id, dataspace_id, dataset_id;
  hsize_t dims[1];
  herr_t  status;

  /* Create HDF5 file.  If file already exists, truncate it */
  file_id = H5Fcreate( fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );


  /* Create the data space for dataset */
  dims[0] = num;
  dataspace_id = H5Screate_simple( 1, dims, NULL );

  /* Create the dataset */
  dataset_id = H5Dcreate( file_id, name, H5T_STD_I64LE, dataspace_id,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
  /* Write matrix data to file */
  status = H5Dwrite( dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, a ); CHKERR( status, "H5Dwrite()" );

  /* Close resources */
  status = H5Dclose( dataset_id ); CHKERR(status, "H5Dclose()");
  status = H5Sclose( dataspace_id ); CHKERR(status, "H5Sclose()");
  status = H5Fclose( file_id ); CHKERR( status, "H5Fclose()" );
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    char* in_name;
    char* out_name;
    int* a;
    int num;

    if ( argc != 3 )
    {
        usage( argv[0] );
        return EXIT_FAILURE;
    }
    in_name = argv[1];
    out_name = argv[2];

    readList( in_name, "/random_list", &a, &num);

    a = mergesort(a, 0, num-1);

    writeList(out_name, "/Sorted", a , num);



    // all done
    delete [] a;
    return 0;
}
