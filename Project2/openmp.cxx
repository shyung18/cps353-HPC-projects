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
