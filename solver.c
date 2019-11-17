
#define ABS(A) ((A) >=0.0 ? (A) : -(A))

static void iter_jacobi(double *A, double *B, double *R, int nx, int ny, double err){
  
#define EM(i,j) A[j *nx +i]
#define FM(i,j) B[j *nx +i]
#define RM(i,j) R[j *nx +i]  

  int running = 1;
  double vsum = 0.0;
  int nc      = 0;
  double verr = 0.0; 
  
  for (;running;){
    for (int j= 1; j < ny -1;++j){
      for (int i= 1; i < nx -1;++i){
	FM(i,j) = (EM(i,j+1) + EM(i+1,j) + EM(i-1,j) + EM(i,j-1) -RM(i,j)) * 0.25;
      }
    }
    vsum = 0.0;
    nc   = 0;
    verr = 0.0; 
    for (int j= 1; j < ny -1;++j){
      for (int i= 1; i < nx -1;++i){
	vsum += ABS((FM(i,j) -EM(i,j)));
	  nc++;
      }
    }
    verr = vsum / (double)(nc);
    if (verr <= err) running = 0;
  }
  
#undef EM
#undef FM
#undef RM


}
