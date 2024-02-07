/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Block size */
#define mc 256
#define kc 128
#define nb 1000

#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc );
void PackMatrixA( int, double *, int, double * );
void PackMatrixB( int, double *, int, double * );

void InnerKernel( int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc, int first_time );
    
/* We block the matrix C as mc x n, the matrix A as mc x kc, the matrix B as kc x nc */
void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
    int i, j, p, pb, ib;
    
    /* This time, we compute a mc x n block of C by a call to the InnerKernel */
    for ( p=0; p<k; p+=kc ) {
        pb = min( k-p, kc ); // if left colums of Matrix B less than kc, then column of blocksize is k-p 
        for ( i=0; i<m; i+=mc) {
            ib = min( m-i, mc ); // if left rows of Matrix A less than kc, then column of blocksize is m-i 
            InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc, i==0 ); // i==0 is flag to indicate if we need PackMatrixB
        }
    }
}

/* This InnerKernel is old MY_MMult to calucate Matrix Multiplication with 4x4 blocking */
void InnerKernel( int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc, int first_time )
{
  int i, j;
  double packedA[ m * k ];
  static double packedB[ kc*nb ];  /* Note: using a static buffer is not thread safe... */

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    if (first_time)
      PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in one routine (four inner products) */
      if (j==0) PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      AddDot4x4( k, &packedA[ i*k ], 4, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}

void PackMatrixA( int k, double *a, int lda, double *a_to )
{
  int j;

  for( j=0; j<k; j++){  /* loop over columns of A */
    double 
      *a_ij_pntr = &A( 0, j );

    *a_to++ = *a_ij_pntr;
    *a_to++ = *(a_ij_pntr+1);
    *a_to++ = *(a_ij_pntr+2);
    *a_to++ = *(a_ij_pntr+3);
  }
}

void PackMatrixB( int k, double *b, int ldb, double *b_to )
{
  int i;
  double 
    *b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
    *b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  for( i=0; i<k; i++){  /* loop over rows of B */
    *b_to++ = *b_i0_pntr++;
    *b_to++ = *b_i1_pntr++;
    *b_to++ = *b_i2_pntr++;
    *b_to++ = *b_i3_pntr++;
  }
}

/* include header file for intrinsic function which can use vector operation */
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

/* define a union for vector (2 double) and double array translation */
typedef union
{
  __m128d v;
  double d[2];
} v2df_t;

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 
     
     in this version, we merge each set of four loops, computing four
     inner products simultaneously. 
     
     in this version, we accumulate in registers and put A( 0, p ) in a register 
     in this version, we use pointer to track where in four columns of B we are
     */ 

    int p;
    
    
    v2df_t
        c_00_c_10_vreg,    c_01_c_11_vreg,    c_02_c_12_vreg,    c_03_c_13_vreg,
        c_20_c_30_vreg,    c_21_c_31_vreg,    c_22_c_32_vreg,    c_23_c_33_vreg,
        a_0p_a_1p_vreg,
        a_2p_a_3p_vreg,
        b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 
    
    double
        /* Point to the current elements in the four columns of B */
        *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 
    
    b_p0_pntr = &B( 0, 0 );
    b_p1_pntr = &B( 0, 1 );
    b_p2_pntr = &B( 0, 2 );
    b_p3_pntr = &B( 0, 3 );

    /*
    c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
    c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
    c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
    c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;
    */
    
    /* initialize the vectors */
    c_00_c_10_vreg.v = _mm_setzero_pd();   
    c_01_c_11_vreg.v = _mm_setzero_pd();
    c_02_c_12_vreg.v = _mm_setzero_pd(); 
    c_03_c_13_vreg.v = _mm_setzero_pd(); 
    c_20_c_30_vreg.v = _mm_setzero_pd();   
    c_21_c_31_vreg.v = _mm_setzero_pd();  
    c_22_c_32_vreg.v = _mm_setzero_pd();   
    c_23_c_33_vreg.v = _mm_setzero_pd(); 
    
    //  inner product
    for ( p=0; p<k; p++ ){
        /* read A(,p) into vector register */
        a_0p_a_1p_vreg.v = _mm_load_pd( (double *) &A( 0, p ) );
        a_2p_a_3p_vreg.v = _mm_load_pd( (double *) &A( 2, p ) );
        
        /* read B into vector register */
        b_p0_vreg.v = _mm_loaddup_pd( (double *) b_p0_pntr++ );   /* load and duplicate */
        b_p1_vreg.v = _mm_loaddup_pd( (double *) b_p1_pntr++ );   /* load and duplicate */
        b_p2_vreg.v = _mm_loaddup_pd( (double *) b_p2_pntr++ );   /* load and duplicate */
        b_p3_vreg.v = _mm_loaddup_pd( (double *) b_p3_pntr++ );   /* load and duplicate */
        
        
        /* First row and second row*/    
        c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
        c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
        c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
        c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

        /* Third row and fourth row*/
        c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
        c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
        c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
        c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;   
    }
    
    // update C(i,j) with value in register
    C( 0, 0 ) += c_00_c_10_vreg.d[0];  C( 0, 1 ) += c_01_c_11_vreg.d[0];  
    C( 0, 2 ) += c_02_c_12_vreg.d[0];  C( 0, 3 ) += c_03_c_13_vreg.d[0]; 

    C( 1, 0 ) += c_00_c_10_vreg.d[1];  C( 1, 1 ) += c_01_c_11_vreg.d[1];  
    C( 1, 2 ) += c_02_c_12_vreg.d[1];  C( 1, 3 ) += c_03_c_13_vreg.d[1]; 

    C( 2, 0 ) += c_20_c_30_vreg.d[0];  C( 2, 1 ) += c_21_c_31_vreg.d[0];  
    C( 2, 2 ) += c_22_c_32_vreg.d[0];  C( 2, 3 ) += c_23_c_33_vreg.d[0]; 

    C( 3, 0 ) += c_20_c_30_vreg.d[1];  C( 3, 1 ) += c_21_c_31_vreg.d[1];  
    C( 3, 2 ) += c_22_c_32_vreg.d[1];  C( 3, 3 ) += c_23_c_33_vreg.d[1]; 
}
