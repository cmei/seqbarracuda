#ifndef BWTALN_CUH
#define BWTALN_CUH

#include "bwt.h"
#include "bwtaln.h"


///////////////////////////////////////////////////////////////
// Begin struct (Dag's test)
///////////////////////////////////////////////////////////////


// This struct is for use in CUDA implementation of the functions 
// bwa_cal_max_diff() and bwa_approx_mapQ(). In the variable
// "strand_type", bits 1-2 are "type", bit 3 is "strand" in 
// corresponding to CPU struct "bwa_seq_t".  
// 

typedef struct __align__(16)
{
	uint32_t len;
	uint8_t strand;
	uint8_t type;
	uint8_t n_mm;
	uint32_t c1;
	uint32_t c2;
} bwa_maxdiff_mapQ_t;




///////////////////////////////////////////////////////////////
// End struct (Dag's test)
///////////////////////////////////////////////////////////////


__device__ 
void bwt_rbwt_cuda_2occ(
    unsigned int k, 
    unsigned int l, 
    unsigned char c, 
    unsigned int *ok, 
    unsigned int *ol, 
    unsigned short bwt_type);

///////////////////////////////////////////////////////////////
// Begin bwa_cal_pac_pos_cuda (Dag's test)
///////////////////////////////////////////////////////////////

void print_cuda_info();


__device__ inline uint4 BWT_RBWT_OCC4b(uint32_t coord_1D,char strand);

__device__ ubyte_t bwt_rbwt_B0(bwtint_t k,char strand);


__device__ uint32_t bwt_rbwt_bwt(bwtint_t k,char strand);

__device__ inline void Conv1DTo2D1(int *coord12D,int *coord22D,int coord1D);

__device__ inline void Conv1DTo2D4(int *coord12D,int *coord22D,int coord1D);

__device__ int cuda_bwa_approx_mapQ(const bwa_maxdiff_mapQ_t *p, int mm);

__device__ 
void update_indices(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty);

__device__ 
void update_indices_in_parallel(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty);

__device__ 
void fetch_read_new_in_parallel(
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf,
    int16_t *sa_origin,
    const bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de,
    const int offset,
    int *n_sa_in_buf,
    int *n_sa_buf_empty,
    int *n_sa_processed,
    int *n_sa_remaining,
    int *sa_next_no,
    const int n_sa_total);

__device__ 
void sort_reads(
    bwtint_t *sa_buf_arr,
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf_arr,
    int16_t *sa_origin,
    bwtint_t *sa_return,
    const int *n_sa_in_buf,
    int *n_sa_in_buf_prev);

__global__ 
void copy_stuff (bwtint_t *B0_de, bwtint_t len);

__global__
void cuda_bwa_cal_pac_pos_parallel1(
    uint8_t *seqs_mapQ_de,
    bwtint_t *seqs_pos_de,
    const bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de,
    const bwtint_t *seqs_sa_de,
    int n_seqs,
    int n_block,
    int n_seq_per_block,
    int block_mod,
    int max_mm,
    float fnr,
    int bwt_sa_intv,
    int rbwt_sa_intv);

void calc_n_block1(
    int *n_sp_to_use, 
    int *n_block, 
    int *n_seq_per_block, 
    int *block_mod, 
    int n_mp_on_device, 
    int n_sp_per_mp, 
    int n_seqs);


///////////////////////////////////////////////////////////////
// End bwa_cal_pac_pos_cuda (Dag's test)
///////////////////////////////////////////////////////////////



#endif // BWTALN_CUH
