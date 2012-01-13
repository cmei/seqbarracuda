#ifndef BWTALN_H
#define BWTALN_H

#include <stdint.h>
#include "bwt.h"

#define BWA_TYPE_NO_MATCH 0
#define BWA_TYPE_UNIQUE 1
#define BWA_TYPE_REPEAT 2
#define BWA_TYPE_MATESW 3

#define SAM_FPD   1 // paired
#define SAM_FPP   2 // properly paired
#define SAM_FSU   4 // self-unmapped
#define SAM_FMU   8 // mate-unmapped
#define SAM_FSR  16 // self on the reverse strand
#define SAM_FMR  32 // mate on the reverse strand
#define SAM_FR1  64 // this is read one
#define SAM_FR2 128 // this is read two
#define SAM_FSC 256 // secondary alignment

#define BWA_AVG_ERR 0.02
#define MAX_SEQUENCE_LENGTH 150 //cannot go beyond 225 (ptx error)
#define MAX_NO_OF_ALIGNMENTS 40
#define MAX_SCORE 32 //How many different scores to store


#ifndef bns_pac
#define bns_pac(pac, k) ((pac)[(k)>>2] >> ((~(k)&3)<<1) & 3)
#endif

typedef struct {
	bwtint_t w;
	int bid;
} bwt_width_t;

typedef struct {
	// The first 2 bytes is length, length = a[0]<<8|a[1]
	// alphabet is pack using only 4 bits, so 1 char contain 2 alphabet.
	unsigned char character[MAX_SEQUENCE_LENGTH/2];
} bwt_sequence_t;

typedef struct {
	bwtint_t width[MAX_SEQUENCE_LENGTH];
	char bid[MAX_SEQUENCE_LENGTH];
} bwt_sequence_width_t;

typedef struct {
	unsigned char n_mm, n_gapo,n_gape, a;
	bwtint_t k, l;
	int score;
	int best_diff;
	int best_cnt;
} bwt_aln1_t;

typedef struct {
	char *name;
	ubyte_t *seq, *rseq, *qual;
	uint32_t len:20, strand:1, type:2, dummy:1, extra_flag:8;
	uint32_t n_mm:8, n_gapo:8, n_gape:8, mapQ:8;
	int score;
	// alignments in SA coordinates
	int n_aln;
	bwt_aln1_t *aln;
	// alignment information
	bwtint_t sa, pos;
	uint64_t c1:28, c2:28, seQ:8; // number of top1 and top2 hits; single-end mapQ
	int n_cigar;
	uint16_t *cigar;
	// for multi-threading only
	int tid;
	// MD tag
	char *md;
} bwa_seq_t;

#define BWA_MODE_GAPE       0x01 // gap extension
#define BWA_MODE_COMPREAD   0x02 // complemented read
#define BWA_MODE_LOGGAP     0x04
#define BWA_MODE_NONSTOP    0x10


typedef struct {
	int s_mm, s_gapo, s_gape;
	int mode;
	int indel_end_skip, max_del_occ, max_entries;
	float fnr;
	int max_diff, max_gapo, max_gape;
	int max_seed_diff, seed_len;
	int n_threads;
	int max_top2;
	int max_aln;
	int mid;
	int cuda_device;
	int split_kernel; // TODO temporary setting
} gap_opt_t;

#define BWA_PET_STD   1
#define BWA_PET_SOLID 2

typedef struct {
	int max_isize;
	int max_occ;
	int type, is_sw;
} pe_opt_t;

typedef struct {
	unsigned int lim_k;
	unsigned int lim_l;
	unsigned char cur_n_mm, cur_n_gapo,cur_n_gape;
	int cur_state;
	int best_diff;
	int best_cnt;
	int sequence_type;
} init_info_t;

//host-device transit structure
typedef struct
{
	bwt_aln1_t alignment_info[MAX_NO_OF_ALIGNMENTS];
	int no_of_alignments;
	int best_score; //marks best score achieved for particular sequence in the forward run - not updated for backward atm
	unsigned int sequence_id;
	char start_pos;
	// also store current state of alignment
	init_info_t init;
	char finished;
}alignment_store_t;


struct align_store_lst_t
{
   bwt_aln1_t val;
   int sequence_id;
   int start_pos;
   char finished;
   struct align_store_lst_t * next;
};

typedef struct align_store_lst_t align_store_lst;

//Temporary host structure to hold alignments
typedef struct
{

	align_store_lst * score_align[MAX_SCORE]; // holds a pointer for each of the scores
	int start_pos; // shared for all sequences - we do sequential runs
	// also store current state of alignment
}main_alignment_store_host_t;





typedef struct
{
	unsigned char string[MAX_SEQUENCE_LENGTH];
}string_t;

struct __bwa_seqio_t;
typedef struct __bwa_seqio_t bwa_seqio_t;


#ifdef __cplusplus
extern "C" {
#endif

	gap_opt_t *gap_init_opt();
	void barracuda_bwa_aln_core(const char *prefix, const char *fn_fa, const gap_opt_t *opt);

	bwa_seqio_t *bwa_seq_open(const char *fn);
	void bwa_seq_close(bwa_seqio_t *bs);
	void seq_reverse(int len, ubyte_t *seq, int is_comp);
	bwa_seq_t *bwa_read_seq(bwa_seqio_t *seq, unsigned int n_needed, unsigned int *n, int is_comp, int mid);
	int bwa_read_seq_one_half_byte (bwa_seqio_t *bs, unsigned char * half_byte_array, unsigned int start_index, unsigned short * length, int mid);
	int bwa_read_seq_one (bwa_seqio_t *bs, unsigned char * byte_array, unsigned short * length);
	void bwa_free_read_seq(int n_seqs, bwa_seq_t *seqs);
	int bwa_aln(int argc, char *argv[]);
	void bwa_deviceQuery();
	int bwa_cal_maxdiff(int l, double err, double thres);
	void bwa_cs2nt_core(bwa_seq_t *p, bwtint_t l_pac, ubyte_t *pac);
	int detect_cuda_device();

	///////////////////////////////////////////////////////////////
	// Begin SAMSE CUDA core
	///////////////////////////////////////////////////////////////

    void launch_bwa_cal_pac_pos_cuda(
	    const char *prefix,
	    int n_seqs,
	    bwa_seq_t *seqs,
	    int max_mm,
	    float fnr,
	    int device);

    int prepare_bwa_cal_pac_pos_cuda(
        const char *prefix,
        const int *g_log_n_ho,
        const int g_log_n_len,
        const int n_seqs_max,
        int device);

    void free_bwa_cal_pac_pos_cuda();

    ///////////////////////////////////////////////////////////////
    // End SAMSE CUDA core
    ///////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif


///////////////////////////////////////////////////////////////
// Begin SAMSE CUDA core
///////////////////////////////////////////////////////////////

void report_cuda_error_CPU(const char * message);
void report_cuda_error_GPU(const char * message);
void copy_g_log_n_cuda(const int *g_log_n_ho,const int g_log_n_len);
//void copy_bwt_rbwt_sa_cuda(const bwt_t *bwt, const bwt_t *rbwt);
void prepare_bwa_cal_pac_pos_seqs_cuda(int n_seqs_max);
void free_bwa_cal_pac_pos_bwt_rbwt_cuda();
void free_bwa_cal_pac_pos_seqs_cuda();
void compare_mapQ_pos(bwa_seq_t *seqs, int n_seqs);
void calc_n_block3(
    int *n_sp_to_use,
    int *n_block,
    int *n_seq_per_block,
    int *block_mod,
    const int n_mp_on_device,
    const int n_sp_per_mp,
    const int n_seqs);


///////////////////////////////////////////////////////////////
// End SAMSE CUDA core
///////////////////////////////////////////////////////////////
#endif
