/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwtaln.cu  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2011, University of Cambridge Metabolic Research Labs.
   Contributers: Petr Klus, Dag Lyberg, Simon Lam and Brian Lam

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 3
   of the License, or (at your option) any later version.

   This program is distribut ed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   This program is based on a modified version of BWA 0.4.9

*/

#define PACKAGE_VERSION "0.6.1 beta"
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdint.h>
#include "bwtaln.h"
#include "bwtgap.h"
#include "utils.h"
#include "bwtaln.cuh"

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

// Performance switches
#define BWT_2_OCC_ENABLE 0 // enable looking up of k and l in the same time for counting character occurrence (slower, so disable by default)
#define BWT_TABLE_LOOKUP_ENABLE 1 // use lookup table when instead of counting character occurrence, (faster so enable by default)

//The followings are settings for memory allocations and memory requirements
#define MIN_MEM_REQUIREMENT 768 // minimal global memory requirement in (MiB).  Currently at 768MB
#define CUDA_TESLA 1350 // enlarged workspace buffer. Currently at 1350MB will be halved if not enough mem available

#define SEQUENCE_TABLE_SIZE_EXPONENTIAL 23// DO NOT CHANGE! buffer size in (2^)units for sequences and alignment storages (batch size)
// Maximum exponential is up to 30 [~ 1  GBytes] for non-debug, non alignment
// Maximum exponential is up to 26 [~ 128MBytes] for debug
// Maximum exponential is up to 23 for alignment with 4GB RAM(default : 23)

//The followings are for DEBUG only
#define OUTPUT_ALIGNMENTS 1 // should leave ON for outputting alignment
#define STDOUT_STRING_RESULT 0 // output alignment in text format (in SA coordinates, not compatible with SAM output modules(samse/pe)
#define STDOUT_BINARY_RESULT 1 //output alignment for samse/sampe (leave ON)
#define MYBINARY 0 // for debugging purposes, outputs the alignments to a separate file called mybinary.sai
#define CUDA_SAMSE 1 //Enable CUDA SAMSE code, debug only (leave ON)

// how much debugging information shall the kernel output? kernel output only works for fermi and above
#define DEBUG_LEVEL 0


// how long should a subsequence be for one kernel launch
// For multikernel design
#define PASS_LENGTH 32  // pass size, also initial seed size
#define SPLIT_ENGAGE PASS_LENGTH + 6 //when splitting starts to happen
#define MAX_SEED_LENGTH 50 // not tested beyond 50

//Global variables for inexact match <<do not change>>
#define STATE_M 0
#define STATE_I 1
#define STATE_D 2

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//CUDA global variables
__device__ __constant__ bwt_t bwt_cuda;
__device__ __constant__ bwt_t rbwt_cuda;
__device__ __constant__ gap_opt_t options_cuda;

//Texture Maps
// uint4 is used because the maximum width for CUDA texture bind of 1D memory is 2^27,
// and uint4 the structure 4xinteger is x,y,z,w coordinates and is 16 bytes long,
// therefore effectively there are 2^27x16bytes memory can be access = 2GBytes memory.
texture<uint4, 1, cudaReadModeElementType> bwt_occ_array;
texture<uint4, 1, cudaReadModeElementType> rbwt_occ_array;
texture<unsigned int, 1, cudaReadModeElementType> sequences_array;
texture<uint2, 1, cudaReadModeElementType> sequences_index_array;

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

unsigned long long copy_bwts_to_cuda_memory( const char * prefix, unsigned int ** bwt,  unsigned int ** rbwt, int mem_available, bwtint_t* forward_seq_len, bwtint_t* backward_seq_len)
// bwt occurrence array to global and bind to texture, bwt structure to constant memory
{
	bwt_t * bwt_src;
	char str[100];
	unsigned long long size_read = 0;

#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] mem left: %d\n", mem_available);
#endif

//	if ( bwt != 0 )
	{
		//Original BWT
		//Load bwt occurrence array from from disk
		strcpy(str, prefix); strcat(str, ".bwt");  bwt_src = bwt_restore_bwt(str);
		size_read += bwt_src->bwt_size*sizeof(uint32_t);

		mem_available = mem_available - int (size_read>>20);

		*forward_seq_len = bwt_src->seq_len;
		if(mem_available > 0)
		{
			//Allocate memory for bwt
			cudaMalloc((void**)bwt, bwt_src->bwt_size*sizeof(uint32_t));
			//copy bwt occurrence array from host to device and dump the bwt to save CPU memory
			cudaMemcpy (*bwt, bwt_src->bwt, bwt_src->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
			//bind global variable bwt to texture memory bwt_occ_array
			cudaBindTexture(0, bwt_occ_array, *bwt, bwt_src->bwt_size*sizeof(uint32_t));
			//copy bwt structure data to constant memory bwt_cuda structure
			cudaMemcpyToSymbol ( bwt_cuda, bwt_src, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
			bwt_destroy(bwt_src);
		}
		else
		{
			fprintf(stderr,"[aln_core] Not enough device memory to perform alignment.\n");
			return 0;
		}


#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] bwt loaded, mem left: %d\n", mem_available);
#endif
	}
//	if ( rbwt != 0 )
	{
		//Reversed BWT
		//Load bwt occurrence array from from disk
		strcpy(str, prefix); strcat(str, ".rbwt");  bwt_src = bwt_restore_bwt(str);
		size_read += bwt_src->bwt_size*sizeof(uint32_t);
		mem_available = mem_available - (bwt_src->bwt_size*sizeof(uint32_t)>>20);


#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] rbwt loaded mem left: %d\n", mem_available);
#endif

		if (mem_available > 0)
		{
			*backward_seq_len = bwt_src->seq_len;

			//Allocate memory for bwt
			cudaMalloc((void**)rbwt, bwt_src->bwt_size*sizeof(uint32_t));
			//copy reverse bwt occurrence array from host to device and dump the bwt to save CPU memory
			cudaMemcpy (*rbwt, bwt_src->bwt, bwt_src->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
			//bind global variable bwt to texture memory bwt_occ_array
			cudaBindTexture(0, rbwt_occ_array, *rbwt, bwt_src->bwt_size*sizeof(uint32_t));
			//copy bwt structure data to constant memory bwt_cuda structure
			cudaMemcpyToSymbol ( rbwt_cuda, bwt_src, sizeof(bwt_t), 0, cudaMemcpyHostToDevice);
			bwt_destroy(bwt_src);
		}
		else
		{
			fprintf(stderr,"[aln_core] Not enough device memory to perform alignment.\n");
			return 0;
		}

	}

	return size_read;
}

void free_bwts_from_cuda_memory( unsigned int * bwt , unsigned int * rbwt )
{
	if ( bwt != 0 )
	{
		cudaUnbindTexture(bwt_occ_array);
		cudaFree(bwt);
	}
	if ( rbwt != 0 )
	{
		cudaUnbindTexture(rbwt_occ_array);
		cudaFree(rbwt);
	}
}

#define write_to_half_byte_array(array,index,data) \
	(array)[(index)>>1]=(unsigned char)(((index)&0x1)?(((array)[(index)>>1]&0xF0)|((data)&0x0F)):(((data)<<4)|((array)[(index)>>1]&0x0F)))

int copy_sequences_to_cuda_memory ( bwa_seqio_t *bs, uint2 * global_sequences_index, uint2 * main_sequences_index, unsigned char * global_sequences, unsigned char * main_sequences, unsigned int * read_size, unsigned short & max_length, int mid, int buffer)
{
	//sum of length of sequences up the the moment
	unsigned int accumulated_length = 0;
	//sequence's read length
	unsigned short read_length = 0;
	unsigned int number_of_sequences = 0;

	while (bwa_read_seq_one_half_byte(bs,main_sequences,accumulated_length,&read_length, mid)>0)
	{
		main_sequences_index[number_of_sequences].x = accumulated_length;
		main_sequences_index[number_of_sequences].y = read_length;
		if (read_length > max_length) max_length = read_length;

		accumulated_length += read_length;
		number_of_sequences++;

		if ( accumulated_length + MAX_SEQUENCE_LENGTH > (1ul<<(buffer+1)) ) break;
	}
	//copy main_sequences_width from host to device
	cudaUnbindTexture(sequences_index_array);
    cudaMemcpy(global_sequences_index, main_sequences_index, (number_of_sequences)*sizeof(uint2), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_index_array, global_sequences_index, (number_of_sequences)*sizeof(uint2));

    //copy main_sequences from host to device, sequences array length should be accumulated_length/2
    cudaUnbindTexture(sequences_array);
    cudaMemcpy(global_sequences, main_sequences, (1ul<<(buffer))*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_array, global_sequences, (1ul<<(buffer))*sizeof(unsigned char));

    if ( read_size ) *read_size = accumulated_length;

    return number_of_sequences;
}

//CUDA DEVICE CODE STARTING FROM THIS LINE
/////////////////////////////////////////////////////////////////////////////


__device__ unsigned char read_char(unsigned int pos, unsigned int * lastpos, unsigned int * data )
// read character back from sequence arrays
// which is packed as half bytes and stored as in a unsigned int array
{
	unsigned char c;
	unsigned int pos_shifted = pos >> 3;
	unsigned int tmp = *data;
	if (*lastpos!=pos_shifted)
	{
		*data = tmp = tex1Dfetch(sequences_array, pos_shifted);
		*lastpos=pos_shifted;
	}

	switch (pos&0x7)
	{
	case 7:
		c = tmp>>24;
		break;
	case 6:
		c = tmp>>28;
		break;
	case 5:
		c = tmp>>16;
		break;
	case 4:
		c = tmp>>20;
		break;
	case 3:
		c = tmp>>8;
		break;
	case 2:
		c = tmp>>12;
		break;
	case 1:
		c = tmp;
		break;
	case 0:
		c = tmp>>4;
		break;
	}

	return c&0xF;
}

__device__ inline unsigned int numbits(unsigned int i, unsigned char c)
// with y of 32 bits which is a string sequence encoded with 2 bits per alphabet,
// count the number of occurrence of c ( one pattern of 2 bits alphabet ) in y
{
	i = ((c&2)?i:~i)>>1&((c&1)?i:~i)&0x55555555;
	i = (i&0x33333333)+(i>>2&0x33333333);
	return((i+(i>>4)&0x0F0F0F0F)*0x01010101)>>24;
}

#define __occ_cuda_aux4(b) (bwt_cuda.cnt_table[(b)&0xff]+bwt_cuda.cnt_table[(b)>>8&0xff]+bwt_cuda.cnt_table[(b)>>16&0xff]+bwt_cuda.cnt_table[(b)>>24])
#define BWTOCC(a) (tex1Dfetch(bwt_occ_array,a))
#define RBWTOCC(a) (tex1Dfetch(rbwt_occ_array,a))

__device__ uint4 bwt_cuda_occ4(bwtint_t k)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	// total number of character c in the up to the interval of k
	uint4 tmp;
	uint4 n = {0,0,0,0};
	unsigned int i = 0;
	unsigned int m = 0;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// uint4 is used because CUDA 1D texture array is limited to width for 2^27
	// to access 2GB of memory, a structure uint4 is needed
	// tmp variable
	unsigned int tmp1,tmp2;//, tmp3;

	if (k == bwt_cuda.seq_len)
	{
		n.x = bwt_cuda.L2[1]-bwt_cuda.L2[0];
		n.y = bwt_cuda.L2[2]-bwt_cuda.L2[1];
		n.z = bwt_cuda.L2[3]-bwt_cuda.L2[2];
		n.w = bwt_cuda.L2[4]-bwt_cuda.L2[3];
		return n;
	}
	if (k == (bwtint_t)(-1)) return n;
	if (k >= bwt_cuda.primary) --k; // because $ is not in bwt

	//tmp3 = k>>7;
	//i = tmp3*3;
	i = ((k>>7)*3);

	// count the number of character c within the 128bits interval
	tmp = BWTOCC(i+1);
	if (k&0x40)
	{
		m = __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		m += __occ_cuda_aux4(tmp.z);
		m += __occ_cuda_aux4(tmp.w);
		tmp = BWTOCC(i+2);
	}
	if (k&0x20)
	{
		m += __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		tmp1=tmp.z;
		tmp2=tmp.w;
	} else {
		tmp1=tmp.x;
		tmp2=tmp.y;
	}
	if (k&0x10)
	{
		m += __occ_cuda_aux4(tmp1);
		tmp1=tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	m += __occ_cuda_aux4(tmp1>>(((~k)&15)<<1));
	n.x = m&0xff; n.y = m>>8&0xff; n.z = m>>16&0xff; n.w = m>>24;

	// retrieve the total count from index the number of character C in the up k/128bits interval
	tmp = BWTOCC(i);
	n.x += tmp.x; n.x -= ~k&15; n.y += tmp.y; n.z += tmp.z; n.w += tmp.w;

	return n;
}

__device__ bwtint_t bwt_cuda_occ(bwtint_t k, ubyte_t c)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	#if BWT_TABLE_LOOKUP_ENABLE == 1
	uint4 ok = bwt_cuda_occ4(k);
	switch ( c )
	{
	case 0:
		return ok.x;
	case 1:
		return ok.y;
	case 2:
		return ok.z;
	case 3:
		return ok.w;
	}
	return 0;
	#else  // USE_LOOKUP_TABLE == 1
	// total number of character c in the up to the interval of k
	unsigned int n = 0;
	unsigned int i = 0;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// uint4 is used because CUDA 1D texture array is limited to width for 2^27
	// to access 2GB of memory, a structure uint4 is needed
	uint4 tmp;
	// tmp variable
	unsigned int tmp1,tmp2;

	if (k == bwt_cuda.seq_len) return bwt_cuda.L2[c+1] - bwt_cuda.L2[c];
	if (k == (bwtint_t)(-1)) return 0;
	if (k >= bwt_cuda.primary) --k; // because $ is not in bwt

	i = ((k>>7)*3);
	// count the number of character c within the 128bits interval
	tmp = BWTOCC(i+1);
	if (k&0x40)
	{
		n += numbits(tmp.x, c);
		n += numbits(tmp.y, c);
		n += numbits(tmp.z, c);
		n += numbits(tmp.w, c);
		tmp = BWTOCC(i+2);
	}
	if (k&0x20)
	{
		n += numbits(tmp.x, c);
		n += numbits(tmp.y, c);
		tmp1=tmp.z;
		tmp2=tmp.w;
	} else {
		tmp1=tmp.x;
		tmp2=tmp.y;
	}
	if (k&0x10)
	{
		n += numbits(tmp1, c);
		tmp1=tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	n += numbits(tmp1>>(((~k)&15)<<1), c);

	// retrieve the total count from index the number of character C in the up k/128bits interval
	tmp = BWTOCC(i);
	switch ( c )
	{
	case 0:
		n += tmp.x;
		// corrected for the masked bits
		n -= ~k&15;
		break;
	case 1:
		n += tmp.y;
		break;
	case 2:
		n += tmp.z;
		break;
	case 3:
		n += tmp.w;
		break;
	}
	return n;
	#endif // USE_LOOKUP_TABLE == 1
}

__device__ uint4 rbwt_cuda_occ4(bwtint_t k)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	// total number of character c in the up to the interval of k
	uint4 tmp;
	uint4 n = {0,0,0,0};
	unsigned int i = 0;
	unsigned int m = 0;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// uint4 is used because CUDA 1D texture array is limited to width for 2^27
	// to access 2GB of memory, a structure uint4 is needed
	// tmp variable
	unsigned int tmp1,tmp2; //, tmp3;

	if (k == rbwt_cuda.seq_len)
	{
		n.x = rbwt_cuda.L2[1]-rbwt_cuda.L2[0];
		n.y = rbwt_cuda.L2[2]-rbwt_cuda.L2[1];
		n.z = rbwt_cuda.L2[3]-rbwt_cuda.L2[2];
		n.w = rbwt_cuda.L2[4]-rbwt_cuda.L2[3];
		return n;
	}
	if (k == (bwtint_t)(-1)) return n;
	if (k >= rbwt_cuda.primary) --k; // because $ is not in bwt

	i = ((k>>7)*3);
//	tmp3 = k>> 7;
//	i = tmp3+tmp3+tmp3;
	// count the number of character c within the 128bits interval
	tmp = RBWTOCC(i+1);
	if (k&0x40)
	{
		m = __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		m += __occ_cuda_aux4(tmp.z);
		m += __occ_cuda_aux4(tmp.w);
		tmp = RBWTOCC(i+2);
	}
	if (k&0x20)
	{
		m += __occ_cuda_aux4(tmp.x);
		m += __occ_cuda_aux4(tmp.y);
		tmp1=tmp.z;
		tmp2=tmp.w;
	} else {
		tmp1=tmp.x;
		tmp2=tmp.y;
	}
	if (k&0x10)
	{
		m += __occ_cuda_aux4(tmp1);
		tmp1=tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	m += __occ_cuda_aux4(tmp1>>(((~k)&15)<<1));
	n.x = m&0xff; n.y = m>>8&0xff; n.z = m>>16&0xff; n.w = m>>24;

	// retrieve the total count from index the number of character C in the up k/128bits interval
	tmp = RBWTOCC(i);
	n.x += tmp.x; n.x -= ~k&15; n.y += tmp.y; n.z += tmp.z; n.w += tmp.w;

	return n;
}

__device__ inline bwtint_t rbwt_cuda_occ(bwtint_t k, ubyte_t c )
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
{
	#if BWT_TABLE_LOOKUP_ENABLE == 1
	uint4 ok = rbwt_cuda_occ4(k);
	switch ( c )
	{
	case 0:
		return ok.x;
	case 1:
		return ok.y;
	case 2:
		return ok.z;
	case 3:
		return ok.w;
	}
	return 0;
	#else  // USE_LOOKUP_TABLE == 1
	// total number of character c in the up to the interval of k
	unsigned int n = 0;
	unsigned int i = 0;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// uint4 is used because CUDA 1D texture array is limited to width for 2^27
	// to access 2GB of memory, a structure uint4 is needed
	uint4 tmp;
	// tmp variable
	unsigned int tmp1, tmp2;

	if (k == bwt_cuda.seq_len)
		return rbwt_cuda.L2[c + 1] - rbwt_cuda.L2[c];
	if (k == (bwtint_t) (-1))
		return 0;
	if (k >= rbwt_cuda.primary)
		--k; // because $ is not in bwt

	i = ((k >> 7) * 3);
	// count the number of character c within the 128bits interval
	tmp = RBWTOCC(i+1);
	if (k & 0x40) {
		n += numbits(tmp.x, c);
		n += numbits(tmp.y, c);
		n += numbits(tmp.z, c);
		n += numbits(tmp.w, c);
		tmp = RBWTOCC(i+2);
	}
	if (k & 0x20) {
		n += numbits(tmp.x, c);
		n += numbits(tmp.y, c);
		tmp1 = tmp.z;
		tmp2 = tmp.w;
	} else {
		tmp1 = tmp.x;
		tmp2 = tmp.y;
	}
	if (k & 0x10) {
		n += numbits(tmp1, c);
		tmp1 = tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	n += numbits(tmp1 >> (((~k) & 15) << 1), c);

	// retrieve the total count from index the number of character C in the up k/128bits interval
	tmp = RBWTOCC(i);
	switch (c) {
	case 0:
		n += tmp.x;
		// corrected for the masked bits
		n -= ~k & 15;
		break;
	case 1:
		n += tmp.y;
		break;
	case 2:
		n += tmp.z;
		break;
	case 3:
		n += tmp.w;
		break;
	}
	return n;
	#endif // USE_LOOKUP_TABLE == 1
}

#if BWT_2_OCC_ENABLE == 1
__device__ void bwts_cuda_2occ(unsigned int k, unsigned int l, unsigned char c, unsigned int *ok, unsigned int *ol, unsigned short bwt_type)
{
	unsigned int _k, _l;
	unsigned int i;
	// tmp variable
	uint4 c_count;
	uint4 dat0, dat1;
	// total number of character c in the up to the interval of k
	unsigned int n, m, count;
	// remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
	// uint4 is used because CUDA 1D texture array is limited to width for 2^27
	// to access 2GB of memory, a structure uint4 is needed
	uint4 tmpn,tmpm;
	// tmp variable
	unsigned int tmp1,tmp2;

	if ( bwt_type == 0 )
	{
		if (k == l) {
			*ok = *ol = bwt_cuda_occ(k, c);
			return;
		}
		_k = (k >= bwt_cuda.primary)? k-1 : k;
		_l = (l >= bwt_cuda.primary)? l-1 : l;
		if	 (	_l>>7 != _k>>7 ||
				k == (bwtint_t)(-1) ||
				l == (bwtint_t)(-1)
			)
		{
			*ok = bwt_cuda_occ(k, c);
			*ol = bwt_cuda_occ(l, c);
			return;
		}
		k = _k;
		l = _l;
		i = ((k>>7)*3);
		// count the number of character c within the 128bits interval
		c_count = BWTOCC(i);
		dat0 = BWTOCC(i+1);
		if ( l & 0x40 )	dat1 = BWTOCC(i+2);
	}
	else
	{
		if (k == l) {
			*ok = *ol = rbwt_cuda_occ(k, c);
			return;
		}
		_k = (k >= rbwt_cuda.primary)? k-1 : k;
		_l = (l >= rbwt_cuda.primary)? l-1 : l;
		if	 (	_l>>7 != _k>>7 ||
				k == (bwtint_t)(-1) ||
				l == (bwtint_t)(-1)
			)
		{
			*ok = rbwt_cuda_occ(k, c);
			*ol = rbwt_cuda_occ(l, c);
			return;
		}
		k = _k;
		l = _l;
		i = ((k>>7)*3);
		// count the number of character c within the 128bits interval
		c_count = RBWTOCC(i);
		dat0 = RBWTOCC(i+1);
		if (l&0x40)	dat1 = RBWTOCC(i+2);
	}

	n = m = 0;
	tmpn = dat0;
	tmpm = dat0;
	if (l&0x40)
	{
		count = numbits(tmpn.x, c);
		count += numbits(tmpn.y, c);
		count += numbits(tmpn.z, c);
		count += numbits(tmpn.w, c);
		m = count;
		tmpm = dat1;
		if (k&0x40)
		{
			n = count;
			tmpn = dat1;
		}
	}
	if (k&0x20)
	{
		n += numbits(tmpn.x, c);
		n += numbits(tmpn.y, c);
		tmp1=tmpn.z;
		tmp2=tmpn.w;
	} else {
		tmp1=tmpn.x;
		tmp2=tmpn.y;
	}
	if (k&0x10)
	{
		n += numbits(tmp1, c);
		tmp1=tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	n += numbits(tmp1>>(((~k)&15)<<1), c);

	if (l&0x20)
	{
		m += numbits(tmpm.x, c);
		m += numbits(tmpm.y, c);
		tmp1=tmpm.z;
		tmp2=tmpm.w;
	} else {
		tmp1=tmpm.x;
		tmp2=tmpm.y;
	}
	if (l&0x10)
	{
		m += numbits(tmp1, c);
		tmp1=tmp2;
	}
	// just shift away the unwanted character, no need to shift back
	// number of c in tmp1 will still be correct
	m += numbits(tmp1>>(((~l)&15)<<1), c);

	// retrieve the total count from index the number of character C in the up k/128bits interval
	switch ( c )
	{
	case 0:
		n += c_count.x;
		m += c_count.x;
		// corrected for the masked bits
		n -= ~k&15;
		m -= ~l&15;
		break;
	case 1:
		n += c_count.y;
		m += c_count.y;
		break;
	case 2:
		n += c_count.z;
		m += c_count.z;
		break;
	case 3:
		n += c_count.w;
		m += c_count.w;
		break;
	}
	*ok=n;
	*ol=m;
}
#endif // BWT_2_OCC_ENABLE == 1

__device__ void bwt_cuda_device_calculate_width (unsigned char* sequence, unsigned short sequence_type, unsigned int * widths, unsigned char * bids, unsigned short length)
//Calculate bids and widths for worst case bound, returns widths[senquence length] and bids[sequence length]
{
	unsigned short bid;
	//suffix array interval k(lower bound) and l(upper bound)
	unsigned int k, l;
	unsigned int i;


	// do calculation and update w and bid
	bid = 0;
	k = 0;
	l = bwt_cuda.seq_len;
	unsigned short bwt_type = sequence_type;

	for (i = 0; i < length; ++i) {
		unsigned char c = sequence[i];
		if (c < 4) {
			#if BWT_2_OCC_ENABLE == 1
			//unsigned int ok, ol;
			//bwts_cuda_2occ(k - 1, l, c, &ok, &ol, bwt_type);
			uint4 ok, ol;
			unsigned int ok1, ol1;
			bwts_cuda_2occ4(k - 1, l, &ok, &ol, bwt_type);
			switch ( c )
			{
			case 0:
				ok1 = ok.x;
				ol1 = ol.x;
				break;
			case 1:
				ok1 = ok.y;
				ol1 = ol.y;
				break;
			case 2:
				ok1 = ok.z;
				ol1 = ol.z;
				break;
			case 3:
				ok1 = ok.w;
				ol1 = ol.w;
				break;
			}
			k = bwt_cuda.L2[c] + ok1 + 1;
			l = bwt_cuda.L2[c] + ol1;
			#else  // BWT_2_OCC_ENABLE == 1
			if ( bwt_type == 0 )
			{
				k = bwt_cuda.L2[c] + bwt_cuda_occ(k - 1, c) + 1;
				l = bwt_cuda.L2[c] + bwt_cuda_occ(l, c);
			}
			else
			{
				k = rbwt_cuda.L2[c] + rbwt_cuda_occ(k - 1, c) + 1;
				l = rbwt_cuda.L2[c] + rbwt_cuda_occ(l, c);
			}
			#endif // BWT_2_OCC_ENABLE == 1
		}
		if (k > l || c > 3) {
			k = 0;
			l = bwt_cuda.seq_len;
			++bid;
		}
		widths[i] = l - k + 1;
		bids[i] = bid;
	}
	widths[length] = k + 1;
	bids[length] = bid;
	return;
}

__device__ void bwt_cuda_device_calculate_width_limit (init_info_t* init_info,unsigned char* sequence, unsigned short sequence_type, unsigned int * widths, unsigned char * bids, unsigned short length)
//Calculate bids and widths for worst case bound, returns widths[senquence length] and bids[sequence length]
{
	unsigned short bid;
	//suffix array interval k(lower bound) and l(upper bound)
	unsigned int k, l;
	unsigned int i;


	// do calculation and update w and bid
	bid = 0;
	//k = init_info->lim_k;
	//l = init_info->lim_l;
	k = 0;
	l = bwt_cuda.seq_len;
	unsigned short bwt_type = sequence_type;

	for (i = 0; i < length; ++i) {
		unsigned char c = sequence[i];
		if (c < 4) {
			#if BWT_2_OCC_ENABLE == 1
			//unsigned int ok, ol;
			//bwts_cuda_2occ(k - 1, l, c, &ok, &ol, bwt_type);
			uint4 ok, ol;
			unsigned int ok1, ol1;
			bwts_cuda_2occ4(k - 1, l, &ok, &ol, bwt_type);
			switch ( c )
			{
			case 0:
				ok1 = ok.x;
				ol1 = ol.x;
				break;
			case 1:
				ok1 = ok.y;
				ol1 = ol.y;
				break;
			case 2:
				ok1 = ok.z;
				ol1 = ol.z;
				break;
			case 3:
				ok1 = ok.w;
				ol1 = ol.w;
				break;
			}
			k = bwt_cuda.L2[c] + ok1 + 1;
			l = bwt_cuda.L2[c] + ol1;
			#else  // BWT_2_OCC_ENABLE == 1
			if ( bwt_type == 0 )
			{
				k = bwt_cuda.L2[c] + bwt_cuda_occ(k - 1, c) + 1;
				l = bwt_cuda.L2[c] + bwt_cuda_occ(l, c);
			}
			else
			{
				k = rbwt_cuda.L2[c] + rbwt_cuda_occ(k - 1, c) + 1;
				l = rbwt_cuda.L2[c] + rbwt_cuda_occ(l, c);
			}
			#endif // BWT_2_OCC_ENABLE == 1
		}
		if (k > l || c > 3) {
			k = 0;
			l = bwt_cuda.seq_len;
			++bid;
		}
		widths[i] = l - k + 1;
		bids[i] = bid;
	}
	widths[length] = k + 1;
	bids[length] = bid;
	return;
}

__device__ int bwa_cuda_cal_maxdiff(int l, double err, double thres)
{
	double elambda = exp(-l * err);
	double sum, y = 1.0;
	int k, x = 1;
	for (k = 1, sum = elambda; k < 1000; ++k) {
		y *= l * err;
		x *= k;
		sum += elambda * y / x;
		if (1.0 - sum < thres) return k;
	}
	return 2;
}

__device__ void gap_stack_shadow_cuda(int x, int len, bwtint_t max, int last_diff_pos, unsigned int * width, unsigned char * bid)
{
	int i, j;
	for (i = j = 0; i < last_diff_pos; ++i)
	{
		if (width[i] > x)
		{
			width[i] -= x;
		}
		else if (width[i] == x)
		{
			bid[i] = 1;
			width[i] = max - (++j);
		} // else should not happen
	}
}

__device__ unsigned int int_log2_cuda(uint32_t v)
//integer log
{
	unsigned int c = 0;
	if (v & 0xffff0000u) { v >>= 16; c |= 16; }
	if (v & 0xff00) { v >>= 8; c |= 8; }
	if (v & 0xf0) { v >>= 4; c |= 4; }
	if (v & 0xc) { v >>= 2; c |= 2; }
	if (v & 0x2) c |= 1;
	return c;
}

__device__ int bwt_cuda_match_exact(unsigned short bwt_type, unsigned int length, const unsigned char * str, bwtint_t *k0, bwtint_t *l0)
//exact match algorithm
{
	int i;
	unsigned int k, l;
	k = *k0; l = *l0;
	for (i = length - 1; i >= 0; --i)
	{
		unsigned char c = str[i];
		//if (c > 3) return 0; // there is an N here. no match

		#if BWT_2_OCC_ENABLE == 1
		unsigned int ok, ol;
		bwts_cuda_2occ(k - 1, l, c, &ok, &ol, bwt_type);
		if ( bwt_type == 0 )
		{
			k = bwt_cuda.L2[c] + ok + 1;
			l = bwt_cuda.L2[c] + ol;
		}
		else
		{
			k = rbwt_cuda.L2[c] + ok + 1;
			l = rbwt_cuda.L2[c] + ol;
		}
		#else  // BWT_2_OCC_ENABLE == 1
		if ( bwt_type == 0 )
		{
			k = bwt_cuda.L2[c] + bwt_cuda_occ(k - 1, c) + 1;
			l = bwt_cuda.L2[c] + bwt_cuda_occ(l, c);
		}
		else
		{
			k = rbwt_cuda.L2[c] + rbwt_cuda_occ(k - 1, c) + 1;
			l = rbwt_cuda.L2[c] + rbwt_cuda_occ(l, c);
		}
		#endif

		*k0 = k;
		*l0 = l;

		// no match
		if (k > l)
			return 0;
	}
	*k0 = k;
	*l0 = l;

	return l - k + 1;
}


//////////////////////////////////////////////////////////////////////////////////////////
//DFS MATCH
//////////////////////////////////////////////////////////////////////////////////////////

__device__ void cuda_dfs_initialize(uint4 *stack, uchar4 *stack_mm, char4 *pushes/*, int * scores*/)
//initialize memory store for dfs_match
{
	int i;
	uint4 temp1 = {0,0,0,0};
	uchar4 temp2 = {0,0,0,0};
	char4 temp3 = {0,0,0,0};;

	// zero out the whole stack
	for (i = 0; i < MAX_SEQUENCE_LENGTH; i++)
	{
		stack[i] = temp1;
		stack_mm[i] = temp2;
		pushes[i] = temp3;
	}
	return;
}

__device__ void cuda_dfs_push(uint4 *stack, uchar4 *stack_mm, char4 *pushes, int i, unsigned int k, unsigned int l, int n_mm, int n_gapo, int n_gape, int state, int is_diff, int current_stage)
//create a new entry in memory store
{
	stack[current_stage].x = k; // k -> x
	stack[current_stage].y = l; // l -> y
	stack[current_stage].z = i; // length -> z
	if (is_diff)stack[current_stage].w = i; //lastdiffpos -> w

	stack_mm[current_stage].x = n_mm; // mm -> x
	stack_mm[current_stage].y = n_gapo; // gapo -> y
	stack_mm[current_stage].z = n_gape; // gape -> z
	stack_mm[current_stage].w = state; // state -> w

	char4 temp = {0,0,0,0};
	pushes[current_stage] = temp;
	return;
}

__device__ int cuda_split_dfs_match(const int len, const unsigned char *str, const int sequence_type, unsigned int *widths, unsigned char *bids, const gap_opt_t *opt, alignment_store_t *aln, int best_score, const int max_aln)
//This function tries to find the alignment of the sequence and returns SA coordinates, no. of mismatches, gap openings and extensions
//It uses a depth-first search approach rather than breath-first as the memory available in CUDA is far less than in CPU mode
//The search rooted from the last char [len] of the sequence to the first with the whole bwt as a ref from start
//and recursively narrow down the k(upper) & l(lower) SA boundaries until it reaches the first char [i = 0], if k<=l then a match is found.
{

	//Initialisations
	int start_pos = aln->start_pos;
	// only obey the sequence_type for the first run
	int best_diff = (start_pos)? aln->init.best_diff :opt->max_diff + 1;
	int max_diff = opt->max_diff;
	//int best_cnt = (start_pos)? aln->init.best_cnt:0;
	int best_cnt = 0;
	const bwt_t * bwt = (sequence_type == 0)? &rbwt_cuda: &bwt_cuda; // rbwt for sequence 0 and bwt for sequence 1;
	const int bwt_type = 1 - sequence_type;
	int current_stage = 0;
	uint4 entries_info[MAX_SEQUENCE_LENGTH];
	uchar4 entries_scores[MAX_SEQUENCE_LENGTH];
	char4 done_push_types[MAX_SEQUENCE_LENGTH];
	int n_aln = (start_pos)? aln->no_of_alignments : 0;
	int loop_count = 0;
	const int max_count = options_cuda.max_entries;


	/* debug to print out seq only for a first 5, trick to unserialise
	if (!start_pos && aln->sequence_id < 5 && sequence_type == 0) {
		// trick to serialise execution
		for (int g = 0; g < 5; g++) {
			if (g == aln->sequence_id) {
				printf("seq id: %d",aln->sequence_id);
				for (int x = 0; x<len; x++) {
					printf(".%d",str[x]);
				}
				printf("\n");
			}
		}
	}
	*/

	//Initialise memory stores first in, last out
	cuda_dfs_initialize(entries_info, entries_scores, done_push_types/*, scores*/); // basically zeroes out the stack

	//push first entry, the first char of the query sequence into memory stores for evaluation
	cuda_dfs_push(entries_info, entries_scores, done_push_types, len, aln->init.lim_k, aln->init.lim_l, aln->init.cur_n_mm, aln->init.cur_n_gapo, aln->init.cur_n_gape, aln->init.cur_state, 0, current_stage); //push initial entry to start


	#if DEBUG_LEVEL > 6
	printf("initial k:%u, l: %u \n", aln->init.lim_k, aln->init.lim_l);
	#endif

	#if DEBUG_LEVEL > 6
	for (int x = 0; x<len; x++) {
		printf(".%d",str[x]);
	}

	// print out the widths and bids
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%i,",bids[x]);
	}
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%d;",widths[x]);
	}


	printf("\n");

	printf("max_diff: %d\n", max_diff);

	#endif



	while(current_stage >= 0)
	{

		int i,j, m;
		int hit_found, allow_diff, allow_M;
		unsigned int k, l;
		char e_n_mm, e_n_gapo, e_n_gape, e_state;
		unsigned int occ;
		loop_count ++;

		int worst_tolerated_score = (options_cuda.mode & BWA_MODE_NONSTOP)? 1000: best_score + options_cuda.s_mm;



		//define break from loop conditions

		if (n_aln == max_aln) {
#if DEBUG_LEVEL > 7
			printf("breaking on n_aln == max_aln\n");
#endif
			break;
		}
		// TODO tweak this, otherwise we miss some branches
		if (best_cnt > options_cuda.max_top2 + (start_pos==0)*2) {
		//if (best_cnt > options_cuda.max_top2) {
#if DEBUG_LEVEL > 7
			printf("breaking on best_cnt>...\n");
#endif
			break;
		}
		if (loop_count > max_count) {
#if DEBUG_LEVEL > 7
			printf("loop_count > max_count\n");
#endif
			break;

		}


		//put extracted entry into local variables
		k = entries_info[current_stage].x; // SA interval
		l = entries_info[current_stage].y; // SA interval
		i = entries_info[current_stage].z; // length
		e_n_mm = entries_scores[current_stage].x; // no of mismatches
		e_n_gapo = entries_scores[current_stage].y; // no of gap openings
		e_n_gape = entries_scores[current_stage].z; // no of gap extensions
		e_state = entries_scores[current_stage].w; // state (M/I/D)


//		// TODO seed length adjustment - get this working after the split length - is it even important?
//		// debug information
//		if (aln->sequence_id == 1 && i > len-2) {
//			printf("\n\ninlocal maxdiff: %d\n", opt->max_diff);
//			printf("inlocal seed diff: %d\n\n", opt->max_seed_diff);
//			printf("inlocal seed length: %d\n", opt->seed_len);
//			printf("inlocal read length: %d\n", len);
//			printf("inlocal start_pos: %d\n", start_pos);
//			printf("inlocal seed_pos: %d, %d\n\n\n", start_pos + (len-i), i);
//		}
		//max_diff = (start_pos + (len-i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;

		// new version not applying seeding after the split
		max_diff = (!start_pos && (len-i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;


		//max_diff = (start_pos + (i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;

//      yet another attempt, did not work
//		if (start_pos) {
//			if (len - i < 1) {
//				max_diff = 4;
//			} else {
//				max_diff = 4;
//			}
//		} else {
//			max_diff = 2;
//		}

		//calculate score
		int score = e_n_mm * options_cuda.s_mm + e_n_gapo * options_cuda.s_gapo + e_n_gape * options_cuda.s_gape;



		//calculate the allowance for differences
		m = max_diff - e_n_mm - e_n_gapo;


#if DEBUG_LEVEL > 7
		printf("k:%u, l: %u, i: %i, score: %d, cur.stage: %d, mm: %d, go: %d, ge: %d, m: %d\n", k, l,i, score, current_stage, e_n_mm, e_n_gapo, e_n_gape, m);
#endif

		if (options_cuda.mode & BWA_MODE_GAPE) m -= e_n_gape;


		if(score > worst_tolerated_score) break;

		// check if the entry is outside boundary or is over the max diff allowed)
		if (m < 0 || (i > 0 && m < bids[i-1]))
		{
#if DEBUG_LEVEL > 6

			printf("breaking: %d, m:%d\n", bids[i-1],m);
#endif
			current_stage --;
			continue;
		}

		// check whether a hit (full sequence when it reaches the last char, i.e. i = 0) is found, if it is, record the alignment information
		hit_found = 0;
		if (!i)
		{
			hit_found = 1;
		}else if (!m) // alternatively if no difference is allowed, just do exact match)
		{
			if ((e_state == STATE_M ||(options_cuda.mode&BWA_MODE_GAPE) || e_n_gape == opt->max_gape))
			{
				if (bwt_cuda_match_exact(bwt_type, i, str, &k, &l))
				{
					hit_found = 1;
				}else
				{
					current_stage --;
					continue; // if there is no hit, then go backwards to parent stage
				}
			}
		}


		if (hit_found)
		{
			// action for found hits
			//int do_add = 1;

			if (score < best_score)
			{
				best_score = score;
				best_diff = e_n_mm + e_n_gapo + (options_cuda.mode & BWA_MODE_GAPE) * e_n_gape;
				best_cnt = 0; //reset best cnt if new score is better
				if (!(options_cuda.mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;

			if (e_n_gapo)
			{ // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array unless the new score is better.
				for (j = 0; j < n_aln; ++j)
					if (aln->alignment_info[j].k == k && aln->alignment_info[j].l == l) break;
				if (j < n_aln)
				{
					if (score < aln->alignment_info[j].score)
						{
							aln->alignment_info[j].score = score;
							aln->alignment_info[j].n_mm = e_n_mm;
							aln->alignment_info[j].n_gapo = e_n_gapo;
							aln->alignment_info[j].n_gape = e_n_gape;
							aln->alignment_info[j].score = score;
							aln->alignment_info[j].best_cnt = best_cnt;
							aln->alignment_info[j].best_diff = best_diff;

						}
					//do_add = 0;
					hit_found = 0;
#if DEBUG_LEVEL > 8
printf("alignment already present, amending score\n");
#endif
				}
			}

			if (hit_found)
			{ // append result the alignment record array
				gap_stack_shadow_cuda(l - k + 1, len, bwt->seq_len, e_state,
						widths, bids);
					// record down number of mismatch, gap open, gap extension and a??

					aln->alignment_info[n_aln].n_mm = entries_scores[current_stage].x;
					aln->alignment_info[n_aln].n_gapo = entries_scores[current_stage].y;
					aln->alignment_info[n_aln].n_gape = entries_scores[current_stage].z;
					aln->alignment_info[n_aln].a = sequence_type;
					// the suffix array interval
					aln->alignment_info[n_aln].k = k;
					aln->alignment_info[n_aln].l = l;
					aln->alignment_info[n_aln].score = score;
					aln->alignment_info[n_aln].best_cnt = best_cnt;
					aln->alignment_info[n_aln].best_diff = best_diff;
#if DEBUG_LEVEL > 8
					printf("alignment added: k:%u, l: %u, i: %i, score: %d, cur.stage: %d, m:%d\n", k, l, i, score, current_stage, m);
#endif
					++n_aln;

			}
			current_stage --;
			continue;
		}




		// proceed and evaluate the next base on sequence
		--i;

		// retrieve Occurrence values and determine all the eligible daughter nodes, done only once at the first instance and skip when it is revisiting the stage
		unsigned int ks[MAX_SEQUENCE_LENGTH][4], ls[MAX_SEQUENCE_LENGTH][4];
		char eligible_cs[MAX_SEQUENCE_LENGTH][5], no_of_eligible_cs=0;

		if(!done_push_types[current_stage].x)
		{
			uint4 cuda_cnt_k = (!sequence_type)? rbwt_cuda_occ4(k-1): bwt_cuda_occ4(k-1);
			uint4 cuda_cnt_l = (!sequence_type)? rbwt_cuda_occ4(l): bwt_cuda_occ4(l);
			ks[current_stage][0] = bwt->L2[0] + cuda_cnt_k.x + 1;
			ls[current_stage][0] = bwt->L2[0] + cuda_cnt_l.x;
			ks[current_stage][1] = bwt->L2[1] + cuda_cnt_k.y + 1;
			ls[current_stage][1] = bwt->L2[1] + cuda_cnt_l.y;
			ks[current_stage][2] = bwt->L2[2] + cuda_cnt_k.z + 1;
			ls[current_stage][2] = bwt->L2[2] + cuda_cnt_l.z;
			ks[current_stage][3] = bwt->L2[3] + cuda_cnt_k.w + 1;
			ls[current_stage][3] = bwt->L2[3] + cuda_cnt_l.w;

			if (ks[current_stage][0] <= ls[current_stage][0])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 0;
			}
			if (ks[current_stage][1] <= ls[current_stage][1])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 1;
			}
			if (ks[current_stage][2] <= ls[current_stage][2])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 2;
			}
			if (ks[current_stage][3] <= ls[current_stage][3])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 3;
			}
			eligible_cs[current_stage][4] = no_of_eligible_cs;
		}else
		{
			no_of_eligible_cs = eligible_cs[current_stage][4];
		}

		// test whether difference is allowed
		allow_diff = 1;
		allow_M = 1;

		if (i)
		{
			if (bids[i-1] > m -1)
			{
				allow_diff = 0;
			}else if (bids[i-1] == m-1 && bids[i] == m-1 && widths[i-1] == widths[i])
			{
				allow_M = 0;
			}
		}

		//donepushtypes stores information for each stage whether a prospective daughter node has been evaluated or not
		//donepushtypes[current_stage].x  exact match, =0 not done, =1 done
		//donepushtypes[current_stage].y  mismatches, 0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].z  deletions, =0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].w  insertions match, =0 not done, =1 done
		//.z and .w are shared among gap openings and extensions as they are mutually exclusive


		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// exact match
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//try exact match first
		if (!done_push_types[current_stage].x)
		{
			#if DEBUG_LEVEL > 8
			printf("trying exact\n");
			#endif

			//shifted already
			int c = str[i];
			//if (start_pos) c = 3;
			done_push_types[current_stage].x = 1;
			if (c < 4)
			{
				#if DEBUG_LEVEL > 8
				printf("c:%i, i:%i\n",c,i);
				 printf("k:%u\n",ks[current_stage][c]);
				 printf("l:%u\n",ls[current_stage][c]);
				#endif

				if (ks[current_stage][c] <= ls[current_stage][c])
				{
					#if DEBUG_LEVEL > 8
					printf("ex match found\n");
					#endif

					cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape, STATE_M, 0, current_stage+1);
					current_stage++;
					continue;
				}
			}
		}else if (score == worst_tolerated_score)
		{
			allow_diff = 0;
		}

		//if (i<20) break;
		if (allow_diff)
		{
			#if DEBUG_LEVEL > 8
			printf("trying inexact...\n");
			#endif
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// mismatch
			////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (done_push_types[current_stage].y < no_of_eligible_cs) //check if done before
			{
				int c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
				done_push_types[current_stage].y++;
				if (allow_M) // daughter node - mismatch
				{
					if (score + options_cuda.s_mm <= worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (c != str[i])
						{
							// TODO is the debug message ok?
							#if DEBUG_LEVEL > 8
							 printf("mismatch confirmed\n");
							#endif
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}else if (done_push_types[current_stage].y < no_of_eligible_cs)
						{
							c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
							done_push_types[current_stage].y++;
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}
					}
				}
			}
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Indels (Insertions/Deletions)
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				if (!e_state) // daughter node - opening a gap insertion or deletion
				{
					if (score + options_cuda.s_gapo <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (e_n_gapo < opt->max_gapo)
						{
							if (!done_push_types[current_stage].w)
							{	//insertions
								done_push_types[current_stage].w = 1;
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
										current_stage++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i, k, l, e_n_mm, e_n_gapo + 1, e_n_gape, STATE_I, 1, current_stage);
										continue;
								}
							}
							else if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
							{	//deletions
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									done_push_types[current_stage].z++;
									cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo + 1, e_n_gape, STATE_D, 1, current_stage+1);
									current_stage++; //advance stage number by 1
									continue;
								}
								else
								{
									done_push_types[current_stage].z++;
								}
							}
						}
					}
				}else if (e_state == STATE_I) //daughter node - extend an insertion entry
				{
					if(!done_push_types[current_stage].w)  //check if done before
					{
						done_push_types[current_stage].w = 1;
						if (e_n_gape < opt->max_gape)  //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									current_stage++; //advance stage number by 1
									cuda_dfs_push(entries_info, entries_scores,  done_push_types, i, k, l, e_n_mm, e_n_gapo, e_n_gape + 1, STATE_I, 1, current_stage);
									continue; //skip the rest and proceed to next stage
								}
							}
						}
					}
				}else if (e_state == STATE_D) //daughter node - extend a deletion entry
				{
					occ = l - k + 1;
					if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
					{
						if (e_n_gape < opt->max_gape) //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								if (e_n_gape + e_n_gapo < max_diff || occ < options_cuda.max_del_occ)
								{
									unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;

									if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
									{
										int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
										done_push_types[current_stage].z++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape + 1, STATE_D, 1, current_stage+1);
										current_stage++; //advance stage number
										continue;
									}
								}
							}
						}
						else
						{
							done_push_types[current_stage].z++;
						}
					}
				} //end else if (e_state == STATE_D)*/

		}//end if (!allow_diff)
		current_stage--;

	} //end do while loop



	aln->no_of_alignments = n_aln;

	return best_score;
}


__device__ int cuda_dfs_match(const int len, const unsigned char *str, const int sequence_type, unsigned int *widths, unsigned char *bids, const gap_opt_t *opt, alignment_store_t *aln, int best_score, const int max_aln)
//This function tries to find the alignment of the sequence and returns SA coordinates, no. of mismatches, gap openings and extensions
//It uses a depth-first search approach rather than breath-first as the memory available in CUDA is far less than in CPU mode
//The search rooted from the last char [len] of the sequence to the first with the whole bwt as a ref from start
//and recursively narrow down the k(upper) & l(lower) SA boundaries until it reaches the first char [i = 0], if k<=l then a match is found.
{

	//Initialisations
	int best_diff = opt->max_diff + 1;

	int best_cnt = 0;
	const bwt_t * bwt = (sequence_type == 0)? &rbwt_cuda: &bwt_cuda; // rbwt for sequence 0 and bwt for sequence 1;
	const int bwt_type = 1 - sequence_type;
	int current_stage = 0;
	uint4 entries_info[MAX_SEQUENCE_LENGTH];
	uchar4 entries_scores[MAX_SEQUENCE_LENGTH];
	char4 done_push_types[MAX_SEQUENCE_LENGTH];
	int n_aln = 0;
	int loop_count = 0;
	const int max_count = options_cuda.max_entries;

	//Initialise memory stores first in, last out
	cuda_dfs_initialize(entries_info, entries_scores, done_push_types/*, scores*/); //initialize initial entry, current stage set at 0 and done push type = 0

	//push first entry, the first char of the query sequence into memory stores for evaluation
	cuda_dfs_push(entries_info, entries_scores, done_push_types, len, 0, bwt->seq_len, 0, 0, 0, 0, 0, current_stage); //push initial entry to start

#if DEBUG_LEVEL > 6
	printf("initial k:%u, l: %u \n", 0, bwt->seq_len);
#endif

#if DEBUG_LEVEL > 6
	for (int x = 0; x<len; x++) {
		printf(".%d",str[x]);
	}

	// print out the widths and bids
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%i,",bids[x]);
	}
	printf("\n");
	for (int x = 0; x<len; x++) {
		printf("%d;",widths[x]);
	}


printf("\n");

printf("max_diff: %d\n", max_diff);

#endif

	while(current_stage >= 0)
	{
		int i,j, m;
		int hit_found, allow_diff, allow_M;
		unsigned int k, l;
		char e_n_mm, e_n_gapo, e_n_gape, e_state;
		unsigned int occ;
		loop_count ++;

		int worst_tolerated_score = (options_cuda.mode & BWA_MODE_NONSTOP)? 1000: best_score + options_cuda.s_mm;


		//define break from loop conditions

		if (n_aln == max_aln) {
#if DEBUG_LEVEL > 7
			printf("breaking on n_aln == max_aln\n");
#endif
			break;
		}
		if (best_cnt > options_cuda.max_top2) {
#if DEBUG_LEVEL > 7
			printf("breaking on best_cnt>...\n");
#endif
			break;
		}
		if (loop_count > max_count) {
#if DEBUG_LEVEL > 7
			printf("loop_count > max_count\n");
#endif
			break;

		}

		//put extracted entry into local variables
		k = entries_info[current_stage].x; // SA interval
		l = entries_info[current_stage].y; // SA interval
		i = entries_info[current_stage].z; // length
		e_n_mm = entries_scores[current_stage].x; // no of mismatches
		e_n_gapo = entries_scores[current_stage].y; // no of gap openings
		e_n_gape = entries_scores[current_stage].z; // no of gap extensions
		e_state = entries_scores[current_stage].w; // state (M/I/D)

		//calculate score
		int score = e_n_mm * options_cuda.s_mm + e_n_gapo * options_cuda.s_gapo + e_n_gape * options_cuda.s_gape;

		//seeding diff
		int max_diff = ((len-i) <  opt->seed_len)? opt->max_seed_diff : opt->max_diff;
//		int max_diff= opt->max_diff;

		//calculate the allowance for differences
		m = max_diff - e_n_mm - e_n_gapo;

#if DEBUG_LEVEL > 7
		printf("k:%u, l: %u, i: %i, score: %d, cur.stage: %d, mm: %d, go: %d, ge: %d, m: %d\n", k, l,i, score, current_stage, e_n_mm, e_n_gapo, e_n_gape, m);
#endif


		if (options_cuda.mode & BWA_MODE_GAPE) m -= e_n_gape;

		if(score > worst_tolerated_score) break;

		// check if the entry is outside boundary or is over the max diff allowed)
		if (m < 0 || (i > 0 && m < bids[i-1]))
		{
#if DEBUG_LEVEL > 6
			printf("breaking: %d, m:%d\n", bids[i-1],m);
#endif
			current_stage --;
			continue;
		}

		// check whether a hit (full sequence when it reaches the last char, i.e. i = 0) is found, if it is, record the alignment information
		hit_found = 0;
		if (!i)
		{
			hit_found = 1;
		}else if (!m) // alternatively if no difference is allowed, just do exact match)
		{
			if ((e_state == STATE_M ||(options_cuda.mode&BWA_MODE_GAPE) || e_n_gape == opt->max_gape))
			{
				if (bwt_cuda_match_exact(bwt_type, i, str, &k, &l))
				{
					hit_found = 1;
				}else
				{
					current_stage --;
					continue; // if there is no hit, then go backwards to parent stage
				}
			}
		}
		if (hit_found)
		{
			// action for found hits
			//int do_add = 1;

			if (score < best_score)
			{
				best_score = score;
				best_diff = e_n_mm + e_n_gapo + (options_cuda.mode & BWA_MODE_GAPE) * e_n_gape;
				best_cnt = 0; //reset best cnt if new score is better
				if (!(options_cuda.mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;

			if (e_n_gapo)
			{ // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array unless the new score is better.
				for (j = 0; j < n_aln; ++j)
					if (aln->alignment_info[j].k == k && aln->alignment_info[j].l == l) break;
				if (j < n_aln)
				{
					if (score < aln->alignment_info[j].score)
						{
							aln->alignment_info[j].score = score;
							aln->alignment_info[j].n_mm = e_n_mm;
							aln->alignment_info[j].n_gapo = e_n_gapo;
							aln->alignment_info[j].n_gape = e_n_gape;
						}
					//do_add = 0;
					hit_found = 0;
				}
			}

			if (hit_found)
			{ // append result the alignment record array
				gap_stack_shadow_cuda(l - k + 1, len, bwt->seq_len, entries_info[current_stage].w, widths, bids);
					// record down number of mismatch, gap open, gap extension and a??

					aln->alignment_info[n_aln].n_mm = entries_scores[current_stage].x;
					aln->alignment_info[n_aln].n_gapo = entries_scores[current_stage].y;
					aln->alignment_info[n_aln].n_gape = entries_scores[current_stage].z;
					aln->alignment_info[n_aln].a = sequence_type;
					// the suffix array interval
					aln->alignment_info[n_aln].k = k;
					aln->alignment_info[n_aln].l = l;
					aln->alignment_info[n_aln].score = score;
#if DEBUG_LEVEL > 8
					printf("alignment added: k:%u, l: %u, i: %i, score: %d, cur.stage: %d, m:%d\n", k, l, i, score, current_stage, m);
#endif

					++n_aln;

			}
			current_stage --;
			continue;
		}

		// proceed and evaluate the next base on sequence
		--i;

		// retrieve Occurrence values and determine all the eligible daughter nodes, done only once at the first instance and skip when it is revisiting the stage
		unsigned int ks[MAX_SEQUENCE_LENGTH][4], ls[MAX_SEQUENCE_LENGTH][4];
		char eligible_cs[MAX_SEQUENCE_LENGTH][5], no_of_eligible_cs=0;

		if(!done_push_types[current_stage].x)
		{
			uint4 cuda_cnt_k = (!sequence_type)? rbwt_cuda_occ4(k-1): bwt_cuda_occ4(k-1);
			uint4 cuda_cnt_l = (!sequence_type)? rbwt_cuda_occ4(l): bwt_cuda_occ4(l);
			ks[current_stage][0] = bwt->L2[0] + cuda_cnt_k.x + 1;
			ls[current_stage][0] = bwt->L2[0] + cuda_cnt_l.x;
			ks[current_stage][1] = bwt->L2[1] + cuda_cnt_k.y + 1;
			ls[current_stage][1] = bwt->L2[1] + cuda_cnt_l.y;
			ks[current_stage][2] = bwt->L2[2] + cuda_cnt_k.z + 1;
			ls[current_stage][2] = bwt->L2[2] + cuda_cnt_l.z;
			ks[current_stage][3] = bwt->L2[3] + cuda_cnt_k.w + 1;
			ls[current_stage][3] = bwt->L2[3] + cuda_cnt_l.w;

			if (ks[current_stage][0] <= ls[current_stage][0])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 0;
			}
			if (ks[current_stage][1] <= ls[current_stage][1])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 1;
			}
			if (ks[current_stage][2] <= ls[current_stage][2])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 2;
			}
			if (ks[current_stage][3] <= ls[current_stage][3])
			{
				eligible_cs[current_stage][no_of_eligible_cs++] = 3;
			}
			eligible_cs[current_stage][4] = no_of_eligible_cs;
		}else
		{
			no_of_eligible_cs = eligible_cs[current_stage][4];
		}

		// test whether difference is allowed
		allow_diff = 1;
		allow_M = 1;

		if (i)
		{
			if (bids[i-1] > m -1)
			{
				allow_diff = 0;
			}else if (bids[i-1] == m-1 && bids[i] == m-1 && widths[i-1] == widths[i])
			{
				allow_M = 0;
			}
		}

		//donepushtypes stores information for each stage whether a prospective daughter node has been evaluated or not
		//donepushtypes[current_stage].x  exact match, =0 not done, =1 done
		//donepushtypes[current_stage].y  mismatches, 0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].z  deletions, =0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].w  insertions match, =0 not done, =1 done
		//.z and .w are shared among gap openings and extensions as they are mutually exclusive


		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// exact match
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//try exact match first
		if (!done_push_types[current_stage].x)
		{
#if DEBUG_LEVEL > 8
			printf("trying exact\n");
#endif
			int c = str[i];
			done_push_types[current_stage].x = 1;
			if (c < 4)
			{
#if DEBUG_LEVEL > 8
				printf("c:%i, i:%i\n",c,i);
				printf("k:%u\n",ks[current_stage][c]);
				printf("l:%u\n",ls[current_stage][c]);
#endif

				if (ks[current_stage][c] <= ls[current_stage][c])
				{
					#if DEBUG_LEVEL > 8
					printf("ex match found\n");
					#endif
					cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape, STATE_M, 0, current_stage+1);
					current_stage++;
					continue;
				}
			}
		}else if (score == worst_tolerated_score)
		{
			allow_diff = 0;
		}

		if (allow_diff)
		{
		#if DEBUG_LEVEL > 8
			printf("trying inexact...\n");
		#endif
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// mismatch
			////////////////////////////////////////////////////////////////////////////////////////////////////////////

			if (done_push_types[current_stage].y < no_of_eligible_cs) //check if done before
			{
				int c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
				done_push_types[current_stage].y++;
				if (allow_M) // daughter node - mismatch
				{
					if (score + options_cuda.s_mm <= worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (c != str[i])
						{
							// TODO is the debug message ok?
							#if DEBUG_LEVEL > 8
							 printf("mismatch confirmed\n");
							#endif
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}else if (done_push_types[current_stage].y < no_of_eligible_cs)
						{
							c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
							done_push_types[current_stage].y++;
							cuda_dfs_push(entries_info, entries_scores, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e_n_mm + 1, e_n_gapo, e_n_gape, STATE_M, 1, current_stage+1);
							current_stage++;
							continue;
						}
					}
				}
			}
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Indels (Insertions/Deletions)
				////////////////////////////////////////////////////////////////////////////////////////////////////////////
				if (!e_state) // daughter node - opening a gap insertion or deletion
				{
					if (score + options_cuda.s_gapo <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						if (e_n_gapo < opt->max_gapo)
						{
							if (!done_push_types[current_stage].w)
							{	//insertions
								done_push_types[current_stage].w = 1;
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
										current_stage++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i, k, l, e_n_mm, e_n_gapo + 1, e_n_gape, STATE_I, 1, current_stage);
										continue;
								}
							}
							else if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
							{	//deletions
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									done_push_types[current_stage].z++;
									cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo + 1, e_n_gape, STATE_D, 1, current_stage+1);
									current_stage++; //advance stage number by 1
									continue;
								}
								else
								{
									done_push_types[current_stage].z++;
								}
							}
						}
					}
				}else if (e_state == STATE_I) //daughter node - extend an insertion entry
				{
					if(!done_push_types[current_stage].w)  //check if done before
					{
						done_push_types[current_stage].w = 1;
						if (e_n_gape < opt->max_gape)  //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									current_stage++; //advance stage number by 1
									cuda_dfs_push(entries_info, entries_scores,  done_push_types, i, k, l, e_n_mm, e_n_gapo, e_n_gape + 1, STATE_I, 1, current_stage);
									continue; //skip the rest and proceed to next stage
								}
							}
						}
					}
				}else if (e_state == STATE_D) //daughter node - extend a deletion entry
				{
					occ = l - k + 1;
					if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
					{
						if (e_n_gape < opt->max_gape) //skip if no of gap ext is beyond limit
						{
							if (score + options_cuda.s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
							{
								if (e_n_gape + e_n_gapo < max_diff || occ < options_cuda.max_del_occ)
								{
									unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;

									if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
									{
										int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
										done_push_types[current_stage].z++;
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape + 1, STATE_D, 1, current_stage+1);
										current_stage++; //advance stage number
										continue;
									}
								}
							}
						}
						else
						{
							done_push_types[current_stage].z++;
						}
					}
				} //end else if (e_state == STATE_D)*/

		}//end if (!allow_diff)
		current_stage--;

	} //end do while loop

	aln->no_of_alignments = n_aln;

	return best_score;
}

__global__ void cuda_inexact_match_caller(int no_of_sequences, unsigned short max_sequence_length, alignment_store_t* global_alignment_store, unsigned char cuda_opt)
//CUDA kernal for inexact match on both strands
//calls bwt_cuda_device_calculate_width to determine the boundaries of the search space
//and then calls dfs_match to search for alignment using dfs approach
{
	// Block ID for CUDA threads, as there is only 1 thread per block possible for now
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	//Local store for sequence widths bids and alignments
	unsigned int local_sequence_widths[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence_bids[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence[MAX_SEQUENCE_LENGTH];
	unsigned char local_rc_sequence[MAX_SEQUENCE_LENGTH];
	alignment_store_t local_alignment_store;

	//fetch the alignment store from memory
	local_alignment_store = global_alignment_store[blockId];


	int max_aln = options_cuda.max_aln;
	//initialize local options for each query sequence
	gap_opt_t local_options = options_cuda;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences )
	{
		//get sequences from texture memory
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, local_alignment_store.sequence_id);
		local_alignment_store.finished = 1; //only one run for simple kernel
		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
		unsigned int last_read = ~0;
		unsigned int last_read_data;

		for (int i = 0; i < sequence_length; ++i)
		{
			unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
			local_sequence[i] = c;
			if (local_options.mode & BWA_MODE_COMPREAD)
			{
				local_rc_sequence[i] = (c > 3)? c : (3 - c);
			}else
			{
				local_rc_sequence[i] = c;
			}
		}

		//initialize local options
		if (options_cuda.fnr > 0.0) local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_length, BWA_AVG_ERR, options_cuda.fnr);
		if (local_options.max_diff < options_cuda.max_gapo) local_options.max_gapo = local_options.max_diff;
		//the worst score is lowered from +1 (bwa) to +0 to tighten the search space esp. for long reads
		int worst_score = aln_score2(local_options.max_diff, local_options.max_gapo, local_options.max_gape, local_options);

		//test if there is too many Ns, if true, skip everything and return 0 number of alignments.

		int N = 0;
		for (int i = 0 ; i < sequence_length; ++i)
		{

			if (local_sequence[i] > 3) ++N;
			if (N > local_options.max_diff)
			{
				global_alignment_store[blockId].no_of_alignments = 0;
				return;
			}
		}

		//work on main sequence, i.e. reverse sequence (bwt for w, rbwt for match)
		int sequence_type = 0;

		// Calculate w
		syncthreads();
		bwt_cuda_device_calculate_width(local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, sequence_length);

		//Align with forward reference sequence
		//syncthreads();
		int best_score = cuda_dfs_match(sequence_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);

		// copy alignment info to global memory
		#if OUTPUT_ALIGNMENTS == 1
		global_alignment_store[blockId] = local_alignment_store;
		int no_aln = local_alignment_store.no_of_alignments;
		#endif // OUTPUT_ALIGNMENTS == 1


		//work on reverse complementary sequence (rbwt for w, bwt for match)
		sequence_type = 1;

		// Calculate w
		syncthreads();
		bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, sequence_length);

		//Align with reverse reference sequence
		syncthreads();
		cuda_dfs_match(sequence_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, best_score, max_aln);

		// copy alignment info to global memory
		#if OUTPUT_ALIGNMENTS == 1
		short rc_no_aln = 0;
		while (rc_no_aln <= (max_aln + max_aln - no_aln) && rc_no_aln < local_alignment_store.no_of_alignments)
		{
			global_alignment_store[blockId].alignment_info[no_aln + rc_no_aln] = local_alignment_store.alignment_info[rc_no_aln];
			rc_no_aln++;
		}
		global_alignment_store[blockId].no_of_alignments = local_alignment_store.no_of_alignments + no_aln;
		#endif // OUTPUT_ALIGNMENTS == 1

	}
	return;
}

__global__ void cuda_split_inexact_match_caller(int no_of_sequences, unsigned short max_sequence_length, alignment_store_t* global_alignment_store, unsigned char cuda_opt)
//CUDA kernal for inexact match on a specified strand
// modified for split kernels
//calls bwt_cuda_device_calculate_width to determine the boundaries of the search space
//and then calls dfs_match to search for alignment using dfs approach
{

	// Block ID for CUDA threads, as there is only 1 thread per block possible for now
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;
	//Local store for sequence widths bids and alignments
	unsigned int local_sequence_widths[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence_bids[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence[MAX_SEQUENCE_LENGTH];
	unsigned char local_rc_sequence[MAX_SEQUENCE_LENGTH];



	alignment_store_t local_alignment_store;

	//fetch the alignment store from memory
	local_alignment_store = global_alignment_store[blockId];

	int max_aln = options_cuda.max_aln;
	//initialize local options for each query sequence
	gap_opt_t local_options = options_cuda;

	const int pass_length = (options_cuda.seed_len > PASS_LENGTH)? options_cuda.seed_len : PASS_LENGTH;
	const int split_engage = pass_length + 6;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences )
	{
#if DEBUG_LEVEL > 5
		printf("start..\n");
#endif

		//get sequences from texture memory
		//const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);

		// sequences no longer align with the block ids
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, local_alignment_store.sequence_id);


		const unsigned int sequence_offset = sequence_info.x;
		unsigned int last_read = ~0;
		unsigned int last_read_data;

		//calculate new length - are we dealing with the last bit?
		int start_pos = local_alignment_store.start_pos;

		unsigned short process_length;

		// decide if this is the last part to process
		if (!start_pos && sequence_info.y >= split_engage) {
			// first round and splitting is going to happen, forcing if necessary
			process_length = min(sequence_info.y, pass_length);
		} else {
			// subsequent rounds or splitting is not happening
			if (sequence_info.y - start_pos < pass_length * 2) {
				// mark this pass as last
				local_alignment_store.finished = 1;
				if (sequence_info.y - start_pos > pass_length) {
					// "natural" splitting finish
					process_length = min(sequence_info.y, sequence_info.y%pass_length + pass_length);
				} else {
					// last pass of "forced" splitting
					process_length = sequence_info.y - start_pos;
				}

			} else {
				process_length = min(sequence_info.y, pass_length);
			}
		}


#if DEBUG_LEVEL > 7
		printf("process length: %d, start_pos: %d, sequence_length: %d\n", process_length, start_pos, sequence_info.y);
#endif
		//const unsigned short sequence_length = (!start_pos) ? process_length : sequence_info.y;

		// TODO can be slightly sped up for one directional matching
		for (int i = 0; i < process_length; ++i)
		{
			//offsetting works fine, again, from the back of the seq.
			// copies from the end to the beginning
			unsigned char c = read_char(sequence_offset + i + (sequence_info.y- process_length - start_pos), &last_read, &last_read_data );

			local_sequence[i] = c;

			if (local_options.mode & BWA_MODE_COMPREAD)
			{
				local_rc_sequence[i] = (c > 3)? c : (3 - c);
			}else
			{
				local_rc_sequence[i] = c;
			}
		}


#define SEEDING 0

		if (options_cuda.fnr > 0.0) {
			//tighten the search for the first bit of sequence
#if SEEDING == 1
			if (!start_pos) {
				local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_length, BWA_AVG_ERR, options_cuda.fnr);

			} else {
#endif
				local_options.max_diff = bwa_cuda_cal_maxdiff(sequence_info.y, BWA_AVG_ERR, options_cuda.fnr);
#if SEEDING == 1
			}
#endif
		}

//		// TODO remove debug out
//		if (blockId == 1) {
//			printf("\n\nlocal maxdiff: %d\n", local_options.max_diff);
//			printf("local seed diff: %d\n\n", local_options.max_seed_diff);
//		}

		if (local_options.max_diff < options_cuda.max_gapo) local_options.max_gapo = local_options.max_diff;

		//the worst score is lowered from +1 (bwa) to +0 to tighten the search space esp. for long reads

		int worst_score = aln_score2(local_options.max_diff, local_options.max_gapo, local_options.max_gape, local_options);


#if DEBUG_LEVEL > 6
		printf("worst score: %d\n", worst_score);
#endif

		//test if there is too many Ns, if true, skip everything and return 0 number of alignments.
		int N = 0;
		for (int i = 0 ; i < process_length; ++i)
		{
			if (local_sequence[i] > 3) ++N;
			if (N > local_options.max_diff)
			{
#if DEBUG_LEVEL > 7
				printf("Not good quality seq, quitting kernel.\n");
#endif
				global_alignment_store[blockId].no_of_alignments = 0;
				return;
			}
		}

		int sequence_type = 0;
		sequence_type = (cuda_opt == 2) ? 1 : local_alignment_store.init.sequence_type;
		// Calculate w
		syncthreads();

#if DEBUG_LEVEL > 7
		printf("calc width..\n");
#endif

		// properly resume for reverse alignment
		if (sequence_type == 1) {
#if DEBUG_LEVEL > 6
			printf("reverse alignment...");
#endif
			//Align to forward reference sequence
			bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);
			cuda_split_dfs_match(process_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		} else {
#if DEBUG_LEVEL > 6
			printf("normal alignment...");
#endif
			bwt_cuda_device_calculate_width(local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);
			cuda_split_dfs_match(process_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		}


		// copy alignment info to global memory
		#if OUTPUT_ALIGNMENTS == 1


		global_alignment_store[blockId] = local_alignment_store;
		#endif // OUTPUT_ALIGNMENTS == 1

		// now align the second strand, only during the first run, subsequent runs do not execute this part
		if (!start_pos && !cuda_opt) {
			int no_aln = local_alignment_store.no_of_alignments;

			sequence_type = 1;
			// Calculate w
			syncthreads();
			bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, process_length);

			//Align to reverse reference sequence
			syncthreads();
			cuda_split_dfs_match(process_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
#if DEBUG_LEVEL > 6
			printf("local_alignment_store.no_of_alignments: %d\n", local_alignment_store.no_of_alignments);
#endif

			// copy alignment info to global memory
			#if OUTPUT_ALIGNMENTS == 1
			short rc_no_aln = 0;
			while (rc_no_aln <= (max_aln + max_aln - no_aln) && rc_no_aln < local_alignment_store.no_of_alignments)
			{
				global_alignment_store[blockId].alignment_info[no_aln + rc_no_aln] = local_alignment_store.alignment_info[rc_no_aln];
				rc_no_aln++;
			}
			global_alignment_store[blockId].no_of_alignments = local_alignment_store.no_of_alignments + no_aln;
			#endif // OUTPUT_ALIGNMENTS == 1
		}
#if DEBUG_LEVEL > 6
		printf("kernel finished\n");
#endif
	}

	/*if (blockId < 5) {
		for (int x = 0; x<len; x++) {
			printf(".%d",str[x]);
		}
	}*/

	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Line below is for BWA CPU MODE


#ifdef HAVE_PTHREAD
#define THREAD_BLOCK_SIZE 1024
#include <pthread.h>
static pthread_mutex_t g_seq_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

// width must be filled as zero
static int bwt_cal_width(const bwt_t *rbwt, int len, const ubyte_t *str, bwt_width_t *width)
{
	bwtint_t k, l, ok, ol;
	int i, bid;
	bid = 0;
	k = 0; l = rbwt->seq_len;

	for (i = 0; i < len; ++i) {
		ubyte_t c = str[i];
		if (c < 4) {
			bwt_2occ(rbwt, k - 1, l, c, &ok, &ol);
			k = rbwt->L2[c] + ok + 1;
			l = rbwt->L2[c] + ol;
		}
		if (k > l || c > 3) { // then restart
			k = 0;
			l = rbwt->seq_len;
			++bid;
		}
		width[i].w = l - k + 1;
		width[i].bid = bid;
	}
	width[len].w = 0;
	width[len].bid = ++bid;

	return bid;
}

static void bwa_cal_sa_reg_gap(int tid, bwt_t *const bwt[2], int no_of_sequences, bwa_seq_t *seqs, const gap_opt_t *opt)
{
	int i, max_l = 0, max_len;
	gap_stack_t *stack;
	bwt_width_t *w[2], *seed_w[2];
	const ubyte_t *seq[2];
	gap_opt_t local_opt = *opt;

	// initiate priority stack
	for (i = max_len = 0; i != no_of_sequences; ++i)
		if (seqs[i].len > max_len) max_len = seqs[i].len;
	if (opt->fnr > 0.0) local_opt.max_diff = bwa_cal_maxdiff(max_len, BWA_AVG_ERR, opt->fnr);
	if (local_opt.max_diff < local_opt.max_gapo) local_opt.max_gapo = local_opt.max_diff;
	stack = gap_init_stack(local_opt.max_diff, local_opt.max_gapo, local_opt.max_gape, &local_opt);

	seed_w[0] = (bwt_width_t*)calloc(opt->seed_len+1, sizeof(bwt_width_t));
	seed_w[1] = (bwt_width_t*)calloc(opt->seed_len+1, sizeof(bwt_width_t));
	w[0] = w[1] = 0;

	for (i = 0; i != no_of_sequences; ++i) {
		bwa_seq_t *p = seqs + i;
		#ifdef HAVE_PTHREAD
		if (opt->n_threads > 1) {
			pthread_mutex_lock(&g_seq_lock);
			if (p->tid < 0) { // unassigned
				int j;
				for (j = i; j < no_of_sequences && j < i + THREAD_BLOCK_SIZE; ++j)
					seqs[j].tid = tid;
			} else if (p->tid != tid) {
				pthread_mutex_unlock(&g_seq_lock);
				continue;
			}
			pthread_mutex_unlock(&g_seq_lock);

		}
		#endif
		p->sa = 0; p->type = BWA_TYPE_NO_MATCH; p->c1 = p->c2 = 0; p->n_aln = 0; p->aln = 0;
		seq[0] = p->seq; seq[1] = p->rseq;
		if (max_l < p->len) {
			max_l = p->len;
			w[0] = (bwt_width_t*)calloc(max_l + 1, sizeof(bwt_width_t));
			w[1] = (bwt_width_t*)calloc(max_l + 1, sizeof(bwt_width_t));
		}
		//fprintf(stderr, "length:%d\n", p->len);
		bwt_cal_width(bwt[0], p->len, seq[0], w[0]);
		bwt_cal_width(bwt[1], p->len, seq[1], w[1]);
		if (opt->fnr > 0.0) local_opt.max_diff = bwa_cal_maxdiff(p->len, BWA_AVG_ERR, opt->fnr);
		if (opt->seed_len >= p->len) local_opt.seed_len = 0x7fffffff;
		if (p->len > opt->seed_len) {
			bwt_cal_width(bwt[0], opt->seed_len, seq[0] + (p->len - opt->seed_len), seed_w[0]);
			bwt_cal_width(bwt[1], opt->seed_len, seq[1] + (p->len - opt->seed_len), seed_w[1]);
		}

		printf("seed length: %d", opt->seed_len);

		#if DEBUG_LEVEL > 6
			for (int x = 0; x<p->len; x++) {
				//printf(".%d",seq[x]);
			}

			// print out the widths and bids
			printf("\n");
			for (int x = 0; x<p->len; x++) {
				printf("%i,",w[0][x].w);
			}
			printf("\n");
			for (int x = 0; x<opt->seed_len; x++) {
				printf("%i,",seed_w[0][x].w);
			}
			printf("\n");
			for (int x = 0; x<p->len; x++) {
				printf("%d;",w[0][x].bid);
			}
			printf("\n");
			for (int x = 0; x<opt->seed_len; x++) {
				printf("%d;",seed_w[0][x].bid);
			}
		#endif



		// core function
		p->aln = bwt_match_gap(bwt, p->len, seq, w, p->len <= opt->seed_len? 0 : seed_w, &local_opt, &p->n_aln, stack);
		// store the alignment


		free(p->name); free(p->seq); free(p->rseq); free(p->qual);
		p->name = 0; p->seq = p->rseq = p->qual = 0;
	}
	free(seed_w[0]); free(seed_w[1]);
	free(w[0]); free(w[1]);
	gap_destroy_stack(stack);
}

#ifdef HAVE_PTHREAD
typedef struct {
	int id;
	bwa_seq_t * sequence;
} thread_data_t;

typedef struct {
	int tid;
	bwt_t *bwt[2];
	int n_seqs;
	bwa_seq_t *seqs;
	const gap_opt_t *opt;
} thread_aux_t;

static void *worker(void *data)
{
	thread_aux_t *d = (thread_aux_t*)data;
	bwa_cal_sa_reg_gap(d->tid, d->bwt, d->n_seqs, d->seqs, d->opt);
	return 0;
}
#endif // HAVE_PTHREAD


// return the difference in second between two timeval structures
double diff_in_seconds(struct timeval *finishtime, struct timeval * starttime)
{
	double sec;
	sec=(finishtime->tv_sec-starttime->tv_sec);
	sec+=(finishtime->tv_usec-starttime->tv_usec)/1000000.0;
	return sec;
}

// Setting default options
gap_opt_t *gap_init_opt()
{
	gap_opt_t *o;
	o = (gap_opt_t*)calloc(1, sizeof(gap_opt_t));
	/* IMPORTANT: s_mm*10 should be about the average base error
	   rate. Violating this requirement will break pairing! */
	o->s_mm = 3; o->s_gapo = 11; o->s_gape = 4;
	o->max_diff = -1;
	o->max_gapo = 1;
	o->max_gape = 6;
	o->indel_end_skip = 5; o->max_del_occ = 10;
	o->max_aln = 10;
	o->max_entries = -1; //max tried is 1M without problem with 125bp can go larger for shorter reads
	o->mode = BWA_MODE_GAPE | BWA_MODE_COMPREAD;
	o->seed_len = PASS_LENGTH;
	o->max_seed_diff = 2;
	o->fnr = 0.04;
	o->n_threads = -1;
	o->max_top2 = 5; // change from 20 in bwa;
	o->mid = 0; //default is no MID tag, i.e 0
	o->cuda_device = -1;
	return o;
}

int bwa_cal_maxdiff(int l, double err, double thres)
{
	double elambda = exp(-l * err);
	double sum, y = 1.0;
	int k, x = 1;
	for (k = 1, sum = elambda; k < 1000; ++k) {
		y *= l * err;
		x *= k;
		sum += elambda * y / x;
		if (1.0 - sum < thres) return k;
	}
	return 2;
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void barracuda_bwa_aln_core(const char *prefix, const char *fn_fa, gap_opt_t *opt, unsigned char cuda_opt)
//Main alignment module caller
//Determines the availability of CUDA devices and
//invokes CUDA kernels cuda_inexact_match_caller
//contains also CPU code for legacy BWA runs

{
	bwa_seqio_t *ks;

	#if STDOUT_BINARY_RESULT == 1
	fwrite(opt, sizeof(gap_opt_t), 1, stdout);
	#endif

	#if MYBINARY == 1
	FILE *mybinary;
	mybinary = fopen ("mybinary.sai","w");
	fwrite(opt, sizeof(gap_opt_t), 1, mybinary);
	#endif

	// total number of sequences read
	unsigned int no_of_sequences = 0;

	// For timing purpose only
	struct timeval start, end;
	double time_used;

	double total_time_used = 0, total_calculation_time_used = 0;
	// total number of sequences read
	unsigned long long total_no_of_sequences = 0;
	// total number of reads in base pairs
	unsigned long long total_no_of_base_pair = 0;
	// total number of bwt_occ structure read in bytes
	unsigned long long bwt_read_size = 0;

	// initialization
	ks = bwa_seq_open(fn_fa);


	if(opt->n_threads == -1) // CUDA MODE
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CUDA mode starts here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		fprintf(stderr,"[aln_core] Running CUDA mode.\n");

		// bwt occurrence array in GPU
		unsigned int * global_bwt = 0;
		// rbwt occurrence array in GPU
		unsigned int * global_rbwt = 0;
		// The length of the longest read;
		unsigned short max_sequence_length=0;
		// maximum read size from sequences in bytes
		unsigned int read_size = 0;
		// sequences reside in global memory of GPU
		unsigned char * global_sequences = 0;
		// sequences reside in main memory of CPU
		unsigned char * main_sequences = 0;
		// sequences index reside in global memory of GPU
		uint2 * global_sequences_index = 0;
		// sequences index reside in main memory of CPU
		uint2 * main_sequences_index = 0;
		// Options from user
		gap_opt_t *options;
		// global alignment stores for device
		alignment_store_t * global_alignment_store_device;
		#if OUTPUT_ALIGNMENTS == 1
		// global alignment stores for host
		alignment_store_t * global_alignment_store_host;
		alignment_store_t * global_alignment_store_host_final;

		#endif // OUTPUT_ALIGNMENTS == 1

		if (opt->max_entries < 0 )
		{
			opt->max_entries = 150000;
		}


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//CUDA options
//		if (!cuda_opt) fprintf(stderr,"[aln_core] Aligning both forward and reverse complementary sequences.\n");
		if (opt->mid) fprintf(stderr,"[aln_core] Not aligning the first %d bases from the 5' end of sequencing reads.\n",opt->mid);
//		if (cuda_opt == 1) fprintf(stderr,"[aln_core] CUDA Options: (f) Align forward sequences only.\n");
//		if (cuda_opt == 2) fprintf(stderr,"[aln_core] CUDA Options: (r) Align reverse complementary sequences only.\n");

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Determine Cuda device
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		size_t mem_available = 0, total_mem = 0; //, max_mem_available = 0;
		cudaDeviceProp properties;
		int sel_device = 0;
		if (opt->cuda_device == -1)
		{
			sel_device = detect_cuda_device();
			if(sel_device >= 0)
			{
				cudaSetDevice(sel_device);
				cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			}
			else
			{
				fprintf(stderr,"[aln_core] Cannot find a suitable CUDA device! aborting!\n");
				return;
			}
		}
		else if (opt->cuda_device >= 0)
		{
			 sel_device = opt->cuda_device;
			 cudaGetDeviceProperties(&properties, sel_device);
			 cudaMemGetInfo(&mem_available, &total_mem);
		     fprintf(stderr, "[aln_core] Using specified CUDA device %d, memory available %d MB.\n", sel_device, int(mem_available>>20));
		     cudaSetDevice(sel_device);
		     cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy bwt occurrences array to from HDD to CPU then to GPU
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		bwtint_t forward_seq_len;
		bwtint_t backward_seq_len;
		cudaMemGetInfo(&mem_available, &total_mem);
		bwt_read_size = copy_bwts_to_cuda_memory(prefix, &global_bwt, &global_rbwt, mem_available>>20, &forward_seq_len, &backward_seq_len)>>20;

		// copy_bwt_to_cuda_memory
		// returns 0 if error occurs
		// mem_available in MiB not in bytes

		if (!bwt_read_size) return; //break

		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		fprintf(stderr, "[aln_core] Finished loading reference sequence assembly, %u MB in %0.2fs (%0.2f MB/s).\n", (unsigned int)bwt_read_size, time_used, ((unsigned int)bwt_read_size)/time_used );

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// allocate GPU working memory
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//Set memory buffer according to memory available

		cudaMemGetInfo(&mem_available, &total_mem);

#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] mem left: %d\n", int(mem_available>>20));
#endif


		//stop if there isn't enough memory available

		int buffer = SEQUENCE_TABLE_SIZE_EXPONENTIAL;
		if ((mem_available>>20) < CUDA_TESLA)
		{
			buffer = buffer - 2; //this will half the memory usage by half to 675MB
			if(mem_available>>20 < (CUDA_TESLA >> 1))
			{
				fprintf(stderr,"[aln_core] Not enough memory to perform alignment (min: %d).\n", CUDA_TESLA >> 1);
				return;
			}
		}
		else
		{
			fprintf(stderr,"[aln_core] Sweet! Running with an enlarged buffer for the Tesla/Quadro series.\n");
		}


		gettimeofday (&start, NULL);
		//allocate global_sequences memory in device
		cudaMalloc((void**)&global_sequences, (1ul<<(buffer))*sizeof(unsigned char));
		main_sequences = (unsigned char *)malloc((1ul<<(buffer))*sizeof(unsigned char));
		//allocate global_sequences_index memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		cudaMalloc((void**)&global_sequences_index, (1ul<<(buffer-3))*sizeof(uint2));
		main_sequences_index = (uint2*)malloc((1ul<<(buffer-3))*sizeof(uint2));
		//allocate and copy options (opt) to device constant memory
		cudaMalloc((void**)&options, sizeof(gap_opt_t));
		cudaMemcpy ( options, opt, sizeof(gap_opt_t), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol ( options_cuda, opt, sizeof(gap_opt_t), 0, cudaMemcpyHostToDevice);
		//allocate alignment stores for host and device
		#if OUTPUT_ALIGNMENTS == 1
		//allocate alignment store memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		global_alignment_store_host = (alignment_store_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_store_t));
		global_alignment_store_host_final = (alignment_store_t*)malloc((1ul<<(buffer-3))*sizeof(alignment_store_t));

		cudaMalloc((void**)&global_alignment_store_device, (1ul<<(buffer-3))*sizeof(alignment_store_t));
		#endif // OUTPUT_ALIGNMENTS == 1
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;

#if DEBUG_LEVEL > 0
		fprintf(stderr,"[aln_debug] Finished allocating CUDA device memory, it took %0.2fs.\n\n", time_used );
		fprintf(stderr,"[aln_debug] mem left: %d\n", int(mem_available>>20));
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Core loop (this loads sequences to host memory, transfers to cuda device and aligns via cuda in CUDA blocks)
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		int loopcount = 0;
		//unsigned int cur_sequence_id = 0; //unique sequence identifier

		// determine block size according to the compute capability
		int blocksize;
		cudaDeviceProp selected_properties;
		cudaGetDeviceProperties(&selected_properties, sel_device);
		if ((int) selected_properties.major > 1) {
			blocksize = 64;
		} else {
			blocksize = 320;
		}

		while ( ( no_of_sequences = copy_sequences_to_cuda_memory(ks, global_sequences_index, main_sequences_index, global_sequences, main_sequences, &read_size, max_sequence_length, opt->mid, buffer) ) > 0 )
		{


			#define GRID_UNIT 32
			int gridsize = GRID_UNIT * (1 + int (((no_of_sequences/blocksize) + ((no_of_sequences%blocksize)!=0))/GRID_UNIT));
			dim3 dimGrid(gridsize);
			dim3 dimBlock(blocksize);

			if(opt->seed_len > (read_size/no_of_sequences))
			{
				fprintf(stderr,"[aln_core] Warning! Specified seed length [%d] exceeds average read length, setting seed length to %d bp.\n", opt->seed_len, int (read_size/no_of_sequences));
				opt->seed_len = read_size/no_of_sequences;
			}

			const int pass_length = (opt->seed_len > PASS_LENGTH)? opt->seed_len: PASS_LENGTH;
			const int split_engage = pass_length + 6;


			// which kernel are we running?
			char split_kernel = (read_size/no_of_sequences >= split_engage);
#if DEBUG_LEVEL > 0
			fprintf(stderr,"[aln_debug] pass length %d, split engage %d.\n", pass_length, split_engage);
#endif

			if (!loopcount) fprintf(stderr, "[aln_core] Now aligning sequence reads to reference assembly, please wait..\n");

			if (!loopcount)	{
#if DEBUG_LEVEL > 1
				fprintf(stderr, "[aln_debug] Average read size: %dbp\n", read_size/no_of_sequences);

				if (split_kernel)
					fprintf(stderr, "[aln_debug] Using split kernel\n");
				else
					fprintf(stderr, "[aln_debug] Using normal kernel\n");
#endif

				fprintf(stderr,"[aln_core] Using SIMT with grid size: %u, block size: %d.\n[aln_core] ", gridsize,blocksize) ;
			}
#if DEBUG_LEVEL > 0
			fprintf(stderr,"\n[aln_debug] Processing %d sequences.", no_of_sequences);
#endif



			// zero out the final alignment store
			memset(global_alignment_store_host_final, 0, (1ul<<(buffer-3))*sizeof(alignment_store_t));


			// create host memory store which persists between kernel calls, on the stack
			main_alignment_store_host_t  main_store;
			memset(main_store.score_align, 0, MAX_SCORE*sizeof(align_store_lst *));

			int run_no_sequences = no_of_sequences;



			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);

			//fprintf(stderr,"time used: %f\n", time_used);

			total_time_used+=time_used;

			// initialise the alignment stores
			memset(global_alignment_store_host, 0, (1ul<<(buffer-3))*sizeof(alignment_store_t));

			for (int i = 0; i < no_of_sequences; i++)
			{

				alignment_store_t* tmp = global_alignment_store_host + i;

				// store the basic info to filter alignments into initialisation file
				tmp->init.lim_k = 0;
				tmp->init.lim_l = forward_seq_len;
				tmp->init.sequence_type = 0;
				tmp->start_pos = 0; //first part
				tmp->sequence_id = i; //cur_sequence_id; cur_sequence_id++;
				//if (!split_kernel) tmp->finished = 1;//mark as finished for normal kernel
			}

			// copy the initialised alignment store to the device
			cudaMemcpy (global_alignment_store_device,global_alignment_store_host, no_of_sequences*sizeof(alignment_store_t), cudaMemcpyHostToDevice);


			///////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Core match function per sequence readings
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////

			gettimeofday (&start, NULL);

#if DEBUG_LEVEL > 3
			printf("cuda opt:%d\n", cuda_opt);
#endif

			//fprintf(stderr,"[aln_debug] kernels run \n", time_used);

			if (split_kernel) {
				cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_store_device, cuda_opt);
			} else {
				cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_store_device, cuda_opt);
			}
			//fprintf(stderr,"[aln_debug] kernels return \n", time_used);

			// Did we get an error running the code? Abort if yes.

			cudaError_t cuda_err = cudaGetLastError();

			if(int(cuda_err))
			{
				fprintf(stderr, "\n[aln_core] CUDA ERROR(s) reported! Last CUDA error message: %s\n[aln_core] Abort!\n", cudaGetErrorString(cuda_err));
				return;
			}


			//Check time
			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			//fprintf(stderr, "Finished!  Time taken: %0.2fs  %d sequences (%d bp) analyzed.\n[aln_core] (%.2f sequences/sec, %.2f bp/sec, avg %0.2f bp/sequence)\n[aln_core]", time_used, no_of_sequences, read_size, no_of_sequences/time_used, read_size/time_used, (float)read_size/no_of_sequences ) ;
			fprintf(stderr, ".");
			// query device for error


			#if OUTPUT_ALIGNMENTS == 1
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// retrieve alignment information from CUDA device to host
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////

			gettimeofday (&start, NULL);

			char cont = 2;
			do
			{
#if DEBUG_LEVEL > 0
				fprintf(stderr,"x");
#else
				fprintf(stderr,".");
#endif
				cudaMemcpy (global_alignment_store_host, global_alignment_store_device, no_of_sequences*sizeof(alignment_store_t), cudaMemcpyDeviceToHost);

				// go through the aligned sequeces and decide which ones are finished and which are not
				int aligned=0;
				int alignments = 0;
				for (int i = 0; i < no_of_sequences; i++)
				{
					alignment_store_t* tmp = global_alignment_store_host + i;


					if (tmp->no_of_alignments > 0)
					{
						aligned += 1;
						alignments += tmp->no_of_alignments;
						//int seq_id = tmp->sequence_id;

						alignment_store_t* final = global_alignment_store_host_final + tmp->sequence_id;


						if (tmp->finished == 1 && final->no_of_alignments == 0) {
						// TODO debug seeding only
						//if (true) {
							memcpy(final, tmp, sizeof(alignment_store_t)); //simply copy the alignment
#if DEBUG_LEVEL > 3
								printf("stored finished alignment for seq: %d\n", tmp->sequence_id);
#endif
						} else {
							// more processing needed, append if finished or enqueue otherwise
							for (int j = 0; j < tmp->no_of_alignments && j < MAX_NO_OF_ALIGNMENTS; j++)
							{
								if (tmp->finished == 1) {
									// append alignments to an existing entry
									int cur_no_aln = final->no_of_alignments;

									if (cur_no_aln + 1 < MAX_NO_OF_ALIGNMENTS) {
										final->alignment_info[cur_no_aln] = tmp->alignment_info[j];
										final->no_of_alignments = cur_no_aln + 1;
									} else {
										break;
									}

	#if DEBUG_LEVEL > 3
									printf("stored finished alignment for seq: %d\n", tmp->sequence_id);
	#endif
								} else {
	#if DEBUG_LEVEL > 3
									printf("continue with unfinished seq: %d\n", tmp->sequence_id);
	#endif
									// otherwise add them to another queue for processing
									int score = tmp->alignment_info[j].score;
									align_store_lst *cur_top = main_store.score_align[score];
									align_store_lst *new_top = (align_store_lst*) malloc( sizeof(align_store_lst) );

									new_top->val = tmp->alignment_info[j];
									new_top->sequence_id = tmp->sequence_id;
									new_top->next = cur_top;
									new_top->start_pos = tmp->start_pos;

									main_store.score_align[score] = new_top;
								}
							}
						}
					}
				}

				#if DEBUG_LEVEL > 0

				fprintf(stderr, "[aln_debug] seq. through: %i \n", aligned);
				fprintf(stderr, "[aln_debug] total alignments: %i \n", alignments);

				#endif

				//print out current new host alignment store
#if DEBUG_LEVEL > 3
				for (int j=0; j<MAX_SCORE; j++) {
						align_store_lst * cur_el = main_store.score_align[j];

						if (cur_el) {
							printf("Alignments with score: %d \n", j);
						}

						while(cur_el) {
							bwt_aln1_t alignment = cur_el->val;
							int cur_len = main_sequences_index[cur_el->sequence_id].y;
							//print some info
							printf("Sequence: %d,  a:%d, k: %d, l: %d, mm: %d, gape: %d, gapo: %d, length: %d, processed: %d\n",cur_el->sequence_id, alignment.a, alignment.k, alignment.l, alignment.n_mm, alignment.n_gape, alignment.n_gapo, cur_len, cur_el->start_pos);

							cur_el = cur_el->next;
						}


				}
				printf("\n");
#endif




				int max_process = (1ul<<(buffer-3)); //taken from the line allocating the memory, maximum we can do in a single run

				int last_index = -1;


				//remove items from the list and free memory accordingly
				for(int i=0; i<MAX_SCORE && max_process > last_index+1; i++) {
					align_store_lst * cur_el = main_store.score_align[i];
					align_store_lst * tmp;

					while(cur_el  && max_process > last_index+1) {
						bwt_aln1_t alignment = cur_el->val;


						// add alignment to the new store
						last_index++;
						alignment_store_t* store_entry = global_alignment_store_host + (last_index);

						// increment start_pos
						store_entry->start_pos = cur_el->start_pos + pass_length;

						store_entry->sequence_id = cur_el->sequence_id;
						store_entry->init.best_cnt = alignment.best_cnt;
						store_entry->init.best_diff = alignment.best_diff;
						store_entry->init.cur_n_gape = alignment.n_gape;
						store_entry->init.cur_n_gapo = alignment.n_gapo;
						store_entry->init.cur_n_mm = alignment.n_mm;
						store_entry->init.lim_k = alignment.k;
						store_entry->init.lim_l = alignment.l;
						store_entry->init.sequence_type = alignment.a;
						store_entry->no_of_alignments = 0; //change to 1 to see the prev. alignment

						tmp = cur_el;
						cur_el = cur_el->next;

						// update the main store to point to the new element
						main_store.score_align[i] = cur_el;

						free(tmp);
					}

				}

				no_of_sequences = last_index + 1;


				if (no_of_sequences > 0) {

#if DEBUG_LEVEL > 3
					printf("aligning %d sequences\n", no_of_sequences);
#endif

					// how many blocks in the current run
					gridsize = GRID_UNIT * (1 + int (((no_of_sequences/blocksize) + ((no_of_sequences%blocksize)!=0))/GRID_UNIT));
					dimGrid = gridsize;

					// transfer the data to the card again
					cudaMemcpy (global_alignment_store_device,global_alignment_store_host, no_of_sequences*sizeof(alignment_store_t), cudaMemcpyHostToDevice);

					//run kernel again
					cuda_split_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_store_device, cuda_opt);

				}
				else {
#if DEBUG_LEVEL > 3
					printf("Nothing to align, finished \n");
#endif
					cont = 0;
				}

			} while(cont);
			// end of kernel loop


			#if STDOUT_STRING_RESULT == 1
			for (int i = 0; i < run_no_sequences; i++)
			{
				alignment_store_t* tmp = global_alignment_store_host_final + i;

				if (tmp->no_of_alignments > 0)
				{
				printf("Sequence %d", tmp->sequence_id);
				printf(", no of alignments: %d\n", tmp->no_of_alignments);

					for (int j = 0; j < tmp->no_of_alignments && j < MAX_NO_OF_ALIGNMENTS; j++)
					{
						printf("  Aligned read %d, ",j+1);
						printf("a: %d, ", tmp->alignment_info[j].a);
						printf("n_mm: %d, ", tmp->alignment_info[j].n_mm);
						printf("n_gape: %d, ", tmp->alignment_info[j].n_gape);
						printf("n_gapo: %d, ", tmp->alignment_info[j].n_gapo);
						printf("k: %u, ", tmp->alignment_info[j].k);
						printf("l: %u, ", tmp->alignment_info[j].l);
						printf("score: %u\n", tmp->alignment_info[j].score);
					}
				}


			}
			//}
			#endif // STDOUT_STRING_RESULT == 1

			#if STDOUT_BINARY_RESULT == 1
			for (int  i = 0; i < run_no_sequences; ++i)
			{
				alignment_store_t* tmp = global_alignment_store_host_final + i;
				fwrite(&tmp->no_of_alignments, 4, 1, stdout);

				if (tmp->no_of_alignments)
				{
					bwt_aln1_t * output;
					output = (bwt_aln1_t*)malloc(tmp->no_of_alignments*sizeof(bwt_aln1_t));

					for (int j = 0; j < tmp->no_of_alignments; j++)
					{
						bwt_aln1_t * temp_output = output + j;
						temp_output->a = tmp->alignment_info[j].a;
						temp_output->k = tmp->alignment_info[j].k;
						temp_output->l = tmp->alignment_info[j].l;
						temp_output->n_mm = tmp->alignment_info[j].n_mm;
						temp_output->n_gapo = tmp->alignment_info[j].n_gapo;
						temp_output->n_gape = tmp->alignment_info[j].n_gape;
						temp_output->score = tmp->alignment_info[j].score;
					}

					fwrite(output, sizeof(bwt_aln1_t), tmp->no_of_alignments, stdout);
					free(output);
				}
			}

			#endif // STDOUT_BINARY_RESULT == 1



			#if MYBINARY== 1
			//fprintf(stderr,"[aln_debug] writing custom binary") ;

			for (int  i = 0; i < run_no_sequences; ++i)
			{
				alignment_store_t* tmp = global_alignment_store_host_final + i;
				fwrite(&tmp->no_of_alignments, 4, 1, mybinary);

				if (tmp->no_of_alignments)
				{
					bwt_aln1_t * output;
					output = (bwt_aln1_t*)malloc(tmp->no_of_alignments*sizeof(bwt_aln1_t));

					for (int j = 0; j < tmp->no_of_alignments; j++)
					{
						bwt_aln1_t * temp_output = output + j;
						temp_output->a = tmp->alignment_info[j].a;
						temp_output->k = tmp->alignment_info[j].k;
						temp_output->l = tmp->alignment_info[j].l;
						temp_output->n_mm = tmp->alignment_info[j].n_mm;
						temp_output->n_gapo = tmp->alignment_info[j].n_gapo;
						temp_output->n_gape = tmp->alignment_info[j].n_gape;
						temp_output->score = tmp->alignment_info[j].score;
					}

					fwrite(output, sizeof(bwt_aln1_t), tmp->no_of_alignments, mybinary);
					free(output);
				}
			}

			#endif // MYBINARY == 1

			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_time_used += time_used;
			//fprintf(stderr, "Finished outputting alignment information... %0.2fs.\n\n", time_used);
			fprintf (stderr, ".");
			#endif // OUTPUT_ALIGNMENTS == 1
			total_no_of_base_pair+=read_size;
			total_no_of_sequences+=run_no_sequences;
			gettimeofday (&start, NULL);
			loopcount ++;
		}
#if MYBINARY== 1
		// clean up fp
		fclose(mybinary);
#endif // MYBINARY == 1

		fprintf(stderr, "\n");

		//report if there is any CUDA error
		if(int(cudaGetLastError()))
		{
			fprintf(stderr, "[aln_core] CUDA ERROR(s) reported!\n" );
		}

#if DEBUG_LEVEL > 0
		fprintf(stderr, "[aln_debug] ERROR message: %s\n", cudaGetErrorString( cudaGetLastError() ) );
#endif

		fprintf(stderr, "[aln_core] Finished!\n[aln_core] Total no. of sequences: %u, size in base pair: %u bp, average length %0.2f bp/sequence.\n", (unsigned int)total_no_of_sequences, (unsigned int)total_no_of_base_pair, (float)total_no_of_base_pair/(unsigned int)total_no_of_sequences);
		fprintf(stderr, "[aln_core] Alignment Speed: %0.2f sequences/sec or %0.2f bp/sec.\n", (float)(total_no_of_sequences/total_time_used), (float)(total_no_of_base_pair/total_time_used));
		fprintf(stderr, "[aln_core] Total program time: %0.2fs.\n", (float)total_time_used);

		//Free memory
		cudaFree(global_sequences);
		free(main_sequences);
		cudaFree(global_sequences_index);
		free(main_sequences_index);
		free_bwts_from_cuda_memory(global_bwt,global_rbwt);

		#if OUTPUT_ALIGNMENTS == 1
		cudaFree(global_alignment_store_device);
		free(global_alignment_store_host);
		#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CUDA mode ends here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
	else if (opt->n_threads > 0) //CPU MODE
	{
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CPU mode starts here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		fprintf(stderr,"[aln_core] Running BWA mode with %d threads\n", opt->n_threads);

		if (opt->max_entries < 0 )
		{
			opt->max_entries = 2000000;
		}

		bwt_t * bwt[2];
		bwa_seq_t * seqs;

		total_time_used = 0;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy bwt occurrences array to from HDD to CPU memory
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		// load BWT to main memory
		char *str = (char*)calloc(strlen(prefix) + 10, 1);
		strcpy(str, prefix); strcat(str, ".bwt");  bwt[0] = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".rbwt"); bwt[1] = bwt_restore_bwt(str);
		free(str);
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		fprintf(stderr, "[bwa_aln_core] Finished loading reference sequence in %0.2fs.\n", time_used );


		//main loop
		gettimeofday (&start, NULL);
		while ((seqs = bwa_read_seq(ks, 0x40000, &no_of_sequences, opt->mode & BWA_MODE_COMPREAD, opt->mid)) != 0)
		{
			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_time_used+=time_used;
			//fprintf(stderr, "Finished loading sequence reads to memory, time taken: %0.2fs, %d sequences (%.2f sequences/sec)\n", time_used, no_of_sequences, no_of_sequences/time_used);

			gettimeofday (&start, NULL);
			fprintf(stderr, "[bwa_aln_core] calculate SA coordinate... \n");
			#ifdef HAVE_PTHREAD
			if (opt->n_threads <= 1) { // no multi-threading at all
				bwa_cal_sa_reg_gap(0, bwt, no_of_sequences, seqs, opt);
			} else {
				pthread_t *tid;
				pthread_attr_t attr;
				thread_aux_t *data;
				int j;
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				data = (thread_aux_t*)calloc(opt->n_threads, sizeof(thread_aux_t));
				tid = (pthread_t*)calloc(opt->n_threads, sizeof(pthread_t));
				for (j = 0; j < opt->n_threads; ++j) {
					data[j].tid = j; data[j].bwt[0] = bwt[0]; data[j].bwt[1] = bwt[1];
					data[j].n_seqs = no_of_sequences; data[j].seqs = seqs; data[j].opt = opt;
					pthread_create(&tid[j], &attr, worker, data + j);
				}
				for (j = 0; j < opt->n_threads; ++j) pthread_join(tid[j], 0);
				free(data); free(tid);
			}
			#else
			bwa_cal_sa_reg_gap(0, bwt, no_of_sequences, seqs, opt);
			#endif

			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			//fprintf(stderr, "Finished!  Time taken: %0.2fs  %d sequences analyzed.\n(%.2f sequences/sec)\n\n", time_used, no_of_sequences, no_of_sequences/time_used );
			#if OUTPUT_ALIGNMENTS == 1
			gettimeofday (&start, NULL);
			fprintf(stderr, "[bwa_aln_core] Writing to the disk... ");



			#if STDOUT_BINARY_RESULT == 1
			for (int i = 0; i < no_of_sequences; ++i) {
				bwa_seq_t *p = seqs + i;
				fwrite(&p->n_aln, 4, 1, stdout);
				if (p->n_aln) fwrite(p->aln, sizeof(bwt_aln1_t), p->n_aln, stdout);
			}
			#endif // STDOUT_BINARY_RESULT == 1


			#if MYBINARY == 1
			for (int i = 0; i < no_of_sequences; ++i) {
				bwa_seq_t *p = seqs + i;
				fwrite(&p->n_aln, 4, 1, mybinary);
				if (p->n_aln) fwrite(p->aln, sizeof(bwt_aln1_t), p->n_aln, mybinary);
			}
			#endif // MYBINARY == 1


			#if STDOUT_STRING_RESULT == 1

			for (int i = 0; i < no_of_sequences; ++i)
			{
				bwa_seq_t *p = &seqs[i];
				if (p->n_aln > 0)
				{
							printf("Sequence %d, no of alignments: %d\n", i, p->n_aln);
							for (int j = 0; j < p->n_aln && j < MAX_NO_OF_ALIGNMENTS; j++)
								{
									bwt_aln1_t * temp = p->aln + j;
									printf("  Aligned read %d, ",j+1);
									printf("a: %d, ", temp->a);
									printf("n_mm: %d, ", temp->n_mm);
									printf("n_gape: %d, ", temp->n_gape);
									printf("n_gapo: %d, ", temp->n_gapo);
									printf("k: %u, ", temp->k);
									printf("l: %u,", temp->l);
									printf("score: %u\n", temp->score);
								}

				}
			}
			#endif



			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			fprintf(stderr, "%0.2f sec\n", time_used);
			#endif // OUTPUT_ALIGNMENTS == 1

			gettimeofday (&start, NULL);
			bwa_free_read_seq(no_of_sequences, seqs);
			total_no_of_sequences += no_of_sequences;
			fprintf(stderr, "[bwa_aln_core] %u sequences have been processed.\n", (unsigned int)total_no_of_sequences);
		}
		fprintf(stderr, "[bwa_aln_core] Total no. of sequences: %u \n", (unsigned int)total_no_of_sequences );
		fprintf(stderr, "[bwa_aln_core] Total compute time: %0.2fs, total program time: %0.2fs.\n", total_calculation_time_used, total_time_used);

		// destroy
		bwt_destroy(bwt[0]); bwt_destroy(bwt[1]);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// CPU mode ends here
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
	bwa_seq_close(ks);

	return;
} //bwa_aln_core ends

int bwa_aln(int argc, char *argv[])
{
	int c, opte = -1;
	gap_opt_t *opt;
	unsigned char cuda_opt = 0;
	// 0 = default
	// 1 = forward only
	// 2 = reverse only

	fprintf(stderr, "Barracuda, Version %s\n",PACKAGE_VERSION);
	opt = gap_init_opt();
	while ((c = getopt(argc, argv, "s:a:C:n:o:e:i:d:l:k:cLR:m:t:NM:O:E")) >= 0) {
		switch (c) {
		case 'a': opt->max_aln = atoi(optarg); break;
		case 's': opt->mid = atoi(optarg); break;
		case 'C': opt->cuda_device = atoi(optarg); break;
		case 'n':
			if (strstr(optarg, ".")) opt->fnr = atof(optarg), opt->max_diff = -1;
			else opt->max_diff = atoi(optarg), opt->fnr = -1.0;
			break;
		case 'o': opt->max_gapo = atoi(optarg); break;
		case 'e': opte = atoi(optarg); break;
		case 'M': opt->s_mm = atoi(optarg); break;
		case 'O': opt->s_gapo = atoi(optarg); break;
		case 'E': opt->s_gape = atoi(optarg); break;
		case 'd': opt->max_del_occ = atoi(optarg); break;
		case 'i': opt->indel_end_skip = atoi(optarg); break;
		case 'l': opt->seed_len = atoi(optarg); break;
		case 'k': opt->max_seed_diff = atoi(optarg); break;
		case 'm': opt->max_entries = atoi(optarg); break;
		case 't': opt->n_threads = atoi(optarg); break;
		case 'L': opt->mode |= BWA_MODE_LOGGAP; break;
		case 'R': opt->max_top2 = atoi(optarg); break;
		case 'c': opt->mode &= ~BWA_MODE_COMPREAD; break;
		case 'N': opt->mode |= BWA_MODE_NONSTOP; opt->max_top2 = 0x7fffffff; break;
//		case 'r': cuda_opt = 2; break;
//		case 'f': cuda_opt = 1; break;
//		case 'x': opt->split_kernel = 1; break;
		default: return 1;
		}
	}
	if (opte > 0) {
		opt->max_gape = opte;
		opt->mode &= ~BWA_MODE_GAPE;
	}


	if (optind + 2 > argc) {
		fprintf(stderr, "\nBWT alignment module using NVIDIA CUDA");
		fprintf(stderr, "\n\n");

		fprintf(stderr, "Usage:   \n");
		fprintf(stderr, "         barracuda aln [options] <reference.fa> <reads.fastq>\n");
		fprintf(stderr, "\n");

		fprintf(stderr, "Options: \n");
		fprintf(stderr, "         -n NUM  max #diff (int) or missing prob under %.2f err rate (float) [default: %.2f]\n",BWA_AVG_ERR, opt->fnr);
		fprintf(stderr, "         -o INT  maximum number or fraction of gap opens [default: %d]\n", opt->max_gapo);
		fprintf(stderr, "         -e INT  maximum number of gap extensions, -1 for disabling long gaps [default: -1]\n");
		fprintf(stderr, "         -i INT  do not put an indel within INT bp towards the ends [default: %d]\n", opt->indel_end_skip);
		fprintf(stderr, "         -d INT  maximum occurrences for extending a long deletion [default: %d]\n", opt->max_del_occ);
		fprintf(stderr, "         -l INT  seed length [default: %d, maximum: %d]\n", opt->seed_len, MAX_SEED_LENGTH);
		fprintf(stderr, "         -k INT  maximum differences in the seed [%d]\n", opt->max_seed_diff);
		fprintf(stderr, "         -m INT  maximum no of loops/entries for matching [default: 150K for CUDA, 2M for BWA]\n");
		fprintf(stderr, "         -M INT  mismatch penalty [default: %d]\n", opt->s_mm);
		fprintf(stderr, "         -O INT  gap open penalty [default: %d]\n", opt->s_gapo);
		fprintf(stderr, "         -E INT  gap extension penalty [default: %d]\n", opt->s_gape);
		fprintf(stderr, "         -R INT  stop searching when >INT equally best hits are found [default: %d]\n", opt->max_top2);
		fprintf(stderr, "         -c      reverse by not complement input sequences for colour space reads\n");
		fprintf(stderr, "         -L      log-scaled gap penalty for long deletions\n");
		fprintf(stderr, "         -N      non-iterative mode: search for all n-difference hits.\n");
		fprintf(stderr, "                 CAUTION this is extremely slow\n");
		fprintf(stderr, "         -s INT  Skip the first INT bases (MID Tag) for alignment.\n");
		fprintf(stderr, "         -t INT  revert to original BWA with INT threads [default: %d]\n", opt->n_threads);
		fprintf(stderr, "                 cannot use with -C, or -a\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "CUDA only options:\n");
		fprintf(stderr, "         -C INT  Specify which CUDA device to use. [default: auto-detect] \n");
		fprintf(stderr, "         -a INT  maximum number of alignments on each strand, max: 20 [default: %d]\n", opt->max_aln);
//		fprintf(stderr, "         -f      single strand mode: align the forward strand only\n"); //function depreciated
//		fprintf(stderr, "         -r      single strand mode: align the reverse complementary strand only\n");//function depreciated
//		fprintf(stderr, "         -x      use split kernels\n");



		fprintf(stderr, "\n");
		return 1;
	}

	if (opt->seed_len > MAX_SEED_LENGTH)
	{
		fprintf(stderr,"[aln_core] Warning, seed length cannot be longer than %d!\n", MAX_SEED_LENGTH);
		return 0;
	}
	if (!opt->n_threads)
	{
		fprintf(stderr, "Error in option (t): No. of threads cannot be 0!\n");
		return 0;
	}
	if (!opt->max_aln)
		{
			fprintf(stderr, "Error in option (a): Max. no. of alignments cannot be 0!\n");
			return 0;
		}

	if (opt->fnr > 0.0) {
		int i, k;
		for (i = 20, k = 0; i <= 150; ++i) {
			int l = bwa_cal_maxdiff(i, BWA_AVG_ERR, opt->fnr);
			if (l != k) fprintf(stderr, "[aln_core] %dbp reads: max_diff = %d\n", i, l);
			k = l;

		}
	}

	barracuda_bwa_aln_core(argv[optind], argv[optind+1], opt, cuda_opt);

	free(opt);
	return 0;
}
//////////////////////////////////////////
// End of ALN_CORE
//////////////////////////////////////////

//////////////////////////////////////////
// CUDA detection code
//////////////////////////////////////////

void bwa_deviceQuery()
// Detect CUDA devices available on the machine, for quick CUDA test and for multi-se/multi-pe shell scripts
{
	int device, num_devices;
	cudaGetDeviceCount(&num_devices);

	if (num_devices)
		{
			  //fprintf(stderr,"[deviceQuery] Querying CUDA devices:\n");
			  for (device = 0; device < num_devices; device++)
			  {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  fprintf(stdout, "%d ", device);
					  fprintf(stdout, "%d %d%d\n", int(properties.totalGlobalMem>>20), int(properties.major),  int(properties.minor));

			  }
			  //fprintf(stderr,"[total] %d\n", device);
		}
	return;
}

int detect_cuda_device()
// Detect CUDA devices available on the machine, used in aln_core and samse_core
{
	int num_devices, device = 0;
	size_t mem_available = 0, total_mem = 0, max_mem_available = 0;
	cudaGetDeviceCount(&num_devices);
	cudaDeviceProp properties;
	int sel_device = -1;

	if (num_devices >= 1)
	{
	     fprintf(stderr, "[detect_cuda_device] Querying CUDA devices:\n");
		 int max_cuda_cores = 0, max_device = 0;
		 for (device = 0; device < num_devices; device++)
		 {
			  cudaGetDeviceProperties(&properties, device);
			  cudaMemGetInfo(&mem_available, &total_mem);
			  fprintf(stderr, "[detect_cuda_device]   Device %d ", device);
			  for (int i = 0; i < 256; i++)
			  {
				  fprintf(stderr,"%c", properties.name[i]);
			  }
				  int cuda_cores = properties.multiProcessorCount<<((1<<properties.major)+1);
				  //calculated by multiprocessors * 8 for 1.x and multiprocessors * 32 for 2.x
				  //determine amount of memory available
			  fprintf(stderr,", CUDA cores %d, memory available %d MB, compute capability %d.%d.\n", int(cuda_cores), int(mem_available>>20), int(properties.major),  int(properties.minor));
			  if (max_cuda_cores <= cuda_cores) //choose the one with highest number of processors
			  {
					  max_cuda_cores = cuda_cores;
					  if (max_mem_available < mem_available) //choose the one with max memory
					  {
						      max_mem_available = mem_available;
							  max_device = device;
					  }
			  }
 		 }
		 if (max_mem_available>>20 >= MIN_MEM_REQUIREMENT)
		 {
			 sel_device = max_device;
			 fprintf(stderr, "[detect_cuda_device] Using CUDA device %d, memory available %d MB.\n", max_device, int(max_mem_available>>20));
			 }
		 else
		 {
			 fprintf(stderr,"[detect_cuda_device] Cannot find a suitable CUDA device with > %d MB of memory available! aborting!\n", MIN_MEM_REQUIREMENT);
			 return -1;
		 }
	}
	else
	{
		 fprintf(stderr,"[detect_cuda_device] No CUDA device found! aborting!\n");
		 return -1;
	}
	return sel_device;
}

//////////////////////////////////////////
// End CUDA detection code
//////////////////////////////////////////

//////////////////////////////////////////
// Below is code for BarraCUDA CUDA SAMSE core
//////////////////////////////////////////
#if CUDA_SAMSE == 1
//CUDA global variables
__device__ __constant__ bwt_t bwt_rbwt_cuda[2];
__device__ __constant__ uint32_t bwt_rbwt_offset[2];
__device__ __constant__ bwtint_t bwt_rbwt_sa_offset[2];
__device__ __constant__ bwtint_t bwt_rbwt_n_sa[2];

const static int BLOCK_SIZE2 = 128;

static int N_MP;

static bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_ho;
static bwa_maxdiff_mapQ_t *seqs_maxdiff_mapQ_de;
static bwtint_t *seqs_sa_ho;
static bwtint_t *seqs_sa_de;
static uint8_t *seqs_mapQ_ho;
static uint8_t *seqs_mapQ_de;
static bwtint_t *seqs_pos_ho;
static bwtint_t *seqs_pos_de;
static uint32_t *bwt_rbwt_de = 0;
static bwtint_t *bwt_rbwt_sa_de = 0;
static int *g_log_n_de;
bwtint_t bwt_sa_intv = 0, rbwt_sa_intv = 0;

texture<uint,2,cudaReadModeElementType> bwt_rbwt_occ_array1;
texture<uint4,2,cudaReadModeElementType> bwt_rbwt_occ_array4;
texture<bwtint_t, 1, cudaReadModeElementType> sa_tex;
texture<bwtint_t, 1, cudaReadModeElementType> bwt_rbwt_sa_tex;
texture<int, 1, cudaReadModeElementType> g_log_n_tex;

//Functions for OCC calculation using 2D texture
#define BWT_RBWT_OCC1(coord_1D) (tex2D(bwt_rbwt_occ_array1,coord_1D & 0xFFFF,coord_1D >> 16))
#define BWT_RBWT_OCC4(coord_1D) (tex2D(bwt_rbwt_occ_array4,(coord_1D & 0xFFFF)>>2,coord_1D >> 16))

unsigned long long copy_bwts_to_cuda_memory_for_samcores(const char * prefix)
// bwt occurrence array to global and bind to texture, bwt structure to constant memory. bwt and
// rbwt are in a joint variable in a 2D texture. [0] = rbwt, [1] = bwt
{
	bwt_t * bwt_src;
	char str[100];
	unsigned long long size_read = 0;


    cudaChannelFormatDesc channelDesc1 = cudaCreateChannelDesc<uint>();
    cudaChannelFormatDesc channelDesc4 = cudaCreateChannelDesc<uint4>();
    size_t pitch_mem_size;
    int n_col = 0x10000; // 2^16, 65536 elements in a column.
    int n_row_per_strand;
    int n_bp_per_strand;// = bwt_src->bwt_size;
    int mod;
    int extra_row;
    uint32_t offset;
    int n_elem_next;
    uint32_t offset_ho[2];
    bwt_t bwt_ho[2];
    size_t mem_available, total_mem;
	cudaMemGetInfo(&mem_available, &total_mem);

	bwtint_t n_sa[2];
    bwtint_t bwt_rbwt_sa_offset_ho[2];
    bwtint_t sa_offset;


#if DEBUG_LEVEL  > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] mem available %d MB\n", int(mem_available>>20));
#endif
	{
	    //Reversed BWT
		//Load bwt occurrence array from from disk
		//TODO: BWT already loaded before remove the line below!
		strcpy(str, prefix); strcat(str, ".rbwt");  bwt_src = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".rsa"); bwt_restore_sa(str, bwt_src);
		n_sa[0] = bwt_src->n_sa;
		sa_offset = n_sa[0];
		rbwt_sa_intv = bwt_src->sa_intv; //store in global variable rbwt_sa_intv
		size_read += bwt_src->bwt_size*sizeof(uint32_t);

		n_bp_per_strand = bwt_src->bwt_size;
		n_row_per_strand = n_bp_per_strand >> 16;
		mod = n_bp_per_strand - (n_row_per_strand << 16);
		extra_row = n_row_per_strand == 0 ? 0 : 1;
		offset = (n_row_per_strand+extra_row)*n_col;

		//fprintf(stderr,"1.0 %u %i %i %i %i %i\n", bwt_src->bwt_size, n_row_per_strand,n_col,mod,extra_row,offset);

		//Allocate memory for bwt,rbwt
		cudaMallocPitch((void**) &bwt_rbwt_de, &pitch_mem_size, sizeof(uint32_t)*n_col, 2*(n_row_per_strand+extra_row));
		report_cuda_error_GPU("cudaMallocPitch bwt_rbwt_de,\n");

		// Copy full rows.
	    if (n_row_per_strand > 0)
	    {
	        cudaMemcpy2D(bwt_rbwt_de,pitch_mem_size,bwt_src->bwt,pitch_mem_size,pitch_mem_size,n_row_per_strand,cudaMemcpyHostToDevice);
	        report_cuda_error_GPU("cudaMemcpy2D bwt_rbwt_de <- bwt_src->bwt\n");

	        n_elem_next = n_row_per_strand*n_col;
	    }

	    // Copy partial row.
	    if (mod > 0)
	    {
	        uint32_t *buf = (uint32_t *) malloc(sizeof(uint32_t)*n_col);

	        for (int i0 = 0; i0 < mod; i0++)
	            buf[i0] = bwt_src->bwt[n_elem_next+i0];

	        cudaMemcpy2D(bwt_rbwt_de+n_elem_next,pitch_mem_size,buf,pitch_mem_size,pitch_mem_size,1,cudaMemcpyHostToDevice);
	        report_cuda_error_GPU("cudaMemcpy2D bwt_rbwt_de+n_elem_next <- buf\n");

	        free(buf);
	    }

        //copy bwt structure data to copy
        bwt_ho[0] = *bwt_src;


        //fprintf (stderr,"1.0 %u %u %i\n", n_sa[0],n_sa[1],offset);

        //Allocate memory for both bwt->sa,rbwt->sa
        cudaMalloc((void**) &bwt_rbwt_sa_de,sizeof(bwtint_t)*n_sa[0]*2);
        report_cuda_error_GPU("cudaMalloc bwt_rbwt_sa_de,\n");

        // Copy rbwt_sa data to device.
        cudaMemcpy(bwt_rbwt_sa_de,bwt_src->sa,sizeof(bwtint_t)*n_sa[0],cudaMemcpyHostToDevice);
        report_cuda_error_GPU("cudaMemcpy bwt_rbwt_sa_de <- rbwt_src->sa\n");

        // Copy offset size to copy.constant memory.
        bwt_rbwt_sa_offset_ho[0] = 0;

	    bwt_destroy(bwt_src);
	}



#if DEBUG_LEVEL  > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] bwt loaded, mem available %d MB\n", int(mem_available>>20));
#endif


    {
        //Forward BWT
        //Load bwt occurrence array from from disk
		//TODO: BWT already loaded before remove the line below!
        strcpy(str, prefix); strcat(str, ".bwt"); bwt_src = bwt_restore_bwt(str);
		strcpy(str, prefix); strcat(str, ".sa"); bwt_restore_sa(str, bwt_src);
		n_sa[1]=bwt_src->n_sa;
		bwt_sa_intv = bwt_src->sa_intv; //store in global variable bwt_sa_intv
        size_read += bwt_src->bwt_size*sizeof(uint32_t);


        // Copy full rows.
        if (n_row_per_strand > 0)
        {
            cudaMemcpy2D(bwt_rbwt_de+offset,pitch_mem_size,bwt_src->bwt,pitch_mem_size,pitch_mem_size,n_row_per_strand,cudaMemcpyHostToDevice);
            report_cuda_error_GPU("cudaMemcpy2D bwt_rbwt_de+offset <- bwt_src->bwt\n");
        }

        // Copy partial row.
        if (mod > 0)
        {
            uint32_t *buf = (uint32_t *) malloc(sizeof(uint32_t)*n_col);

            for (int i0 = 0; i0 < mod; i0++)
                buf[i0] = bwt_src->bwt[n_elem_next+i0];

            cudaMemcpy2D(bwt_rbwt_de+offset+n_elem_next,pitch_mem_size,buf,pitch_mem_size,pitch_mem_size,1,cudaMemcpyHostToDevice);
            report_cuda_error_GPU("cudaMemcpy2D bwt_rbwt_de+offset+n_elem_next <- buf\n");

            free(buf);
        }

        //copy bwt structure data to copy
        bwt_ho[1] = *bwt_src;


        // Copy bwtdata to device.
        cudaMemcpy(bwt_rbwt_sa_de+sa_offset,bwt_src->sa,sizeof(bwtint_t)*n_sa[1],cudaMemcpyHostToDevice);
        report_cuda_error_GPU("cudaMemcpy rbwt_rbwt_sa_de+offset <- bwt_src->sa\n");

        // Copy offset size to copy.
        bwt_rbwt_sa_offset_ho[1] = sa_offset;

        bwt_destroy(bwt_src);
    }

#if DEBUG_LEVEL  > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] rbwt loaded, mem available %d MB\n", int(mem_available>>20));
#endif

    // Copy offset size to constant memory.
    offset_ho[0] = 0;
    offset_ho[1] = offset;
    cudaMemcpyToSymbol(bwt_rbwt_offset,&offset_ho,sizeof(uint32_t)*2,0,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("cudaMemcpyToSymbol bwt_rbwt_offset[1] <- offset\n");

    //copy bwt structure data to constant memory bwt_rbwt_cuda structure
    cudaMemcpyToSymbol(bwt_rbwt_cuda,&bwt_ho,sizeof(bwt_t)*2,0,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("cudaMemcpyToSymbol bwt_rbwt_cuda <- bwt_src\n");

    //fprintf(stderr,"r/bwt: %u %u\n",rbwt->n_sa,bwt->n_sa);
    //fprintf(stderr,"r/bwt_ho: %u %u\n",bwt_ho[1].n_sa,bwt_ho[0].n_sa);

    // Bind memory to texture.
    cudaBindTexture2D(0,&bwt_rbwt_occ_array1,bwt_rbwt_de,&channelDesc1,n_col,2*(n_row_per_strand+extra_row),pitch_mem_size);
    report_cuda_error_GPU("cudaBindTexture2D bwt_rbwt_occ_array1\n");

    cudaBindTexture2D(0,&bwt_rbwt_occ_array4,bwt_rbwt_de,&channelDesc4,(n_col>>2),2*(n_row_per_strand+extra_row),pitch_mem_size);
    report_cuda_error_GPU("cudaBindTexture2D bwt_rbwt_occ_array4\n");

//From sa

    // Copy offset size constant memory.
    cudaMemcpyToSymbol(bwt_rbwt_sa_offset,bwt_rbwt_sa_offset_ho,sizeof(bwtint_t)*2,0,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("cudaMemcpyToSymbol rbwt_rbwt_sa_offset <- 0, bwt_s->n_sa\n");

    // Bind memory to texture.
    cudaBindTexture(0,bwt_rbwt_sa_tex,bwt_rbwt_sa_de,sizeof(bwtint_t)*(n_sa[0])*2);
    report_cuda_error_GPU("cudaBindTexture bwt_rbwt_sa_tex\n");

    // Copy (r)bwt.n_sa to constant memory (because it does not exist in "bwt_rbwt_cuda").
    cudaMemcpyToSymbol(bwt_rbwt_n_sa,&n_sa,sizeof(bwtint_t)*2,0,cudaMemcpyHostToDevice);


	return size_read;
}

void free_bwts_from_cuda_memory_for_samcores()
{
    if ( bwt_rbwt_de != 0 )
    {
        cudaUnbindTexture(bwt_rbwt_occ_array1);
        cudaUnbindTexture(bwt_rbwt_occ_array4);
        cudaFree(bwt_rbwt_de);
    }
}

int prepare_bwa_cal_pac_pos_cuda(
    const char *prefix,
    const int *g_log_n_ho,
    const int g_log_n_len,
    const int n_seqs_max,
    int device)
{
    unsigned long long int bwt_read_size;
    cudaDeviceProp prop;
    //int device = -1; //set default device to -1 unless it is specified by user
    size_t mem_available, total_mem;


	if(device >= 0)
	{
		fprintf(stderr,"[samse_core] Using specified CUDA device %d.\n",device);
	}
    //auto device selection
	if (device < 0)
	{
		device = detect_cuda_device();

		if(device >= 0)
		{
			cudaSetDevice(device);
		}
		else
		{
			fprintf(stderr,"[samse_core]Error! Cannot find a suitable CUDA device, Aborting...\n");
			return -1;
		}

	}else
	{
		cudaSetDevice(device);

	}

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaMemGetInfo(&mem_available, &total_mem);

    ////////////////////////////////////////////////////////////
    // Load BWT to GPU.
    ////////////////////////////////////////////////////////////
    cudaGetDeviceProperties(&prop, device);

    // For timing purpose only
	struct timeval start, end;
	double time_used;


	fprintf(stderr,"[samse_core] Loading BWTs, please wait..");
	// load forward & reverse bwts
	gettimeofday (&start, NULL);

    bwt_read_size = copy_bwts_to_cuda_memory_for_samcores(prefix);
    // copy_bwt_to_cuda_memory
    // returns 0 if error occurs

	gettimeofday (&end, NULL);
	time_used = diff_in_seconds(&end,&start);

	fprintf(stderr, "Done! \n[samse_core] Time used: %0.2fs\n", time_used);


    cudaMemGetInfo(&mem_available, &total_mem);
    if (int(mem_available>>20)<0)
    {
    	fprintf(stderr,"[samse_core]Not enough memory available! Aborting...\n");
    	return -1;
    }

#if DEBUG_LEVEL  > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] finished loading BWTs, mem available %d MB\n", int(mem_available>>20));
#endif

    if (bwt_read_size == 0)
    {
    	report_cuda_error_CPU("[samse_core] Error copying BWT and RBWT to device.\n");
    	return -1;
    }

    ////////////////////////////////////////////////////////////
    // Copy input data in "g_log_n" to device, and bind texture of "g_log_n_de".
    ////////////////////////////////////////////////////////////
    copy_g_log_n_cuda(g_log_n_ho,g_log_n_len);

#if DEBUG_LEVEL  > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] finished loading g_log_n, mem available %d MB\n", int(mem_available>>20));
#endif


	////////////////////////////////////////////////////////////
    // Prepare buffers for sequences.
    ////////////////////////////////////////////////////////////
    prepare_bwa_cal_pac_pos_seqs_cuda(n_seqs_max);

#if DEBUG_LEVEL > 1
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] finished allocating buffers, mem available %d MB\n", int(mem_available>>20));
#endif

    cudaMemGetInfo(&mem_available, &total_mem);
    if (int(mem_available>>20) < 50) // in MiB
    {
    	fprintf(stderr,"[samse_core]Not enough memory available! Aborting...\n");
    	return -1;
    }

	return device;
}

void copy_g_log_n_cuda(const int *g_log_n_ho,const int g_log_n_len)
{
    // Reserve memory.
    cudaMalloc((void**)&g_log_n_de,sizeof(int)*g_log_n_len);
    report_cuda_error_GPU("Error reserving memory for \"g_log_n_de\".\n");

    // Copy data from host to device.
    cudaMemcpy(g_log_n_de,g_log_n_ho,sizeof(int)*g_log_n_len,cudaMemcpyHostToDevice);
    report_cuda_error_GPU("Error copying to \"g_log_n_de\".\n");

    // Bind texture.
    cudaBindTexture(0,g_log_n_tex,g_log_n_de,sizeof(int)*g_log_n_len);
    report_cuda_error_GPU("Error binding texture to \"g_log_n_tex\".\n");

    return;
}

void prepare_bwa_cal_pac_pos_seqs_cuda(int n_seqs_max)
{

	////////////////////////////////////////////////////////////
	// Allocate memory to input variables "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho" on host.
	////////////////////////////////////////////////////////////
	seqs_maxdiff_mapQ_ho = (bwa_maxdiff_mapQ_t *) malloc(sizeof(bwa_maxdiff_mapQ_t)*n_seqs_max);
	if (seqs_maxdiff_mapQ_ho == NULL) report_cuda_error_CPU("[samse_core]Error reserving memory for \"seqs_maxdiff_mapq_ho\".\n");
	seqs_sa_ho = (bwtint_t *) malloc(sizeof(bwtint_t)*n_seqs_max);
	if (seqs_sa_ho == NULL) report_cuda_error_CPU("[samse_core]Error reserving memory for \"seqs_sa_ho\".\n");

    ////////////////////////////////////////////////////////////
	// Reserve memory for "seqs_maxdiff_mapQ_de" and "seqs_sa_de" on device.
	////////////////////////////////////////////////////////////
	// Reserve memory.
	cudaMalloc(&seqs_maxdiff_mapQ_de,sizeof(bwa_maxdiff_mapQ_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core]Error reserving memory for \"seqs_maxdiff_mapQ_de\".\n");

	// Reserve memory.
	cudaMalloc(&seqs_sa_de,sizeof(bwtint_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core]Error reserving memory for \"seqs_sa_de\".\n");

	////////////////////////////////////////////////////////////
	// Reserve memory for output data variables in "seqs_mapQ_ho" and "seqs_pos_ho" on host.
	////////////////////////////////////////////////////////////
	//cudaGetLastError();
	// Reserve memory for return data "mapQ_ho" and "pos_ho" on the host.
	seqs_mapQ_ho = (uint8_t *) malloc(sizeof(uint8_t)*n_seqs_max);
	if (seqs_mapQ_ho == NULL) report_cuda_error_CPU("[samse_core]Error reserving memory for \"seqs_mapQ_ho\".\n");
	seqs_pos_ho = (bwtint_t *) malloc(sizeof(bwtint_t)*n_seqs_max);
	if (seqs_pos_ho == NULL) report_cuda_error_CPU("[samse_core]Error reserving memory for \"seqs_pos_ho\".\n");

	////////////////////////////////////////////////////////////
	// Reserve memory for output data variables "seqs_mapQ_de" and "seqs_pos_de" on device.
	////////////////////////////////////////////////////////////
	cudaMalloc(&seqs_mapQ_de,sizeof(uint8_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core]Error reserving memory for \"seqs_mapQ_de\".\n");
	cudaMalloc(&seqs_pos_de,sizeof(bwtint_t)*n_seqs_max);
	report_cuda_error_GPU("[samse_core]Error reserving memory for \"seqs_pos_de\".\n");

#if DEBUG_LEVEL  > 1
	size_t mem_available, total_mem;
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] finished loading sequences, mem available %d MB\n", int(mem_available>>20));
#endif

}

void free_bwa_cal_pac_pos_cuda()
{
    free_bwa_cal_pac_pos_bwt_rbwt_cuda();
    free_bwa_cal_pac_pos_seqs_cuda();
    return;
}

void free_bwa_cal_pac_pos_bwt_rbwt_cuda()
{

    ////////////////////////////////////////////////////////////
    // Clean up data.
    ////////////////////////////////////////////////////////////
    // Erase BWT on GPU device.
    free_bwts_from_cuda_memory_for_samcores();

    // Delete memory used.
    cudaFree(bwt_rbwt_sa_de);
    cudaFree(g_log_n_de);

    // Unbind texture to reads.
    //cudaUnbindTexture(sa_tex);

    // Unbind textures to BWT and RBWT.
    cudaUnbindTexture(bwt_rbwt_sa_tex);

    // Unbind texture to "g_log_n_tex".
    cudaUnbindTexture(g_log_n_tex);

    // Free constant memory.

}

void free_bwa_cal_pac_pos_seqs_cuda()
{
	////////////////////////////////////////////////////////////
	// Clean up data.
	////////////////////////////////////////////////////////////
	free(seqs_maxdiff_mapQ_ho);
	cudaFree(seqs_maxdiff_mapQ_de);
	free(seqs_sa_ho);
	cudaFree(seqs_sa_de);
	free(seqs_pos_ho);
	cudaFree(seqs_pos_de);
	free(seqs_mapQ_ho);
	cudaFree(seqs_mapQ_de);
}

// This function is meant to be a GPU implementation of bwa_cal_pac_pos(). Currently,
// only the forward strand is being tested for bwt_sa(). After that, test the reverse
// strand. Lastly, make GPU implementations of bwa_cal_maxdiff() and bwa_approx_mapQ().
void launch_bwa_cal_pac_pos_cuda(
	const char *prefix,
	int n_seqs,
	bwa_seq_t *seqs,
	int max_mm,
	float fnr,
	//bwt_t *bwt,
	//bwt_t *rbwt,
	int device)
{

#if DEBUG_LEVEL > 1
	fprintf(stderr, "bwt_sa_intv: %u %i\n",bwt_sa_intv, int(sizeof(bwt_sa_intv)));
	fprintf(stderr, "rbwt_sa_intv: %u %i\n",rbwt_sa_intv, int(sizeof(rbwt_sa_intv)));
#endif

	////////////////////////////////////////////////////////////
	// Declare and initiate variables.
	////////////////////////////////////////////////////////////

	cudaDeviceProp prop;
	int n_block;
	int n_seq_per_block;
	int block_mod;
	size_t mem_available, total_mem;

	// Obtain information on CUDA devices.
	cudaGetDeviceProperties(&prop, device);
	cudaMemGetInfo(&mem_available, &total_mem);

#if DEBUG_LEVEL  > 1
	fprintf(stderr,"\n[SAMSE debug] selected CUDA device: %d\n", device);
	cudaMemGetInfo(&mem_available, &total_mem);
	fprintf(stderr,"[SAMSE debug] finished allocating buffers, mem available %d MB\n", int(mem_available>>20));
#endif
	////////////////////////////////////////////////////////////
	// Allocate memory and copy reads in "seqs" to "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho".
	////////////////////////////////////////////////////////////
	for (int i = 0; i < n_seqs; i++)
	{
		seqs_maxdiff_mapQ_ho[i].len = seqs[i].len;
		//seqs_maxdiff_mapQ_ho[i].strand_type = ((seqs[i].strand<<2) | seqs[i].type);
		seqs_maxdiff_mapQ_ho[i].strand = seqs[i].strand;
		seqs_maxdiff_mapQ_ho[i].type = seqs[i].type;
		seqs_maxdiff_mapQ_ho[i].n_mm = seqs[i].n_mm;
		seqs_maxdiff_mapQ_ho[i].c1 = seqs[i].c1;
		seqs_maxdiff_mapQ_ho[i].c2 = seqs[i].c2;
		seqs_sa_ho[i] = seqs[i].sa;
	}


	////////////////////////////////////////////////////////////
	// Copy input data in "seqs_maxdiff_mapQ_ho" and "seqs_sa_ho" to device.
	////////////////////////////////////////////////////////////
	// Copy data from host to device.
	cudaMemcpy(seqs_maxdiff_mapQ_de,seqs_maxdiff_mapQ_ho,sizeof(bwa_maxdiff_mapQ_t)*n_seqs,cudaMemcpyHostToDevice);
	report_cuda_error_GPU("[samse_core]Error copying to \"seqs_maxdiff_mapQ_de\".\n");

	// Copy data from host to device.
	cudaMemcpy(seqs_sa_de,seqs_sa_ho,sizeof(bwtint_t)*n_seqs,cudaMemcpyHostToDevice);
	report_cuda_error_GPU("[samse_core]Error copying to \"seqs_sa_de\".\n");

	////////////////////////////////////////////////////////////
	// Process bwa_cal_pac_pos_cuda()
	////////////////////////////////////////////////////////////
	// Calculate the no. of blocks and sequences per block.
	calc_n_block1(&N_MP,&n_block,&n_seq_per_block,&block_mod,prop.multiProcessorCount,48,n_seqs);

#if DEBUG_LEVEL > 1
	fprintf(stderr,"[SAMSE debug] N_MP %i n_block %i n_seq_per_block %i block_mod %i\n", N_MP, n_block, n_seq_per_block, block_mod);
	fprintf(stderr,"[SAMSE debug] n_seqs %i\n", n_seqs);
#endif
	// Set block and grid sizes.

	dim3 dimBlock(BLOCK_SIZE2);
	dim3 dimGrid(n_block);

#if DEBUG_LEVEL > 1
	fprintf(stderr,"[SAMSE debug] Using block size: %d, grid size: %d\n", BLOCK_SIZE2,n_block);
#endif

	// Execute bwt_sa function.

	cuda_bwa_cal_pac_pos_parallel1 <<<dimGrid, dimBlock>>>(
		seqs_mapQ_de,
		seqs_pos_de,
		seqs_maxdiff_mapQ_de,
		seqs_sa_de,
		n_seqs,
		n_block,
		n_seq_per_block,
		block_mod,
		max_mm,
		fnr,
		bwt_sa_intv,
		rbwt_sa_intv);

	report_cuda_error_GPU("[samse_core]Error running \"cuda_bwa_cal_pac_pos()\".\n");
	cudaThreadSynchronize();
	report_cuda_error_GPU("[samse_core]After synchronizing after \"cuda_bwa_cal_pac_pos()\".\n");



	////////////////////////////////////////////////////////////
	// Copy data of output data variables in "seqs_mapQ_de" and "seqs_pos_de" to host.
	////////////////////////////////////////////////////////////

	// cudaGetLastError();
	// Return data to host.
	cudaMemcpy(seqs_mapQ_ho, seqs_mapQ_de, sizeof(uint8_t)*n_seqs, cudaMemcpyDeviceToHost);
	report_cuda_error_GPU("[samse_core]Error copying to \"seqs_mapQ_ho\".\n");
	cudaMemcpy(seqs_pos_ho, seqs_pos_de, sizeof(bwtint_t)*n_seqs, cudaMemcpyDeviceToHost);
	report_cuda_error_GPU("[samse_core]Error copying to \"seqs_pos_ho\".\n");

#if DEBUG_LEVEL > 6
	////////////////////////////////////////////////////////////
	// Compare GPU data with CPU data to verify results.
	////////////////////////////////////////////////////////////
	compare_mapQ_pos(seqs,n_seqs);
#endif //DEBUG_LEVEL ==7

	////////////////////////////////////////////////////////////
	// Save output data variables to "seqs".
	////////////////////////////////////////////////////////////
	for (int i = 0; i < n_seqs; i++)
	{
	 	seqs[i].mapQ = seqs_mapQ_ho[i];
		seqs[i].seQ = seqs_mapQ_ho[i];
		seqs[i].pos = seqs_pos_ho[i];
	}
}

#if DEBUG_LEVEL > 6
void compare_mapQ_pos(bwa_seq_t *seqs, int n_seqs)
{
	////////////////////////////////////////////////////////////
	// Compare GPU data with CPU data to verify results.
	////////////////////////////////////////////////////////////

	int n_fwd = 0;
	int n_rvs = 0;
	int n_type[4] = {0, 0, 0, 0};

	for (int i = 0; i < n_seqs; i++)
	{
		bwa_seq_t *s = seqs + i;
		if (s->type == BWA_TYPE_UNIQUE || s->type == BWA_TYPE_REPEAT)
		{
			if (s->strand)
			{
				if (s->mapQ != seqs_mapQ_ho[i]) fprintf(stderr,"mapQ at %i wrong %u %u %u\n", i, s->strand, s->mapQ, seqs_mapQ_ho[i]);
				//else fprintf(stderr,"mapQ at %i OK %u %u %u\n", i, s->strand, s->mapQ, seqs_mapQ_ho[i]);
				if (s->seQ != seqs_mapQ_ho[i]) fprintf(stderr,"seQ at %i wrong %u %i %u\n", i, s->strand, int(s->seQ), seqs_mapQ_ho[i]);
				//if (s->pos != seqs_pos_ho[i]) fprintf(stderr,"pos at %i wrong: %u %u %u %i\n", i, s->strand, s->pos, seqs_pos_ho[i], int(s->pos-seqs_pos_ho[i]));
				//else fprintf(stderr,"pos at %i OK %u %u %u\n", i, s->strand, s->pos, seqs_pos_ho[i]);
				n_fwd++;
			}
			else
			{
                if (s->mapQ != seqs_mapQ_ho[i]) fprintf(stderr,"mapQ at %i wrong %u %u %u\n", i, s->strand, s->mapQ, seqs_mapQ_ho[i]);
                //else fprintf(stderr,"mapQ at %i OK %u %u %u\n", i, s->strand, s->mapQ, seqs_mapQ_ho[i]);
                if (s->seQ != seqs_mapQ_ho[i]) fprintf(stderr,"seQ at %i wrong %u %i %u\n", i, s->strand, int(s->seQ), seqs_mapQ_ho[i]);
                //if (s->pos != seqs_pos_ho[i]) fprintf(stderr,"pos at %i wrong: %u %u %u %i\n", i, s->strand, s->pos, seqs_pos_ho[i], int(s->pos-seqs_pos_ho[i]));
                //else fprintf(stderr,"pos at %i OK %u %u %u\n", i, s->strand, s->pos, seqs_pos_ho[i]);
				n_rvs++;
			}
		}
		n_type[s->type]++;
	}


	fprintf(stderr, "n_seqs: %i n_fwd: %i n_rvs: %i\n",n_seqs,n_fwd,n_rvs);
	fprintf(stderr, "n_type: %i %i %i %i\n",n_type[0],n_type[1],n_type[2],n_type[3]);
}
#endif // DEBUG_LEVEL >6

__device__ inline uint4 BWT_RBWT_OCC4b(uint32_t coord_1D,char strand)
// This function uses joint texture for bwt/rbst
{
    coord_1D = coord_1D << 2;
    coord_1D += bwt_rbwt_offset[strand];
    return BWT_RBWT_OCC4(coord_1D);

    //uint32_t pos_x = coord_1D >> 16;
    //uint32_t pos_y = (coord_1D & 0xFFFF)>>2;
    //return tex2D(bwt_rbwt_occ_array4,pos_x,pos_y);
}

__device__ uint32_t bwt_rbwt_bwt(bwtint_t k,char strand)
// This function uses joint texture for bwt/rbst
{
    //int pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL) / 16;
    uint32_t pos = (k)/OCC_INTERVAL*12 + 4 + ((k)%OCC_INTERVAL >> 4);
    //uint one_integer = tex2D(bwt_rbwt_occ_array,pos>>16,pos & 0xFFFF);
    uint one_integer = BWT_RBWT_OCC1(bwt_rbwt_offset[strand]+pos);

    return one_integer;
}

__device__ ubyte_t bwt_rbwt_B0(bwtint_t k,char strand)
{
	uint32_t tmp = bwt_rbwt_bwt(k,strand)>>((~(k)&0xf)<<1)&3;
	ubyte_t c = ubyte_t(tmp);
	return c;
}

__device__ inline void Conv1DTo2D1(int *coord1_2D,int *coord2_2D,int coord_1D)
{
    *coord1_2D = (0xFFFF & coord_1D);
    *coord2_2D = coord_1D >> 16;
}

__device__ inline void Conv1DTo2D4(int *coord1_2D,int *coord2_2D,int coord_1D)
{
    *coord1_2D = (0xFFFF & coord_1D)>>2;
    *coord2_2D = coord_1D >> 16;
}

__device__
int cuda_bwa_approx_mapQ(const bwa_maxdiff_mapQ_t *p, int mm)
{
    int n, g_log;
    if (p->c1 == 0) return 23;
    if (p->c1 > 1) return 0;
    if (p->n_mm == mm) return 25;
    if (p->c2 == 0) return 37;
    n = (p->c2 >= 255)? 255 : p->c2;
    g_log = tex1Dfetch(g_log_n_tex,n);

    return (23 < g_log)? 0 : 23 - g_log;
}

__device__
void update_indices(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty)
{
    (*n_sa_processed)++;
    (*n_sa_remaining)--;
    (*n_sa_in_buf)--;
    (*n_sa_buf_empty)++;
}

__device__
void update_indices_in_parallel(
    int *n_sa_processed,
    int *n_sa_remaining,
    int *n_sa_in_buf,
    int *n_sa_buf_empty)
{
    atomicAdd(*&n_sa_processed,1);
    atomicSub(*&n_sa_remaining,1);
    atomicSub(*&n_sa_in_buf,1);
    atomicAdd(*&n_sa_buf_empty,1);
}

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
    const int n_sa_total,
    const char strand)
{
    while (*sa_next_no < n_sa_total)
    {
        int read_no_new = atomicAdd(*&sa_next_no,1);

        if (read_no_new < n_sa_total)
        {
            // Get new read from global memory.
            *maxdiff_mapQ_buf = seqs_maxdiff_mapQ_de[offset+read_no_new];
            //sa_buf_arr[tid] = seqs_sa_de[offset+read_no_new];
            // Check whether read can be used.
            if ((*maxdiff_mapQ_buf).strand == strand && ((*maxdiff_mapQ_buf).type == BWA_TYPE_UNIQUE ||
                (*maxdiff_mapQ_buf).type == BWA_TYPE_REPEAT))
            {
                *sa_origin = read_no_new;
                //sa_return[tid] = 0;
                atomicAdd(*&n_sa_in_buf,1);
                atomicSub(*&n_sa_buf_empty,1);
                break;
            }
            else
            {
                atomicAdd(*&n_sa_processed,1);
                atomicSub(*&n_sa_remaining,1);
                // Show that read is not being used.
            }
        }
    }
}

__device__
void sort_reads(
    bwtint_t *sa_buf_arr,
    bwa_maxdiff_mapQ_t *maxdiff_mapQ_buf_arr,
    int16_t *sa_origin,
    bwtint_t *sa_return,
    const int *n_sa_in_buf,
    int *n_sa_in_buf_prev)
{
    int sa_empty_no = *n_sa_in_buf_prev;
    *n_sa_in_buf_prev = *n_sa_in_buf;

    for (int j = 0; j < sa_empty_no; j++)
    {
        if (sa_origin[j] == -1)
        {
            for (int k = sa_empty_no-1; k > j; k--)
            {
                sa_empty_no--;
                if (sa_origin[k] != -1)
                {
                    sa_buf_arr[j] = sa_buf_arr[k];
                    maxdiff_mapQ_buf_arr[j] = maxdiff_mapQ_buf_arr[k];
                    sa_origin[j] = sa_origin[k];
                    sa_return[j] = sa_return[k];
                    sa_origin[k] = -1;
                    break;
                }
            }
        }
    }
}

__device__
uint32_t inline __occ_bwt_rbwt_cuda_aux4(
		uint32_t b, char strand)
// This function uses joint texture for bwt/rbwt
{
    uint32_t tmp = bwt_rbwt_cuda[strand].cnt_table[(b)&0xff];
    tmp += bwt_rbwt_cuda[strand].cnt_table[(b)>>8&0xff];
    tmp += bwt_rbwt_cuda[strand].cnt_table[(b)>>16&0xff];
    tmp += bwt_rbwt_cuda[strand].cnt_table[(b)>>24];
    return tmp;
}

__device__
uint4 bwt_rbwt_cuda_occ4(
		bwtint_t k,char strand)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
// This function uses joint texture for bwt/rbwt
{
    // total number of character c in the up to the interval of k
    uint4 tmp;
    uint4 n = {0,0,0,0};
    unsigned int i = 0;
    unsigned int m = 0;
    // remarks: uint4 in CUDA is 4 x integer ( a.x,a.y,a.z,a.w )
    // uint4 is used because CUDA 1D texture array is limited to width for 2^27
    // to access 2GB of memory, a structure uint4 is needed
    // tmp variable
    unsigned int tmp1,tmp2;//, tmp3;

    if (k == bwt_rbwt_cuda[strand].seq_len)
    {
        n.x = bwt_rbwt_cuda[strand].L2[1]-bwt_rbwt_cuda[strand].L2[0];
        n.y = bwt_rbwt_cuda[strand].L2[2]-bwt_rbwt_cuda[strand].L2[1];
        n.z = bwt_rbwt_cuda[strand].L2[3]-bwt_rbwt_cuda[strand].L2[2];
        n.w = bwt_rbwt_cuda[strand].L2[4]-bwt_rbwt_cuda[strand].L2[3];
        return n;
    }
    if (k == (bwtint_t)(-1)) return n;
    if (k >= bwt_rbwt_cuda[strand].primary) --k; // because $ is not in bwt

    //tmp3 = k>>7;
    //i = tmp3*3;
    i = ((k>>7)*3);

    // count the number of character c within the 128bits interval
    tmp = BWT_RBWT_OCC4b(i+1,strand);

    if (k&0x40)
    {
        m = __occ_bwt_rbwt_cuda_aux4(tmp.x,strand);
        m += __occ_bwt_rbwt_cuda_aux4(tmp.y,strand);
        m += __occ_bwt_rbwt_cuda_aux4(tmp.z,strand);
        m += __occ_bwt_rbwt_cuda_aux4(tmp.w,strand);
        tmp = BWT_RBWT_OCC4b(i+2,strand);
    }
    if (k&0x20)
    {
        m += __occ_bwt_rbwt_cuda_aux4(tmp.x,strand);
        m += __occ_bwt_rbwt_cuda_aux4(tmp.y,strand);
        tmp1=tmp.z;
        tmp2=tmp.w;
    } else {
        tmp1=tmp.x;
        tmp2=tmp.y;
    }
    if (k&0x10)
    {
        m += __occ_bwt_rbwt_cuda_aux4(tmp1,strand);
        tmp1=tmp2;
    }
    // just shift away the unwanted character, no need to shift back
    // number of c in tmp1 will still be correct
    m += __occ_bwt_rbwt_cuda_aux4(tmp1>>(((~k)&15)<<1),strand);
    n.x = m&0xff; n.y = m>>8&0xff; n.z = m>>16&0xff; n.w = m>>24;

    // retrieve the total count from index the number of character C in the up k/128bits interval
    tmp = BWT_RBWT_OCC4b(i,strand);
    n.x += tmp.x; n.x -= ~k&15; n.y += tmp.y; n.z += tmp.z; n.w += tmp.w;

    return n;
}

__device__
bwtint_t bwt_rbwt_cuda_occ(
		bwtint_t k, ubyte_t c, char strand)
// return occurrence of c in bwt with k smallest suffix by reading it from texture memory
// This function uses joint texture for bwt/rbwt
{
    uint4 ok = bwt_rbwt_cuda_occ4(k,strand);
    switch ( c )
    {
    case 0:
        return ok.x;
    case 1:
        return ok.y;
    case 2:
        return ok.z;
    case 3:
        return ok.w;
    }
    return 0;
}

// This function can process a maximum of 2**15 reads per block.
// bwt_sa() with texture reads (alignment 1).
// BWT and RBWT are separated by order (run in succession).
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
    int rbwt_sa_intv)
{
    // Declare and initialize variables.
    // Thread ID and offset.
    const int tid = threadIdx.x;
    const int bid = blockIdx.x / 2; //blockIdx.x < n_block/2 ? blockIdx.x : blockIdx.x/2;
    const int n_sa_total = n_seq_per_block + (blockIdx.x < 2*block_mod ? 1 : 0);

    __shared__ int n_sa_processed;
    __shared__ int n_sa_remaining;
    __shared__ int n_sa_in_buf;
    __shared__ int n_sa_in_buf_prev;
    __shared__ int n_sa_buf_empty;
    __shared__ int sa_next_no;
    __shared__ char block_strand;
    __shared__ int offset;
    __shared__ int block_sa_intv;
    //__shared__ int

    __shared__ bwtint_t sa_buf_arr[BLOCK_SIZE2];    // Array of "sa".
    __shared__ bwa_maxdiff_mapQ_t maxdiff_mapQ_buf_arr[BLOCK_SIZE2];    // Array of "maxdiff" elements.
    __shared__ int16_t sa_origin[BLOCK_SIZE2];  // Index of reads.
    __shared__ bwtint_t sa_return[BLOCK_SIZE2]; // Return value.

    // "n_sa_total" is the total number of reads of the block, "n_sa_processed" is the number of finished
    // reads: "n_total = n_sa_processed + n_sa_remaining". "n_sa_in_buf" (<= BLOCK_SIZE2) is the number of
    // reads in process in the buffer, and "n_sa_buf_empty" is the number of empty elements in the buffer:
    // BUFFER_SIZE2 = n_sa_in_buf + n_sa_buf_empty". "sa_next_no" (< "n_total") is the number of the read
    // to fetch next from global or texture memory.

    // "block_strand" is the strand of the block.

    // Blocks come in pairs. Blocks 0 and 1 do the same first no. of reads given by "n_sa_total".
    // But strand 0 does the forward strand and 1 the reverse strand, so there is no duplicity of
    // effort.


    ////////////////////////////////////////////////////////////
    // Run BWT.
    ////////////////////////////////////////////////////////////

    // Which strand to use.
    block_strand = blockIdx.x & 0x1;
    block_sa_intv = block_strand ? bwt_sa_intv : rbwt_sa_intv ;

    __syncthreads();

    offset = bid < block_mod ? (n_seq_per_block+1)*bid : (n_seq_per_block+1)*block_mod + n_seq_per_block*(bid-block_mod);
    n_sa_processed = 0;
    n_sa_remaining = n_sa_total;
    n_sa_in_buf = min(n_sa_total,BLOCK_SIZE2);
    n_sa_in_buf_prev = n_sa_in_buf;
    n_sa_buf_empty = BLOCK_SIZE2 - n_sa_in_buf;
    sa_next_no = n_sa_in_buf;


    __syncthreads();

    // Fill arrays with initial values. (Do this first to reduce latency as reading from global
    // memory is time-consuming).
    if (tid < n_sa_in_buf)
    {
        maxdiff_mapQ_buf_arr[tid] = seqs_maxdiff_mapQ_de[offset+tid];
        sa_buf_arr[tid] = seqs_sa_de[offset+tid];
    }

    // Set the position in the position array and state which threads are not in use (-1).
    sa_origin[tid] = tid < n_sa_in_buf ? tid : -1;

    // Initialize the return values
    sa_return[tid] = 0;

    // Get new reads.
    if (tid < n_sa_in_buf &&
        !(maxdiff_mapQ_buf_arr[tid].strand == block_strand
        && (maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_UNIQUE ||
        maxdiff_mapQ_buf_arr[tid].type == BWA_TYPE_REPEAT)))
    {
        update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);
        sa_origin[tid] = -1;

        fetch_read_new_in_parallel(
            &maxdiff_mapQ_buf_arr[tid],
            &sa_origin[tid],
            seqs_maxdiff_mapQ_de,
            offset,
            &n_sa_in_buf,
            &n_sa_buf_empty,
            &n_sa_processed,
            &n_sa_remaining,
            &sa_next_no,
            n_sa_total,
            block_strand);

        if (sa_origin[tid] != -1)
        {
            sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
                    //tex1Dfetch(sa_tex,offset+sa_origin[tid]);
            sa_return[tid] = 0;
        }
    }

    // Get rid of reads that are on the wrong strand, fetch new ones.
    __syncthreads();

    if (n_sa_in_buf < BLOCK_SIZE2 && tid == 0)
    {
        sort_reads(
            &sa_buf_arr[0],
            &maxdiff_mapQ_buf_arr[0],
            &sa_origin[0],
            &sa_return[0],
            &n_sa_in_buf,
            &n_sa_in_buf_prev);
    }

    __syncthreads();

    // Start bwt_sa() in a loop until all reads have been processed.
    while (true)
    {
        // Return finished reads, fetch new reads if possible. Run in parallel, not sequentially.
        if //(sa_origin[tid] != -1)
           (tid < n_sa_in_buf)
        {
            char continuation = 1;
            if (sa_buf_arr[tid] % block_sa_intv == 0) {continuation = 0;}
            else if (sa_buf_arr[tid] == bwt_rbwt_cuda[block_strand].primary)
            {
                sa_return[tid]++;
                sa_buf_arr[tid] = 0;
                continuation = 0;
            }

            if (!continuation)
            {
                int max_diff = bwa_cuda_cal_maxdiff(maxdiff_mapQ_buf_arr[tid].len,BWA_AVG_ERR,fnr);
                uint8_t mapQ = cuda_bwa_approx_mapQ(&maxdiff_mapQ_buf_arr[tid],max_diff);
                bwtint_t pos =
                    sa_return[tid] +
                    tex1Dfetch(bwt_rbwt_sa_tex,bwt_rbwt_sa_offset[block_strand]+sa_buf_arr[tid]/block_sa_intv);

                // If on the reverse strand.
                if (!block_strand)
                {
                    pos = bwt_rbwt_cuda[!block_strand].seq_len - (pos + maxdiff_mapQ_buf_arr[tid].len);
                }

                // Return read that is finished.
                seqs_pos_de[offset+sa_origin[tid]] = pos;

                // Return "mapQ".
                seqs_mapQ_de[offset+sa_origin[tid]] = mapQ;
                sa_origin[tid] = -1;

                // Update indices.
                update_indices_in_parallel(&n_sa_processed,&n_sa_remaining,&n_sa_in_buf,&n_sa_buf_empty);

                // Get new read.
                fetch_read_new_in_parallel(
                    &maxdiff_mapQ_buf_arr[tid],
                    &sa_origin[tid],
                    seqs_maxdiff_mapQ_de,
                    offset,
                    &n_sa_in_buf,
                    &n_sa_buf_empty,
                    &n_sa_processed,
                    &n_sa_remaining,
                    &sa_next_no,
                    n_sa_total,
                    block_strand);

                if (sa_origin[tid] != -1)
                {
                    sa_buf_arr[tid] = seqs_sa_de[offset+sa_origin[tid]];
                            //tex1Dfetch(sa_tex,offset+sa_origin[tid]);
                    sa_return[tid] = 0;
                }
            }
        }

        __syncthreads();

        //break;

        if (n_sa_remaining <= 0) break;

        // This section puts reads in the buffer first to allow full warps to be run.
        if (n_sa_in_buf < BLOCK_SIZE2)
        {
            if (tid == 0)
            {
                sort_reads(
                    &sa_buf_arr[0],
                    &maxdiff_mapQ_buf_arr[0],
                    &sa_origin[0],
                    &sa_return[0],
                    &n_sa_in_buf,
                    &n_sa_in_buf_prev);
            }

            __syncthreads();
        }

        sa_return[tid]++;

        if //(sa_origin[tid] != -1)
        (tid < n_sa_in_buf)
        {
            ////////////////////////////////////////////////////////////
            // Start bwt_sa (bwtint_t bwt_sa(const bwt_t *bwt, bwtint_t k))
            ////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////
            // Start #define bwt_invPsi(bwt, k)
            ////////////////////////////////////////////////////////////
            // First conditional expression.
            // Moved to the section above where "else if (sa_arr[k] == bwt_cuda.primary)".

            // Second conditional expression.
            bwtint_t invPsi1 = sa_buf_arr[tid] < bwt_rbwt_cuda[block_strand].primary ? sa_buf_arr[tid] : sa_buf_arr[tid]-1;
            ubyte_t invPsi2 = bwt_rbwt_B0(invPsi1,block_strand);
            invPsi1 = bwt_rbwt_cuda_occ(sa_buf_arr[tid],invPsi2,block_strand);
            sa_buf_arr[tid] = bwt_rbwt_cuda[block_strand].L2[invPsi2]+invPsi1;
        }
    }
}

// Calculate blocks for device. 2 blocks share the same reads; there is one block for each strand.
void calc_n_block1(
    int *n_sp_to_use,
    int *n_block,
    int *n_seq_per_block,
    int *block_mod,
    int n_mp_on_device,
    int n_sp_per_mp,
    int n_seqs)
{
	// No. of MPs on device.
	n_sp_to_use[0] = n_mp_on_device*n_sp_per_mp;
	// No. of blocks to use.
	n_block[0] = n_sp_to_use[0];
	// No of sequences per block.
	n_seq_per_block[0] = n_seqs / n_block[0];
	// Extra sequences to be shared amongst the first blocks.
	block_mod[0] = n_seqs - n_seq_per_block[0] * n_block[0];
	// No. of blocks in total; one for each strand.
	n_block[0] += n_block[0];
#if DEBUG_LEVEL > 1
	fprintf(stderr, "n_sp_to_use: %i n_block: %i n_seq_per_block: %i block_mod: %i n_mp_on_device: %i n_sp_per_mp: %i n_seqs: %i\n", *n_sp_to_use, *n_block, *n_seq_per_block, *block_mod, n_mp_on_device, n_sp_per_mp, n_seqs);
#endif
	return;
}

void report_cuda_error_GPU(const char *message)
{
	cudaError_t cuda_err = cudaGetLastError();

	if(cudaSuccess != cuda_err)
	{
		fprintf(stderr,"%s\n",message);
		fprintf(stderr,"%s\n", cudaGetErrorString(cuda_err));
		exit(1);
	}
}

void report_cuda_error_CPU(const char * message)
{
	fprintf(stderr,"%s\n",message);
	exit(1);
}
#endif
///////////////////////////////////////////////////////////////
// End CUDA SAMSE core
///////////////////////////////////////////////////////////////


