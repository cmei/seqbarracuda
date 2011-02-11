/*
   Barracuda - A Short Sequence Aligner for NVIDIA Graphics Cards

   Module: bwtaln.cu  Read sequence reads from file, modified from BWA to support barracuda alignment functions

   Copyright (C) 2010, Brian Lam and Simon Lam

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License
   as published by the Free Software Foundation; either version 3
   of the License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

   This program is based on a modified version of BWA 0.4.9

*/

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

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

#define OUTPUT_ALIGNMENTS 1 // should leave ON for outputting alignment
#define STDOUT_STRING_RESULT 0 // output alignment in text format (in SA coordinates, not compatible with SAM output modules(samse/pe)
#define STDOUT_BINARY_RESULT 1//output alignment for samse/sampe (should leave ON)
#define DO_RC 1 //enable looking for matches on the complementary strand

//for function inexact match
#define STATE_M 0
#define STATE_I 1
#define STATE_D 2

//define whether to use DFS or BFS based search, default is DFS (where it is BFS in BWA)
#define DFS 1 //for cuda code (for DEBUG ONLY, should leave on)
#define CPU_DFS 0 //for cpu code (for DEBUG ONLY, should leave off)


//define maximum stack entries for BFS cannot go beyond 512 (ptx error), not used for DFS
#define MAX_NO_OF_GAP_ENTRIES 512

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


// CUDA includes
//#include <cutil.h>

// enable looking up of k and l in the same time for counting character occurrence (slower, so disable by default)
#define BWT_2_OCC_ENABLE 0
// use lookup table when instead of counting character occurrence, (faster so enable by default)
#define BWT_TABLE_LOOKUP_ENABLE 1

// Maximum exponential is up to 30 [~ 1  GBytes] for non-debug, non alignment
// Maximum exponential is up to 26 [~ 128MBytes] for debug
// Maximum exponential is up to 23 for alignment with 4GB RAM(default : 23)
#define SEQUENCE_TABLE_SIZE_EXPONENTIAL 23



__device__ __constant__ bwt_t bwt_cuda;
__device__ __constant__ bwt_t rbwt_cuda;
__device__ __constant__ gap_opt_t options_cuda;

// uint4 is used because the maximum width for CUDA texture bind of 1D memory is 2^27,
// and uint4 the structure 4xinteger is x,y,z,w coordinates and is 16 bytes long,
// therefore effectively there are 2^27x16bytes memory can be access = 2GBytes memory.
texture<uint4, 1, cudaReadModeElementType> bwt_occ_array;
texture<uint4, 1, cudaReadModeElementType> rbwt_occ_array;
texture<unsigned int, 1, cudaReadModeElementType> sequences_array;
texture<uint2, 1, cudaReadModeElementType> sequences_index_array;

// The following line will copy the
// bwt occurrence array to global and bind to texture, bwt structure to constant memory
unsigned long long copy_bwts_to_cuda_memory( const char * prefix, unsigned int ** bwt,  unsigned int ** rbwt )
{
	bwt_t * bwt_src;
	char str[100];
	unsigned long long size_read = 0;

	if ( bwt != 0 )
	{
		//Original BWT
		//Load bwt occurrence array from from disk
		strcpy(str, prefix); strcat(str, ".bwt");  bwt_src = bwt_restore_bwt(str);
		size_read += bwt_src->bwt_size*sizeof(uint32_t);
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
	if ( rbwt != 0 )
	{
		//Reversed BWT
		//Load bwt occurrence array from from disk
		strcpy(str, prefix); strcat(str, ".rbwt");  bwt_src = bwt_restore_bwt(str);
		size_read += bwt_src->bwt_size*sizeof(uint32_t);
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

int copy_sequences_to_cuda_memory ( bwa_seqio_t *bs, uint2 * global_sequences_index, uint2 * main_sequences_index, unsigned char * global_sequences, unsigned char * main_sequences, unsigned int * read_size, unsigned short & max_length, int mid)
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

		if ( accumulated_length + MAX_SEQUENCE_LENGTH > (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL+1)) ) break;
	}
	//copy main_sequences_width from host to device
	cudaUnbindTexture(sequences_index_array);
    cudaMemcpy(global_sequences_index, main_sequences_index, (number_of_sequences)*sizeof(uint2), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_index_array, global_sequences_index, (number_of_sequences)*sizeof(uint2));

    //copy main_sequences from host to device, sequences array length should be accumulated_length/2
    cudaUnbindTexture(sequences_array);
    cudaMemcpy(global_sequences, main_sequences, (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL))*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sequences_array, global_sequences, (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL))*sizeof(unsigned char));

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

#if DFS == 0

//new gap stack functions, for bfs based matchgap function (obsolete)
//////////////////////////////////

typedef struct
{
	uint4 sa_length [MAX_NO_OF_GAP_ENTRIES];
	// sa_length.x = k, .y = l, .z = length, .w = last_diff_pos
	uchar4 mm_gap_state [MAX_NO_OF_GAP_ENTRIES];
	// mm_gap_state.x = nmm, y = ngapo, z = ngape, w = state
	unsigned int score_table[MAX_NO_OF_GAP_ENTRIES];
} stack_table_t;

typedef struct
{
	int no_of_entries;
	int best_score_location;
	unsigned int next_best_score;
} stack_table_league_t;

__device__ void stack_init(stack_table_t *stack, stack_table_league_t *stack_league)
{

	stack_league->no_of_entries = 0;
	stack_league->best_score_location = 0;
	stack_league->next_best_score = 0XFFFF;

	for (int i = 0; i < MAX_NO_OF_GAP_ENTRIES; i++)
	{
		stack->sa_length[i].x = 0;
		stack->sa_length[i].y = 0;
		stack->sa_length[i].z = 0;
		stack->sa_length[i].w = 0;
		stack->mm_gap_state[i].x = 0;
		stack->mm_gap_state[i].y = 0;
		stack->mm_gap_state[i].z = 0;
		stack->mm_gap_state[i].w = 0;
		stack->score_table[i]= 0XFFFF;
	}
	return;
}
__device__ void cuda_stack_push(stack_table_t *stack, stack_table_league_t *stack_league, int a, int i, bwtint_t k, bwtint_t l, int n_mm, int n_gapo, int n_gape,
							int state, int is_diff, const gap_opt_t *opt)
{
		if (stack_league->no_of_entries < MAX_NO_OF_GAP_ENTRIES)
		{
			int this_entry = stack_league->no_of_entries;
			//if score is best, record score location
			unsigned int this_score = aln_score(n_mm, n_gapo, n_gape, opt);
			if (this_score < stack->score_table[stack_league->best_score_location])
			{
				stack_league->next_best_score = stack->score_table[stack_league->best_score_location];
				stack_league->best_score_location = this_entry;
			}
			else if(this_score < stack_league->next_best_score)
			{
				stack_league->next_best_score = this_score;
			}

			// sa_length.x = k, .y = l, .z = length, .w = last_diff_pos
			// mm_gap_state.x = nmm, y = ngapo, z = ngape, w = state

			//write stuffs to array
			// record down gap entry information
			stack->sa_length[this_entry].x = k;
			stack->sa_length[this_entry].y = l;
			stack->sa_length[this_entry].z = i ;
			if (is_diff) stack->sa_length[this_entry].w = i;
			stack->mm_gap_state[this_entry].x = n_mm;
			stack->mm_gap_state[this_entry].y = n_gapo;
			stack->mm_gap_state[this_entry].z = n_gape;
			stack->mm_gap_state[this_entry].w = (a<<7)|state;
			stack->score_table[this_entry]=this_score;
			stack_league->no_of_entries ++;
		}
		return;
}

__device__ void cuda_stack_pop(stack_table_t *stack, stack_table_league_t *stack_league, gap_entry_t * e, const gap_opt_t *opt)
{

	gap_entry_t temp;
	// sa_length.x = k, .y = l, .z = length, .w = last_diff_pos
	// mm_gap_state.x = nmm, y = ngapo, z = ngape, w = state
	temp.k = stack->sa_length[stack_league->best_score_location].x;
	temp.l = stack->sa_length[stack_league->best_score_location].y;
	temp.length = stack->sa_length[stack_league->best_score_location].z;
	temp.last_diff_pos = stack->sa_length[stack_league->best_score_location].w;
	temp.n_mm = stack->mm_gap_state[stack_league->best_score_location].x;
	temp.n_gapo = stack->mm_gap_state[stack_league->best_score_location].y;
	temp.n_gape = stack->mm_gap_state[stack_league->best_score_location].z;
	temp.state = stack->mm_gap_state[stack_league->best_score_location].w;

	*e = temp;
	int original_best_score = stack->score_table[stack_league->best_score_location];

	//if this is not the last entry
	if (stack_league->no_of_entries > 1)
	{
		//removed popped entry from table and slot the last entry into empty space;

		stack->sa_length[stack_league->best_score_location] = stack->sa_length[stack_league->no_of_entries-1];
		stack->mm_gap_state[stack_league->best_score_location] = stack->mm_gap_state[stack_league->no_of_entries-1];

		stack->score_table[stack_league->best_score_location] = stack->score_table[stack_league->no_of_entries-1];

		stack_league->no_of_entries --;

		unsigned int new_best_score = 0xFFFF;
		int i = 0;

		while (i < stack_league->no_of_entries)
		{
				if (stack->score_table[i] <= original_best_score)
				{
					stack_league->best_score_location = i;
					break;
				}
				else if (stack->score_table[i] == stack_league->next_best_score)
				{
					stack_league->best_score_location = i;
					stack_league->next_best_score = 0XFFFF;
					break;
				}
				else if (stack->score_table[i] < new_best_score)
				{
					new_best_score = stack->score_table[i];
					stack_league->best_score_location = i;
				}
				i++;
		}
	}
	else
	{
		stack_league->no_of_entries = 0;
	}
	return;
}
#endif

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

#if DFS == 1

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
__device__ int cuda_dfs_match(const int len, const unsigned char *str, const int sequence_type, unsigned int *widths, unsigned char *bids, const gap_opt_t *opt, alignment_store_t *aln, int best_score, const int max_aln)
//This function tries to find the alignment of the sequence and returns SA coordinates, no. of mismatches, gap openings and extensions
//It uses a depth-first search approach rather than breath-first as the memory available in CUDA is far less than in CPU mode
//The search rooted from the last char [len] of the sequence to the first with the whole bwt as a ref from start
//and recursively narrow down the k(upper) & l(lower) SA boundaries until it reaches the first char [i = 0], if k<=l then a match is found.
{

	//Initializations
	int best_diff = opt->max_diff + 1;
	int max_diff = opt->max_diff;
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

	//Initialize memory stores first in, last out
	cuda_dfs_initialize(entries_info, entries_scores, done_push_types/*, scores*/); //initialize initial entry, current stage set at 0 and done push type = 0

	//push first entry, the first char of the query sequence into memory stores for evaluation
	cuda_dfs_push(entries_info, entries_scores, done_push_types, len, 0, bwt->seq_len, 0, 0, 0, 0, 0, current_stage); //push initial entry to start

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
		if (n_aln == max_aln)break;
		if (best_cnt > options_cuda.max_top2) break;
		if (loop_count > max_count) break;

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

		//calculate the allowance for differences
		m = max_diff - e_n_mm - e_n_gapo;
		if (options_cuda.mode & BWA_MODE_GAPE) m -= e_n_gape;

		if(score > worst_tolerated_score) break;

		// check if the entry is outside boundary or is over the max diff allowed)
		if (m < 0 || (i > 0 && m < bids[i-1]))
		{
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
			int c = str[i];
			done_push_types[current_stage].x = 1;
			if (c < 4)
			{
				if (ks[current_stage][c] <= ls[current_stage][c])
				{
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
								done_push_types[current_stage].z++;
								unsigned int tmp = (options_cuda.mode & BWA_MODE_LOGGAP)? (int_log2_cuda(e_n_gape + e_n_gapo))>>1 + 1 : e_n_gapo + e_n_gape;
								if (i >= options_cuda.indel_end_skip + tmp && len - i >= options_cuda.indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo + 1, e_n_gape, STATE_D, 1, current_stage+1);
									current_stage++; //advance stage number by 1
									continue;
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
						done_push_types[current_stage].z++;
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
										cuda_dfs_push(entries_info, entries_scores, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e_n_mm, e_n_gapo, e_n_gape + 1, STATE_D, 1, current_stage+1);
										current_stage++; //advance stage number
										continue;
									}
								}
							}
						}
					}
				} //end else if (e_state == STATE_D)

		}//end if (!allow_diff)
		current_stage--;

	} //end do while loop

	aln->no_of_alignments = n_aln;

	return best_score;
}
#endif

#if DFS == 0
/////////////////////////////////////////////////////////////////////////
//match gap, bfs, now obsolete
/////////////////////////////////////////////////////////////////////////
__device__ void cuda_inexact_match(const int len, const unsigned char *str, const int sequence_type, unsigned int *widths, unsigned char *bids, const gap_opt_t *opt, alignment_store_t *aln, int best_score)
{
	int best_diff = opt->max_diff + 1, max_diff = opt->max_diff;
	int best_cnt = 0;
	int j, n_aln = 0;
	bwt_t * bwt = (sequence_type == 0)? &rbwt_cuda: &bwt_cuda; // rbwt for sequence 0 and bwt for sequence 1;
	int bwt_type = 1-sequence_type;
	//initialization of stack of gap entries.
	stack_table_t stack_table;
	stack_table_league_t stack_index;
	stack_init(&stack_table, &stack_index);
	cuda_stack_push(&stack_table, &stack_index, sequence_type, len, 0, bwt->seq_len, 0, 0, 0, 0, 0, opt);

	syncthreads();

	while (stack_index.no_of_entries && stack_index.no_of_entries < opt->max_entries)
	{

		gap_entry_t e;
		int a, i, m;
		//int m_seed = 0;
		int hit_found, allow_diff, allow_M, tmp;
		bwtint_t k, l, occ;
		bwtint_t cnt_k[4], cnt_l[4];
		int worst_tolerated_score = best_score + options_cuda.s_mm;

		if (stack_index.no_of_entries > opt->max_entries) break;
		if (n_aln > (MAX_NO_OF_ALIGNMENTS>>1)) break;
		cuda_stack_pop(&stack_table, &stack_index, &e, opt);

		k = e.k; l = e.l; // SA interval
		a = e.state>>7; i = e.length; // strand, length
		if (!(opt->mode & BWA_MODE_NONSTOP) && aln_score(e.n_mm,e.n_gapo,e.n_gape,opt) > best_score + opt->s_mm) break; // no need to proceed

		m = max_diff - (e.n_mm + e.n_gapo) - (opt->mode & BWA_MODE_GAPE) * e.n_gape;

		int score = aln_score(e.n_mm, e.n_gapo, e.n_gape, opt);

		//if (opt->mode & BWA_MODE_GAPE) m -= e.n_gape;
		if (m < 0) continue;

		/*
		if (seed) { // apply seeding
			m_seed = opt->max_seed_diff - (e.n_mm + e.n_gapo);
			if (opt->mode & BWA_MODE_GAPE) m_seed -= e.n_gape;
		}
		*/

		if (i > 0 && m < bids[i-1]) continue;

		// check whether a hit is found
		hit_found = 0;
		if (i == 0) hit_found = 1;
		else if (m == 0 && ((e.state&0x7F) == STATE_M || (opt->mode&BWA_MODE_GAPE) || e.n_gape == opt->max_gape)) { // no diff allowed
			if (bwt_cuda_match_exact(bwt_type, i, str, &k, &l)) hit_found = 1;
			else continue; // no hit, skip
		}

		if (hit_found) { // action for found hits

			int do_add = 1;
			if (n_aln == 0) {
				best_score = score;
				best_diff = e.n_mm + e.n_gapo;
				if (opt->mode & BWA_MODE_GAPE) best_diff += e.n_gape;
				if (!(opt->mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;
			else if (best_cnt > opt->max_top2) break; // top2b behaviour
			if (e.n_gapo) { // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array
				for (j = 0; j != n_aln; ++j)
					if (aln->alignment_info[j].k == k && aln->alignment_info[j].l == l) break;
				if (j < n_aln) do_add = 0;
			}

			if (do_add) { // append result the alignment record array
				gap_stack_shadow_cuda(l - k + 1, len, bwt->seq_len, e.last_diff_pos, widths, bids);
				if (n_aln < (MAX_NO_OF_ALIGNMENTS/2))
				{
					// record down number of mismatch, gap open, gap extension and a??
					aln->alignment_info[n_aln].n_mm = e.n_mm;
					aln->alignment_info[n_aln].n_gapo = e.n_gapo;
					aln->alignment_info[n_aln].n_gape = e.n_gape;
					aln->alignment_info[n_aln].a = a;
					// the suffix array interval
					aln->alignment_info[n_aln].k = k;
					aln->alignment_info[n_aln].l = l;
					aln->alignment_info[n_aln].score = score;
					++n_aln;
				}
			}

			continue;
		}


		--i;

		uint4 cuda_cnt_k, cuda_cnt_l;

		// retrieve Occurrence values !!!slow!!!!!
		{
			cuda_cnt_k = (!sequence_type)? rbwt_cuda_occ4(k-1): bwt_cuda_occ4(k-1);
			cuda_cnt_l = (!sequence_type)? rbwt_cuda_occ4(l): bwt_cuda_occ4(l);
		}

		//bwts_cuda_2occ4(k-1, l, &cuda_cnt_k, &cuda_cnt_l, bwt_type);

		cnt_k[0] = cuda_cnt_k.x;
		cnt_k[1] = cuda_cnt_k.y;
		cnt_k[2] = cuda_cnt_k.z;
		cnt_k[3] = cuda_cnt_k.w;
		cnt_l[0] = cuda_cnt_l.x;
		cnt_l[1] = cuda_cnt_l.y;
		cnt_l[2] = cuda_cnt_l.z;
		cnt_l[3] = cuda_cnt_l.w;

		occ = l - k + 1;

		// test whether difference is allowed
		allow_diff = allow_M = 1;

		if (i > 0) {
			//int ii = i - (len - opt->seed_len);
			if (bids[i-1] > m-1) allow_diff = 0;
			else if (bids[i-1] == m-1 && bids[i] == m-1 && widths[i-1] == widths[i]) allow_M = 0;

			/*if (seed && ii > 0) {
				if (seeded_bids[ii-1] > m_seed-1) allow_diff = 0;
				else if (seeded_bids[ii-1] == m_seed-1 && seeded_bids[ii] == m_seed-1
						 && seeded_widths[ii-1] == seeded_widths[ii]) allow_M = 0;
			}*/
		}

		// insertion and deletions
		tmp = (opt->mode & BWA_MODE_LOGGAP)? int_log2_cuda(e.n_gape + e.n_gapo)/2+1 : e.n_gapo + e.n_gape;
		if (allow_diff && i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp) {
			if ((e.state&0x7F) == STATE_M) { // gap open
				if (e.n_gapo < opt->max_gapo) { // gap open is allowed
					// insertion

					if((score + options_cuda.s_gapo <= worst_tolerated_score) ||(options_cuda.mode & BWA_MODE_NONSTOP))
					cuda_stack_push(&stack_table, &stack_index, a, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_I, 1, opt);

					// deletion
					for (j = 0; j != 4; ++j) {
						k = bwt->L2[j] + cnt_k[j] + 1;
						l = bwt->L2[j] + cnt_l[j];
						if (k <= l)
						{
							if((score + options_cuda.s_gapo <= worst_tolerated_score) ||(options_cuda.mode & BWA_MODE_NONSTOP))
							cuda_stack_push(&stack_table, &stack_index, a, i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_D, 1, opt);
						}
					}
				}
			} else if ((e.state&0x7F) == STATE_I) { // Extension of an insertion
				if (e.n_gape < opt->max_gape) // gap extension is allowed
				{
					if((score + options_cuda.s_gape <= worst_tolerated_score) ||(options_cuda.mode & BWA_MODE_NONSTOP))
					cuda_stack_push(&stack_table, &stack_index, a, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, STATE_I, 1, opt);
				}

			} else if ((e.state&0x7F) == STATE_D) { // Extension of a deletion
				if (e.n_gape < opt->max_gape) { // gap extension is allowed
					if (e.n_gape + e.n_gapo < max_diff || occ < opt->max_del_occ) {
						for (j = 0; j != 4; ++j) {
							k = bwt->L2[j] + cnt_k[j] + 1;
							l = bwt->L2[j] + cnt_l[j];
							if (k <= l)
							{
								if((score + options_cuda.s_gape <= worst_tolerated_score) ||(options_cuda.mode & BWA_MODE_NONSTOP))
								cuda_stack_push(&stack_table,  &stack_index,a, i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, STATE_D, 1, opt);
							}
						}
					}
				}
			}
		}

		// mismatches
		if (allow_diff && allow_M) { // mismatch is allowed
			for (j = 1; j <= 4; ++j) {
				int c = (str[i] + j) & 3;
				int is_mm = (j != 4 || str[i] > 3);
				k = bwt->L2[c] + cnt_k[c] + 1;
				l = bwt->L2[c] + cnt_l[c];
				if (k <= l)
				{
					//if((score + is_mm * options_cuda.s_mm <= worst_tolerated_score) ||(options_cuda.mode & BWA_MODE_NONSTOP))
					cuda_stack_push(&stack_table, &stack_index, a, i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, STATE_M, is_mm, opt);
				}
			}
		} else if (str[i] < 4) { // try exact match only

			int c = str[i] & 3;
			k = bwt->L2[c] + cnt_k[c] + 1;
			l = bwt->L2[c] + cnt_l[c];
			if (k <= l) cuda_stack_push(&stack_table, &stack_index, a, i, k, l, e.n_mm, e.n_gapo, e.n_gape, STATE_M, 0, opt);
		}
	}
	aln->no_of_alignments = n_aln;

	return;
}
#endif

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
	int max_aln = options_cuda.max_aln;
	//initialize local options for each query sequence
	gap_opt_t local_options = options_cuda;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences )
	{
		//get sequences from texture memory
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);
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
		#if DFS == 0
		cuda_inexact_match(sequence_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score);
		#endif //DFS == 0
		#if DFS == 1
		//syncthreads();
		int best_score = cuda_dfs_match(sequence_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		#endif //DFS == 1

		// copy alignment info to global memory
		#if OUTPUT_ALIGNMENTS == 1
		global_alignment_store[blockId] = local_alignment_store;
		int no_aln = local_alignment_store.no_of_alignments;
		#endif // OUTPUT_ALIGNMENTS == 1


		#if DO_RC == 1

		//work on reverse complementary sequence (rbwt for w, bwt for match)
		sequence_type = 1;

		// Calculate w
		syncthreads();
		bwt_cuda_device_calculate_width(local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, sequence_length);

		//Align with reverse reference sequence
		#if DFS == 0
		syncthreads();
		int best_score = local_alignment_store.alignment_info[0].score;
		cuda_inexact_match(sequence_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, best_score);
		#endif //DFS == 0
		#if DFS == 1
		syncthreads();
		cuda_dfs_match(sequence_length, local_rc_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, best_score, max_aln);
		#endif //DFS == 1

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

		#endif // DO_RC ==1
	}
	return;
}


__global__ void cuda_directional_inexact_match_caller(int no_of_sequences, unsigned short max_sequence_length, alignment_store_t* global_alignment_store, unsigned char cuda_opt)
//CUDA kernal for inexact match on a specified strand
//calls bwt_cuda_device_calculate_width to determine the boundaries of the search space
//and then calls dfs_match to search for alignment using dfs approach
{
	// Block ID for CUDA threads, as there is only 1 thread per block possible for now
	unsigned int blockId = blockIdx.x * blockDim.x + threadIdx.x;

	//Local store for sequence widths bids and alignments
	unsigned int local_sequence_widths[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence_bids[MAX_SEQUENCE_LENGTH];
	unsigned char local_sequence[MAX_SEQUENCE_LENGTH];
	alignment_store_t local_alignment_store;
	int max_aln = options_cuda.max_aln;
	//initialize local options for each query sequence
	gap_opt_t local_options = options_cuda;

	//Core function
	// work on valid sequence only
	if ( blockId < no_of_sequences )
	{
		//get sequences from texture memory
		const uint2 sequence_info = tex1Dfetch(sequences_index_array, blockId);
		const unsigned int sequence_offset = sequence_info.x;
		const unsigned short sequence_length = sequence_info.y;
		unsigned int last_read = ~0;
		unsigned int last_read_data;

		for (int i = 0; i < sequence_length; ++i)
		{
			unsigned char c = read_char(sequence_offset + i, &last_read, &last_read_data );
			if (cuda_opt == 1)
			{
				local_sequence[i] = c;
			}
			else if (cuda_opt == 2)
			{
				if (local_options.mode & BWA_MODE_COMPREAD)
				{
					local_sequence[i] = (c > 3)? c : (3 - c);
				}else
				{
					local_sequence[i] = c;
				}
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

		int sequence_type = 0;
		if (cuda_opt == 2) sequence_type = 1;

		// Calculate w
		syncthreads();
		bwt_cuda_device_calculate_width(local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, sequence_length);

		//Align with forward reference sequence
		#if DFS == 0
		cuda_inexact_match(sequence_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score);
		#endif //DFS == 0
		#if DFS == 1
		cuda_dfs_match(sequence_length, local_sequence, sequence_type, local_sequence_widths, local_sequence_bids, &local_options, &local_alignment_store, worst_score, max_aln);
		#endif //DFS == 1

		// copy alignment info to global memory
		#if OUTPUT_ALIGNMENTS == 1
		global_alignment_store[blockId] = local_alignment_store;
		#endif // OUTPUT_ALIGNMENTS == 1
	}
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////
//Line below is for BWA CPU code


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



#if CPU_DFS == 1

		alignment_store_t alignments;
		alignments.no_of_alignments = 0;
		// core function
		dfs_match(bwt, p->len, seq, w, p->len <= opt->seed_len? 0 : seed_w, &local_opt, &alignments);


		#if STDOUT_STRING_RESULT == 1
					if (alignments.no_of_alignments > 0)
					{

						//for (int j = 0; j < 37; j++)
						{
							printf("Sequence %d", i); //seq[0][j]);

						}
						printf(", no of alignments: %d\n", alignments.no_of_alignments);
						for (int j = 0; j < alignments.no_of_alignments && j < MAX_NO_OF_ALIGNMENTS; j++)
						{
							printf("  Aligned read %d, ",j+1);
							printf("a: %d, ", alignments.alignment_info[j].a);
							printf("n_mm: %d, ", alignments.alignment_info[j].n_mm);
							printf("n_gape: %d, ", alignments.alignment_info[j].n_gape);
							printf("n_gapo: %d, ", alignments.alignment_info[j].n_gapo);
							printf("k: %u, ", alignments.alignment_info[j].k);
							printf("l: %u, ", alignments.alignment_info[j].l);
							printf("score: %u\n", alignments.alignment_info[j].score);
						}

					}
		#endif
#endif

#if CPU_DFS == 0
		// core function
		p->aln = bwt_match_gap(bwt, p->len, seq, w, p->len <= opt->seed_len? 0 : seed_w, &local_opt, &p->n_aln, stack);
		// store the alignment
#endif

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
	o->max_gapo = 0; // changed from 1 (BWA) to 0 as GAPPED alignment slows down DFS and not benefit much on alignment mappings
	o->max_gape = 6;
	o->indel_end_skip = 5; o->max_del_occ = 10;
	o->max_aln = 10;
	o->max_entries = -1; //max tried is 1M without problem with 125bp can go larger for shorter reads
#if DFS == 0
	o->max_entries = MAX_NO_OF_GAP_ENTRIES;
#endif
	o->mode = BWA_MODE_GAPE | BWA_MODE_COMPREAD;
	o->seed_len = 0x7fffffff; o->max_seed_diff = 2;
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

void barracuda_bwa_aln_core(const char *prefix, const char *fn_fa, gap_opt_t *opt, unsigned char cuda_opt)
//Main alignment module caller
//Determines the availability of CUDA devices and
//invokes CUDA kernels cuda_inexact_match_caller & cuda_directional_inexact_match_caller
//contains also CPU code for legacy BWA runs

{
	bwa_seqio_t *ks;

	#if STDOUT_BINARY_RESULT == 1
	fwrite(opt, sizeof(gap_opt_t), 1, stdout);
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

		fprintf(stderr,"[aln_core] Running CUDA mode\n");

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
		#endif // OUTPUT_ALIGNMENTS == 1

		if (opt->max_entries < 0 )
		{
			opt->max_entries = 250000;
		}


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//CUDA options
		if (!cuda_opt) fprintf(stderr,"[aln_core] Aligning both forward and reverse complementary sequences.\n");
		if (opt->mid) fprintf(stderr,"[aln_core] Not aligning the first %d bases from the 5' end of sequencing reads.\n",opt->mid);
		if (cuda_opt == 1) fprintf(stderr,"[aln_core] CUDA Options: (f) Align forward sequences only.\n");
		if (cuda_opt == 2) fprintf(stderr,"[aln_core] CUDA Options: (r) Align reverse complementary sequences only.\n");


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Determine Cuda device
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Detect and choose the fastest CUDA device available on the machine
		int num_devices, device;
		cudaGetDeviceCount(&num_devices);

		if (opt->cuda_device == -1)
		{
			if (num_devices >= 1)
			{
			     fprintf(stderr, "[aln_core] Querying CUDA devices:\n");
				 int max_multiprocessors = 0, max_device = 0;
				 for (device = 0; device < num_devices; device++)
				 {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  fprintf(stderr, "[aln_core]   Device %d ", device);
					  for (int i = 0; i < 256; i++)
					  {
						  fprintf(stderr,"%c", properties.name[i]);
					  }
					  fprintf(stderr,", multiprocessor count %d, CUDA compute capability %d.%d.\n", properties.multiProcessorCount, properties.major,  properties.minor);
					  if (max_multiprocessors < properties.multiProcessorCount)
					  {
							  max_multiprocessors = properties.multiProcessorCount;
							  max_device = device;
					  }

	  		     }
				 fprintf(stderr, "[aln_core] Using CUDA device %d.\n", max_device);
				 cudaSetDevice(max_device);
			}
			else
			{
				 fprintf(stderr,"[aln_core] No CUDA device found! aborting!\n");
				 return;
			}

		}
		else if (opt->cuda_device >= 0)
		{
			 fprintf(stderr, "[aln_core] Using specified CUDA device %d.\n", opt->cuda_device);
			 cudaSetDevice(opt->cuda_device);
		}


		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Copy bwt occurrences array to from HDD to CPU then to GPU
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		bwt_read_size = copy_bwts_to_cuda_memory(prefix, &global_bwt, &global_rbwt)>>20;
		// copy_bwt_to_cuda_memory
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		fprintf(stderr, "[aln_core] Finished loading reference sequence assembly, %u MB in %0.2fs (%0.2f MB/s).\n", (unsigned int)bwt_read_size, time_used, ((unsigned int)bwt_read_size)/time_used );

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// allocate GPU working memory
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		//allocate global_sequences memory in device
		cudaMalloc((void**)&global_sequences, (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL))*sizeof(unsigned char));
		main_sequences = (unsigned char *)malloc((1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL))*sizeof(unsigned char));
		//allocate global_sequences_index memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		cudaMalloc((void**)&global_sequences_index, (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL-3))*sizeof(uint2));
		main_sequences_index = (uint2*)malloc((1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL-3))*sizeof(uint2));
		//allocate and copy options (opt) to device constant memory
		cudaMalloc((void**)&options, sizeof(gap_opt_t));
		cudaMemcpy ( options, opt, sizeof(gap_opt_t), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol ( options_cuda, opt, sizeof(gap_opt_t), 0, cudaMemcpyHostToDevice);
		//allocate alignment stores for host and device
		#if OUTPUT_ALIGNMENTS == 1
		//allocate alignment store memory in device assume the average length is bigger the 16bp (currently -3, -4 for 32bp, -3 for 16bp)long
		global_alignment_store_host = (alignment_store_t*)malloc((1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL-3))*sizeof(alignment_store_t));
		cudaMalloc((void**)&global_alignment_store_device, (1ul<<(SEQUENCE_TABLE_SIZE_EXPONENTIAL-3))*sizeof(alignment_store_t));
		#endif // OUTPUT_ALIGNMENTS == 1
		gettimeofday (&end, NULL);
		time_used = diff_in_seconds(&end,&start);
		total_time_used += time_used;
		//fprintf(stderr, "Finished allocating CUDA device memory, it took %0.2fs.\n\n", time_used );

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Core loop (this loads sequences to host memory, transfers to cuda device and aligns via cuda in CUDA blocks)
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		gettimeofday (&start, NULL);
		int loopcount = 0;
		while ( ( no_of_sequences = copy_sequences_to_cuda_memory(ks, global_sequences_index, main_sequences_index, global_sequences, main_sequences, &read_size, max_sequence_length, opt->mid) ) > 0 )
		{

#if DFS == 1
			#define BLOCK_SIZE 320
#endif
#if DFS == 0
			#define BLOCK_SIZE 128
#endif
			#define GRID_UNIT 32
			int gridsize = GRID_UNIT * (1 + int (((no_of_sequences/BLOCK_SIZE) + ((no_of_sequences%BLOCK_SIZE)!=0))/GRID_UNIT));
			dim3 dimGrid(gridsize);
			int blocksize = BLOCK_SIZE;
			dim3 dimBlock(blocksize);

			if (!loopcount)
			{
				fprintf(stderr,"[aln_core] Using SIMT with grid size: %u, block size: %d.\n", gridsize,blocksize) ;
			}

			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_time_used+=time_used;

			//debug only
			//	fprintf(stderr, "[aln_core_debug] Finished loading sequence reads to CUDA device, time taken: %0.2fs, %d sequences, %d bp(%.2f sequences/sec, %.2f bp/sec ", diff_in_seconds(&end,&start), no_of_sequences, read_size, no_of_sequences/time_used, read_size/time_used);
			//	fprintf(stderr, "longest read = %d bp)\n", max_sequence_length);
			//	int l = bwa_cal_maxdiff(max_sequence_length, BWA_AVG_ERR, opt->fnr);
			//	fprintf(stderr, "[aln_core_debug] max_diff: %d\n", l);


			///////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Core match function per sequence readings
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////

			gettimeofday (&start, NULL);
			// calculated W via CUDA
			if (!loopcount) fprintf(stderr, "[aln_core] Now aligning sequence reads to reference assembly, please wait..\n[aln_core] ");

			if (!cuda_opt)
			{
				cuda_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_store_device, cuda_opt);
			}
			else
			{
				cuda_directional_inexact_match_caller<<<dimGrid,dimBlock>>>(no_of_sequences, max_sequence_length, global_alignment_store_device, cuda_opt);

			}


			//Check time
			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_calculation_time_used += time_used;
			total_time_used += time_used;
			//fprintf(stderr, "Finished!  Time taken: %0.2fs  %d sequences (%d bp) analyzed.\n[aln_core] (%.2f sequences/sec, %.2f bp/sec, avg %0.2f bp/sequence)\n[aln_core]", time_used, no_of_sequences, read_size, no_of_sequences/time_used, read_size/time_used, (float)read_size/no_of_sequences ) ;
			fprintf(stderr, ".");

			#if OUTPUT_ALIGNMENTS == 1
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// retrieve alignment information from CUDA device to host
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////

			gettimeofday (&start, NULL);
			cudaMemcpy (global_alignment_store_host, global_alignment_store_device, no_of_sequences*sizeof(alignment_store_t), cudaMemcpyDeviceToHost);

			#if STDOUT_STRING_RESULT == 1
			for (int i = 0; i < no_of_sequences; i++)
			{
				alignment_store_t* tmp = global_alignment_store_host + i;

				if (tmp->no_of_alignments > 0)
				{
				printf("Sequence %d", i);
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
			#endif // STDOUT_STRING_RESULT == 1

			#if STDOUT_BINARY_RESULT == 1

			for (int  i = 0; i < no_of_sequences; ++i)
			{
				alignment_store_t* tmp = global_alignment_store_host + i;


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
				}
			}

			#endif // STDOUT_BINARY_RESULT == 1
			gettimeofday (&end, NULL);
			time_used = diff_in_seconds(&end,&start);
			total_time_used += time_used;
			//fprintf(stderr, "Finished outputting alignment information... %0.2fs.\n\n", time_used);
			fprintf (stderr, ".");
			#endif // OUTPUT_ALIGNMENTS == 1

			total_no_of_base_pair+=read_size;
			total_no_of_sequences+=no_of_sequences;
			gettimeofday (&start, NULL);
			loopcount ++;
		}
		fprintf(stderr, "\n[aln_core] Finished!\n[aln_core] Total no. of sequences: %u, size in base pair: %u bp, average length %0.2f bp/sequence.\n", (unsigned int)total_no_of_sequences, (unsigned int)total_no_of_base_pair, (float)total_no_of_base_pair/(unsigned int)total_no_of_sequences);
		fprintf(stderr, "[aln_core] Alignment Speed: %0.2f sequences/sec or %0.2f bp/sec.\n", (float)total_no_of_sequences/total_calculation_time_used, (float)total_no_of_base_pair/total_calculation_time_used);
		fprintf(stderr, "[aln_core] Total compute time: %0.2fs, total program time: %0.2fs.\n", total_calculation_time_used, total_time_used);

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

#if CPU_DFS == 0

			#if STDOUT_BINARY_RESULT == 1
			for (int i = 0; i < no_of_sequences; ++i) {
				bwa_seq_t *p = seqs + i;
				fwrite(&p->n_aln, 4, 1, stdout);
				if (p->n_aln) fwrite(p->aln, sizeof(bwt_aln1_t), p->n_aln, stdout);
			}
			#endif // STDOUT_BINARY_RESULT == 1

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

	fprintf(stderr, "Barracuda, Version 0.5.1 beta\n");
	opt = gap_init_opt();
	while ((c = getopt(argc, argv, "s:a:C:n:o:e:i:d:l:k:cLR:m:t:NM:O:E:rf")) >= 0) {
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
		case 'r': cuda_opt = 2; break;
		case 'f': cuda_opt = 1; break;
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
		fprintf(stderr, "         -o INT  maximum number or fraction of gap opens [default: %d], [BWA default: 1]\n", opt->max_gapo);
		fprintf(stderr, "         -e INT  maximum number of gap extensions, -1 for disabling long gaps [default: -1]\n");
		fprintf(stderr, "         -i INT  do not put an indel within INT bp towards the ends [default: %d]\n", opt->indel_end_skip);
		fprintf(stderr, "         -d INT  maximum occurrences for extending a long deletion [default: %d]\n", opt->max_del_occ);
//		fprintf(stderr, "         -l INT  seed length [%d]\n", opt->seed_len);
//		fprintf(stderr, "         -k INT  maximum differences in the seed [%d]\n", opt->max_seed_diff);
		fprintf(stderr, "         -m INT  maximum no of loops/entries for matching [default: 250K for CUDA, 2M for BWA]\n");
		fprintf(stderr, "         -M INT  mismatch penalty [default: %d]\n", opt->s_mm);
		fprintf(stderr, "         -O INT  gap open penalty [de	fault: %d]\n", opt->s_gapo);
		fprintf(stderr, "         -E INT  gap extension penalty [default: %d]\n", opt->s_gape);
		fprintf(stderr, "         -R INT  stop searching when >INT equally best hits are found [default: %d]\n", opt->max_top2);
		fprintf(stderr, "         -c      reverse by not complement input sequences for color space reads\n");
		fprintf(stderr, "         -L      log-scaled gap penalty for long deletions\n");
		fprintf(stderr, "         -N      non-iterative mode: search for all n-difference hits.\n");
		fprintf(stderr, "                 CAUTION this is extremely slow\n");
		fprintf(stderr, "         -s INT  Skip the first INT bases (MID Tag) for alignment.\n");
		fprintf(stderr, "         -t INT  revert to original BWA with INT threads [default: %d]\n", opt->n_threads);
		fprintf(stderr, "                 cannot use with -C, -a, -f or -r\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "CUDA only options:\n");
		fprintf(stderr, "         -C INT  Specify which CUDA device to use. [default: auto-detect] \n");
		fprintf(stderr, "         -a INT  maximum number of alignments on each strand, max: 20 [default: %d]\n", opt->max_aln);
		fprintf(stderr, "         -f      single strand mode: align the forward strand only\n");
		fprintf(stderr, "         -r      single strand mode: align the reverse complementary strand only\n");


		fprintf(stderr, "\n");
		return 1;
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
void bwa_deviceQuery()
{
	// Detect CUDA devices available on the machine
	int device, num_devices;
	cudaGetDeviceCount(&num_devices);

	if (num_devices)
		{
			  fprintf(stderr,"[deviceQuery] Querying CUDA devices:\n");
			  for (device = 0; device < num_devices; device++)
			  {
					  cudaDeviceProp properties;
					  cudaGetDeviceProperties(&properties, device);
					  fprintf(stdout, "%d ", device);
					  fprintf(stdout, "%d %d%d\n", int(properties.totalGlobalMem/1048576), int(properties.major),  int(properties.minor));

			  }
			  fprintf(stderr,"[total] %d\n", device);
		}
//cudaSetDevice(0);
	return;
}
