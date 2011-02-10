#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bwtgap.h"
#include "bwtaln.h"

#define STDOUT_RESULT 0
#define USE_STACK_TABLE 0
#define MAX_NO_OF_GAP_ENTRIES 512

gap_stack_t *gap_init_stack(int max_mm, int max_gapo, int max_gape, const gap_opt_t *opt)
{
	int i;
	gap_stack_t *stack;
	stack = (gap_stack_t*)calloc(1, sizeof(gap_stack_t));
	stack->n_stacks = aln_score(max_mm+1, max_gapo+1, max_gape+1, opt);
	stack->stacks = (gap_stack1_t*)calloc(stack->n_stacks, sizeof(gap_stack1_t));
	for (i = 0; i != stack->n_stacks; ++i) {
		gap_stack1_t *p = stack->stacks + i;
		p->m_entries = 4;
		p->stack = (gap_entry_t*)calloc(p->m_entries, sizeof(gap_entry_t));
	}
	return stack;
}


void gap_destroy_stack(gap_stack_t *stack)
{
	int i;
	for (i = 0; i != stack->n_stacks; ++i) free(stack->stacks[i].stack);
	free(stack->stacks);
	free(stack);
}

static void gap_reset_stack(gap_stack_t *stack)
{
	int i;
	for (i = 0; i != stack->n_stacks; ++i)
		stack->stacks[i].n_entries = 0;
	stack->best = stack->n_stacks;
	stack->n_entries = 0;
}


static inline void gap_push(gap_stack_t *stack, int a, int i, bwtint_t k, bwtint_t l, int n_mm, int n_gapo, int n_gape,
							int state, int is_diff, const gap_opt_t *opt)
{

	int score;
	gap_entry_t *p;
	gap_stack1_t *q;
	score = aln_score(n_mm, n_gapo, n_gape, opt);
	q = &stack->stacks[score];
	if (q->n_entries == q->m_entries) {
		q->m_entries +=1;// <<= 1;
		q->stack = (gap_entry_t*)realloc(q->stack, sizeof(gap_entry_t) * q->m_entries);
	}
	p = q->stack + q->n_entries;
	p->length = i ; p->k = k; p->l = l;
	p->n_mm = n_mm; p->n_gapo = n_gapo; p->n_gape = n_gape; p->state = (a<<7)|state;
	if (is_diff) p->last_diff_pos = i;
	++(q->n_entries);
	++(stack->n_entries);
	if (stack->best > score) stack->best = score;

}


static inline void gap_pop(gap_stack_t *stack, gap_entry_t *e)
{
	gap_stack1_t *q;
	q = &stack->stacks[stack->best];
	*e = q->stack[q->n_entries - 1];
	--(q->n_entries);
	--(stack->n_entries);
	if (q->n_entries == 0 && stack->n_entries) { // reset best
		int i;
		for (i = stack->best + 1; i < stack->n_stacks; ++i)
			if (stack->stacks[i].n_entries != 0) break;
		stack->best = i;
	} else if (stack->n_entries == 0) stack->best = stack->n_stacks;
}

static inline void gap_shadow(int x, int len, bwtint_t max, int last_diff_pos, bwt_width_t *w)
{
	int i, j;
	for (i = j = 0; i < last_diff_pos; ++i) {
		if (w[i].w > x) w[i].w -= x;
		else if (w[i].w == x) {
			w[i].bid = 1;
			w[i].w = max - (++j);
		} // else should not happen
	}
}

static inline int int_log2(uint32_t v)
{
	int c = 0;
	if (v & 0xffff0000u) { v >>= 16; c |= 16; }
	if (v & 0xff00) { v >>= 8; c |= 8; }
	if (v & 0xf0) { v >>= 4; c |= 4; }
	if (v & 0xc) { v >>= 2; c |= 2; }
	if (v & 0x2) c |= 1;
	return c;
}

bwt_aln1_t *bwt_match_gap(bwt_t *const bwts[2], int len, const ubyte_t *seq[2], bwt_width_t *w[2], bwt_width_t *seed_w[2], const gap_opt_t *opt, int *_n_aln, gap_stack_t *stack)
{
	int best_score = aln_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
	int best_diff = opt->max_diff + 1, max_diff = opt->max_diff;
	int best_cnt = 0;
	//int max_entries = 0;
	int j, _j, n_aln, m_aln;
	bwt_aln1_t *aln;



	m_aln = 4; n_aln = 0;
	aln = (bwt_aln1_t*)calloc(m_aln, sizeof(bwt_aln1_t));

	//printf("Forward Sequence:");
	// check whether there are too many N
	for (j = _j = 0; j < len; ++j)
	{
		if (seq[0][j] > 3) ++_j;
	//	printf("%d", seq[0][j]);
	}
//	printf("\nReverse Sequence:");

	//for (j = 0; j < len; ++j)
//	{
	//	printf("%d", seq[1][j]);

//	}
	//printf("\n");
	if (_j > max_diff) {
		*_n_aln = n_aln;
		return aln;
	}

	//printf("max diff: %d\n", max_diff);
	//for (j = 0; j != len; ++j) printf("#0 %d: [%d,%u]\t[%d,%u]\n", j, w[0][j].bid, w[0][j].w, w[1][j].bid, w[1][j].w);
	gap_reset_stack(stack); // reset stack
	gap_push(stack, 0, len, 0, bwts[0]->seq_len, 0, 0, 0, 0, 0, opt);
	gap_push(stack, 1, len, 0, bwts[0]->seq_len, 0, 0, 0, 0, 0, opt);

	int loop_count = 0; //debug only
	while (stack->n_entries)
	{
		gap_entry_t e;
		int a, i, m, m_seed = 0, hit_found, allow_diff, allow_M, tmp;
		bwtint_t k, l, cnt_k[4], cnt_l[4], occ;
		const bwt_t *bwt;
		const ubyte_t *str;
		const bwt_width_t *seed_width = 0;
		bwt_width_t *width;
		loop_count ++;
		int worst_tolerated_score = best_score + opt->s_mm;
//		printf("best score %d, worst tolerated score %d\n", best_score, worst_tolerated_score);
//		printf("Entering loop %d, no of entries %d\n", loop_count, stack->n_entries); //debug only

		if (stack->n_entries > opt->max_entries) break;

//		if (stack->n_entries > max_entries) max_entries = stack->n_entries;

		gap_pop(stack, &e); // get the besqt entry

		k = e.k; l = e.l; // SA interval
		a = e.state>>7; i = e.length; // strand, length


		int score = aln_score(e.n_mm, e.n_gapo, e.n_gape, opt);

		if (!(opt->mode & BWA_MODE_NONSTOP) && score > worst_tolerated_score) break; // no need to proceed


//		printf("\nParent_1,");
//		printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, i, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);

		m = max_diff - (e.n_mm + e.n_gapo);
		if (opt->mode & BWA_MODE_GAPE) m -= e.n_gape;
		if (m < 0) continue;
		bwt = bwts[1-a]; str = seq[a]; width = w[a];

		if (seed_w) { // apply seeding
			seed_width = seed_w[a];
			m_seed = opt->max_seed_diff - (e.n_mm + e.n_gapo);
			if (opt->mode & BWA_MODE_GAPE) m_seed -= e.n_gape;
		}
		//printf("#1\t[%d,%d,%d,%c]\t[%d,%d,%d]\t[%u,%u]\t[%u,%u]\t%d\n", stack->n_entries, a, i, "MID"[e.state], e.n_mm, e.n_gapo, e.n_gape, width[i-1].bid, width[i-1].w, k, l, e.last_diff_pos);
		if (i > 0 && m < width[i-1].bid)
		{
			continue;
		}

		// check whether a hit is found
		hit_found = 0;
		if (i == 0) hit_found = 1;
		else if (m == 0 && ((e.state&0x7F) == STATE_M || (opt->mode&BWA_MODE_GAPE) || e.n_gape == opt->max_gape)) { // no diff allowed
			if (bwt_match_exact_alt(bwt, i, str, &k, &l)) hit_found = 1;
			else
			{
				continue; // no hit, skip
			}
		}

		if (hit_found) { // action for found hits

			int do_add = 1;
			//fprintf(stderr,"#2 hits found: %d:(%u,%u)\n", e.n_mm+e.n_gapo, k, l);
			//printf("#2 hits found: %d:(%u,%u)\n", e.n_mm+e.n_gapo, k, l);
			//printf("max_diff before %d,", max_diff);
			if (n_aln == 0) {
				best_score = score;
				best_diff = e.n_mm + e.n_gapo;
				if (opt->mode & BWA_MODE_GAPE) best_diff += e.n_gape;
				if (!(opt->mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;
			else if (best_cnt > opt->max_top2)
				{
					break; // top2b behaviour
				}
			if (e.n_gapo) { // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array
				for (j = 0; j != n_aln; ++j)
					if (aln[j].k == k && aln[j].l == l)
						{
							break;
						}
				if (j < n_aln) do_add = 0;
			}

			if (do_add) { // append result the alignment record array

				bwt_aln1_t *p;
				gap_shadow(l - k + 1, len, bwt->seq_len, e.last_diff_pos, width);
				if (n_aln == m_aln) {
					m_aln <<= 1;
					aln = (bwt_aln1_t*)realloc(aln, m_aln * sizeof(bwt_aln1_t));
					memset(aln + m_aln/2, 0, m_aln/2*sizeof(bwt_aln1_t));
				}
				p = aln + n_aln;
				// record down number of mismatch, gap open, gap extension and a??
				p->n_mm = e.n_mm; p->n_gapo = e.n_gapo; p->n_gape = e.n_gape; p->a = a;
				// the suffix array interval
				p->k = k; p->l = l;
				// the score as a alignment record
				p->score = score;
				++n_aln;
			}
			continue;
		}

		--i;
		bwt_2occ4(bwt, k - 1, l, cnt_k, cnt_l); // retrieve Occurrence values
		occ = l - k + 1;

		// test whether difference is allowed
		allow_diff = allow_M = 1;

		if (i > 0) {
			int ii = i - (len - opt->seed_len);
			if (width[i-1].bid > m-1) allow_diff = 0;
			else if (width[i-1].bid == m-1 && width[i].bid == m-1 && width[i-1].w == width[i].w) allow_M = 0;

			if (seed_w && ii > 0) {
				if (seed_width[ii-1].bid > m_seed-1) allow_diff = 0;
				else if (seed_width[ii-1].bid == m_seed-1 && seed_width[ii].bid == m_seed-1
						 && seed_width[ii-1].w == seed_width[ii].w) allow_M = 0;
			}
		}

		// insertion and deletions
		tmp = (opt->mode & BWA_MODE_LOGGAP)? int_log2(e.n_gape + e.n_gapo)/2+1 : e.n_gapo + e.n_gape;
		if (allow_diff && i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp) {
			if ((e.state&0x7F) == STATE_M) { // gap open
				if (e.n_gapo < opt->max_gapo) { // gap open is allowed
				// insertion
//					printf("\nParent,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//					printf("daughter,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_I);
					//if((score + opt->s_gapo+1 <= worst_tolerated_score) ||(opt->mode & BWA_MODE_NONSTOP))
					gap_push(stack, a, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_I, 1, opt);

					// deletion
					for (j = 0; j != 4; ++j) {
						k = bwt->L2[j] + cnt_k[j] + 1;
						l = bwt->L2[j] + cnt_l[j];
						if (k <= l)
						{
//							printf("\nParent,");
//							printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//							printf("daughter,");
//							printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",1, i, k, l, e.n_mm, e.n_gapo+1, e.n_gape, STATE_D);
						//	if((score + opt->s_gapo <= worst_tolerated_score) ||(opt->mode & BWA_MODE_NONSTOP))
							gap_push(stack, a, i + 1, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_D, 1, opt);
						}
					}
				}
			} else if ((e.state&0x7F) == STATE_I) { // Extension of an insertion
				if (e.n_gape < opt->max_gape) // gap extension is allowed
				{
//					printf("\nParent,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//					printf("daughter,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo, e.n_gape+1, STATE_I);
					//if((score + opt->s_gape <= worst_tolerated_score) ||(opt->mode & BWA_MODE_NONSTOP))
					gap_push(stack, a, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, STATE_I, 1, opt);
				}

			} else if ((e.state&0x7F) == STATE_D) { // Extension of a deletion
				if (e.n_gape < opt->max_gape) { // gap extension is allowed
					if (e.n_gape + e.n_gapo < max_diff || occ < opt->max_del_occ) {
						for (j = 0; j != 4; ++j) {
							k = bwt->L2[j] + cnt_k[j] + 1;
							l = bwt->L2[j] + cnt_l[j];
							if (k <= l)
							{
//								printf("\nParent,");
//								printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//								printf("daughter,");
//								printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo, e.n_gape+1, STATE_D);
								//if((score + opt->s_gape <= worst_tolerated_score) ||(opt->mode & BWA_MODE_NONSTOP))
								gap_push(stack, a, i + 1, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, STATE_D, 1, opt);
							}
						}
					}
				}
			}
		}

		// mismatches
		if (allow_diff && allow_M)
		{ // mismatch is allowed
			for (j = 1; j <= 4; ++j) {
				int c = (str[i] + j) & 3;
				int is_mm = (j != 4 || str[i] > 3);
				k = bwt->L2[c] + cnt_k[c] + 1;
				l = bwt->L2[c] + cnt_l[c];
				if (k <= l)
				{
//					printf("\nParent,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//					printf("daughter,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, STATE_M);
					if(((score + is_mm * opt->s_mm) <= worst_tolerated_score) ||(opt->mode & BWA_MODE_NONSTOP))
					gap_push(stack, a, i, k, l, e.n_mm + is_mm, e.n_gapo, e.n_gape, STATE_M, is_mm, opt); //these pushes four times?
				}

			}
		} else if (str[i] < 4) { // try exact match only

			int c = str[i] & 3;
			k = bwt->L2[c] + cnt_k[c] + 1;
			l = bwt->L2[c] + cnt_l[c];
			if (k <= l)
			{
//				printf("\nParent,");
//				printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, e.length, e.k, e.l, e.n_mm, e.n_gapo, e.n_gape, e.state&0x7F);
//				printf("daughter,");
//				printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo, e.n_gape,  STATE_M);
				gap_push(stack, a, i, k, l, e.n_mm, e.n_gapo, e.n_gape, STATE_M, 0, opt);
			}

		}
	}
	*_n_aln = n_aln;
//	printf("max_entries %d:\n", max_entries);
//	printf("loop count: %d\n", loop_count);
	return aln;
}

//////////////////////////////////////////////////////////////////////////////////////////
//DFS MATCH FUNCTIONS GETS IN HERE - FOR USE WITH DEVELOPMENT ONLY
//////////////////////////////////////////////////////////////////////////////////////////

typedef struct
{
	gap_entry_t entries[MAX_SEQUENCE_LENGTH];
} entry_stack;

typedef struct
{
	int x;
	int y;
	int z;
	int w;
} char_4;
void dfs_initialize(entry_stack *stack, char_4 *done_push_types)
{
	int i, j;

	for (i = 0; i < MAX_SEQUENCE_LENGTH; i++)
	{
		stack->entries[i].k = 0;
		stack->entries[i].l = 0;
		stack->entries[i].last_diff_pos = 0;
		stack->entries[i].length = 0;
		stack->entries[i].n_gape = 0;
		stack->entries[i].n_gapo = 0;
		stack->entries[i].n_mm = 0;
		stack->entries[i].state = 0;
		for (j = 0; j < 4; j++)
		{
			done_push_types[i].x = 0;
			done_push_types[i].y = 0;
			done_push_types[i].z = 0;
			done_push_types[i].w = 0;
		}
	}

}

void dfs_push(entry_stack *stack, char_4 *done_push_types, int i, unsigned int k, unsigned int l, int n_mm, int n_gapo, int n_gape, int state, int is_diff, int current_stage)
{
	int j;
	if(current_stage < MAX_NO_OF_GAP_ENTRIES)
	{
		stack->entries[current_stage].k = k;
		stack->entries[current_stage].l = l;
		stack->entries[current_stage].length = i;
		stack->entries[current_stage].n_gape = n_gape;
		stack->entries[current_stage].n_gapo = n_gapo;
		stack->entries[current_stage].n_mm = n_mm;
		stack->entries[current_stage].state = state;
		if (is_diff) stack->entries[current_stage].last_diff_pos = i;
	}
	for (j = 0; j < 4; j++)
	{
		done_push_types[current_stage].x = 0;
		done_push_types[current_stage].y = 0;
		done_push_types[current_stage].z = 0;
		done_push_types[current_stage].w = 0;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////


void dfs_match(bwt_t *const bwts[2], int len, const ubyte_t *seq[2], bwt_width_t *w[2], bwt_width_t *seed_w[2], const gap_opt_t *opt, alignment_store_t *aln)
{
	int best_score = aln_score(opt->max_diff+1, opt->max_gapo+1, opt->max_gape+1, opt);
	int best_diff = opt->max_diff + 1, max_diff = opt->max_diff;
	int best_cnt = 0;
	int j,n_aln;
	int loop_count = 0;
	entry_stack stack;
	//int sequence_type = 0;	// 0 = seq; 1 = rseq;
	n_aln = 0;
	//int _j;
	int current_stage = 0;
	char_4 done_push_types[MAX_SEQUENCE_LENGTH];
	/*
	// check whether there are too many Ns
//	printf("now working on sequence ");
	for (j = _j = 0; j < len; ++j)
	{
	//	printf("%u", seq[0][j]);
		if (seq[0][j] > 3) ++_j;
	}
//	printf("\n");
	if (_j > max_diff) {
		n_aln = 0;
		aln->no_of_alignments = n_aln;
		return;
	}
*/
	dfs_initialize(&stack, done_push_types); //initialize initial entry, current stage set at 0 and done push type = 0
	dfs_push(&stack, done_push_types, len, 0, bwts[0]->seq_len, 0, 0, 0, 0, 0, current_stage); //push initial entry to start

	while(current_stage >= 0)
	{

		gap_entry_t e;
		int a, i, m;
		//int m_seed = 0;
		int hit_found, allow_diff, allow_M;
		bwtint_t k, l, cnt_k[4], cnt_l[4], occ;
		const bwt_t *bwt;
		const ubyte_t *str;
		//const bwt_width_t *seed_width = 0;
		bwt_width_t *width;
		loop_count ++;
		int worst_tolerated_score = best_score + opt->s_mm + (opt->mode & BWA_MODE_NONSTOP) * 1000;

		//define break from loop conditions
		if (loop_count > 250000) break; //if max loop counts, i.e. 2000000 reached, break
		if (n_aln == opt->max_aln)break;
		if (best_cnt > opt->max_top2) break;

		e = stack.entries[current_stage];// pop entry from latest stage

		//put extracted entry into local variables
		k = e.k; l = e.l; // SA interval
		i = e.length; // strand, length
		a = 0;
//		printf("\nEntering stage: %d, Loop no: %d, worst_score: %d,  ", current_stage, loop_count, worst_tolerated_score);
//		printf("pushes, x: %d, y: %d, z: %d, w: %d,", done_push_types[current_stage].x, done_push_types[current_stage].y, done_push_types[current_stage].z, done_push_types[current_stage].w);
//		printf("parent,");
//		printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d,",a, i, k, l, e.n_mm, e.n_gapo, e.n_gape, e.state);


		int score = aln_score(e.n_mm, e.n_gapo, e.n_gape, opt);

		if (score > worst_tolerated_score)
		{
			break;						// no need to proceed this branch
		}

		m = max_diff - (e.n_mm + e.n_gapo);

		if (opt->mode & BWA_MODE_GAPE) m -= e.n_gape;

		if (m < 0)
		{
			current_stage --;
			continue;
		}

		bwt = bwts[1-a]; str = seq[a]; width = w[a];

/*
		if (seed_w) { // apply seeding
			seed_width = seed_w[a];
			m_seed = opt->max_seed_diff - (e.n_mm + e.n_gapo);
			if (opt->mode & BWA_MODE_GAPE) m_seed -= e.n_gape;
		}*/

		if (i > 0 && m < width[i-1].bid)
		{
			current_stage --;
			continue;
		}

		// check whether a hit is found
		hit_found = 0;
		if (i == 0) hit_found = 1;
		else if (m == 0 && (e.state == STATE_M || (opt->mode&BWA_MODE_GAPE) || e.n_gape == opt->max_gape))
		{ // no diff allowed
			if (bwt_match_exact_alt(bwt, i, str, &k, &l)) hit_found = 1;
			else
			{
				current_stage --;
				continue; // no hit, skip
			}
		}

		if (hit_found)
		{
			// action for found hits

			int do_add = 1;
//			printf("#2 hits found: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!%d:(%u,%u)\n", e.n_mm+e.n_gapo, k, l);
			if (score < best_score)
			{
				best_score = score;
				best_diff = e.n_mm + e.n_gapo + (opt->mode & BWA_MODE_GAPE) * e.n_gape;
				best_cnt = 0; //reset best cnt if new score is better
				if (!(opt->mode & BWA_MODE_NONSTOP))
					max_diff = (best_diff + 1 > opt->max_diff)? opt->max_diff : best_diff + 1; // top2 behaviour
			}
			if (score == best_score) best_cnt += l - k + 1;

			if (e.n_gapo)
			{ // check whether the hit has been found. this may happen when a gap occurs in a tandem repeat
				// if this alignment was already found, do not add to alignment record array
				for (j = 0; j != n_aln; ++j)
					if (aln->alignment_info[j].k == k && aln->alignment_info[j].l == l) break;
				if (j < n_aln)
				{
					if (score < aln->alignment_info[j].score)
						{
							aln->alignment_info[j].score = score;
							aln->alignment_info[j].n_mm = e.n_mm;
							aln->alignment_info[j].n_gapo = e.n_gapo;
							aln->alignment_info[j].n_gape = e.n_gape;
						}
					do_add = 0;
				}
			}

			if (do_add)
			{ // append result the alignment record array
				gap_shadow(l - k + 1, len, bwt->seq_len, e.last_diff_pos, width);
				bwt_aln1_t * p;

				if (n_aln < MAX_NO_OF_ALIGNMENTS)
				{
					p = &aln->alignment_info[n_aln];
					// record down number of mismatch, gap open, gap extension and a??
					p->n_mm = e.n_mm;
					p->n_gapo = e.n_gapo;
					p->n_gape = e.n_gape;
					p->a = a;
					// the suffix array interval
					p->k = k;
					p->l = l;
					// the score as a alignment record
					p->score = score;
					++n_aln;
				}
			}

			current_stage --;
			continue;
		}

		--i;

		// retrieve Occurrence values and determine all the eligible daughter nodes, done only once at the first instance and skip when it is revisiting the stage
		unsigned int ks[MAX_SEQUENCE_LENGTH][4], ls[MAX_SEQUENCE_LENGTH][4];
		int eligible_cs[MAX_SEQUENCE_LENGTH][5], no_of_eligible_cs=0;

		if(!done_push_types[current_stage].x)
		{
			//uint4 cuda_cnt_k = (!sequence_type)? rbwt_cuda_occ4(k-1): bwt_cuda_occ4(k-1);
			//uint4 cuda_cnt_l = (!sequence_type)? rbwt_cuda_occ4(l): bwt_cuda_occ4(l);
			bwt_2occ4(bwt, k - 1, l, cnt_k, cnt_l); // retrieve Occurrence values

			ks[current_stage][0] = bwt->L2[0] + cnt_k[0] + 1;
			ls[current_stage][0] = bwt->L2[0] + cnt_l[0];
			ks[current_stage][1] = bwt->L2[1] + cnt_k[1] + 1;
			ls[current_stage][1] = bwt->L2[1] + cnt_l[1];
			ks[current_stage][2] = bwt->L2[2] + cnt_k[2] + 1;
			ls[current_stage][2] = bwt->L2[2] + cnt_l[2];
			ks[current_stage][3] = bwt->L2[3] + cnt_k[3] + 1;
			ls[current_stage][3] = bwt->L2[3] + cnt_l[3];

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
		allow_diff = allow_M = 1;


		if (i > 0) {
			//int ii = i - (len - opt->seed_len);
			if (width[i-1].bid > m-1) allow_diff = 0;
			else if (width[i-1].bid == m-1 && width[i].bid == m-1 && width[i-1].w == width[i].w) allow_M = 0;

			/*if (seed_w && ii > 0) {
				if (seed_width[ii-1].bid > m_seed-1) allow_diff = 0;
				else if (seed_width[ii-1].bid == m_seed-1 && seed_width[ii].bid == m_seed-1
						 && seed_width[ii-1].w == seed_width[ii].w) allow_M = 0;
			}*/
		}

		//donepushtypes store information for each stage whether a prospective daughter node has been evaluated or not
		//donepushtypes[current_stage].x  exact match, =0 not done, =1 done
		//donepushtypes[current_stage].y  mismatches, 0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].z  deletions, =0 not done, =no of eligible cs with a k<=l done
		//donepushtypes[current_stage].w  insertions match, =0 not done, =1 done
		//.z and .w are shared among gap openings and extensions as they are mutually exclusive

//		printf("no of k<=ls: %d, ", no_of_eligible_cs);
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// exact match
		////////////////////////////////////////////////////////////////////////////////////////////////////////////

//		printf("m: %d, diff: %d, ",allow_M, allow_diff);
		//try exact match first
		if (!done_push_types[current_stage].x)
		{
			done_push_types[current_stage].x = 1;
			if (str[i] < 4)
			{
				int c = str[i];
				if (ks[current_stage][c] <= ls[current_stage][c])
				{
					dfs_push(&stack, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo, e.n_gape, STATE_M, 0, current_stage+1);
//					printf("daughter,");
//					printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo, e.n_gape,  STATE_M);
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
				if (allow_M) // daughter node - mismatch
				{
					if (score + opt->s_mm <= worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
					{
						int c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
						done_push_types[current_stage].y++;
						if (c != str[i])
						{
							dfs_push(&stack, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e.n_mm + 1, e.n_gapo, e.n_gape, STATE_M, 1, current_stage+1);
//							printf("daughter,");
//							printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, ks[current_stage][c], ls[current_stage][c], e.n_mm + 1, e.n_gapo, e.n_gape,  STATE_M);
							current_stage++;
							continue;
						}else if (done_push_types[current_stage].y < no_of_eligible_cs)
						{
							c = eligible_cs[current_stage][(done_push_types[current_stage].y)];
							done_push_types[current_stage].y++;
							dfs_push(&stack, done_push_types, i, ks[current_stage][c], ls[current_stage][c], e.n_mm + 1, e.n_gapo, e.n_gape, STATE_M, 1, current_stage+1);
//							printf("daughter,");
//							printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, ks[current_stage][c], ls[current_stage][c], e.n_mm + 1, e.n_gapo, e.n_gape,  STATE_M);
							current_stage++;
							continue;
						}
					}
				}
			}


			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Indels (Insertions/Deletions)
			////////////////////////////////////////////////////////////////////////////////////////////////////////////
			if (e.state == STATE_M) // daughter node - opening a gap insertion or deletion
			{
				if (score + opt->s_gapo <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
				{
					if (e.n_gapo < opt->max_gapo)
					{
						unsigned int tmp = (opt->mode & BWA_MODE_LOGGAP)? ((int_log2(e.n_gape + e.n_gapo)) >> 1) + 1 : e.n_gapo + e.n_gape;

						if (i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp)
						{
							//insertions
							if (!done_push_types[current_stage].w)  //check if done before
							{
								done_push_types[current_stage].w = 1;
								dfs_push(&stack, done_push_types, i, k, l, e.n_mm, e.n_gapo + 1, e.n_gape, STATE_I, 1, current_stage+1);
//								printf("daughter,");
//								printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo+1, e.n_gape,  STATE_I);
								current_stage++;
								continue;
							}
							//deletions
							else if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
							{
								int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
								done_push_types[current_stage].z++;
								dfs_push(&stack, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo + 1, e.n_gape, STATE_D, 1, current_stage+1);
//								printf("daughter,");
//								printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i + 1, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo+1, e.n_gape,  STATE_D);
								current_stage++; //advance stage number by 1
								continue;
							}
						}
					}
				}
			}else if (e.state == STATE_I) //daughter node - extend an insertion entry
			{
				if(!done_push_types[current_stage].w)  //check if done before
				{
					done_push_types[current_stage].w = 1;
					if (e.n_gape < opt->max_gape)  //skip if no of gap ext is beyond limit
					{
						if (score + opt->s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
						{
							unsigned int tmp = (opt->mode & BWA_MODE_LOGGAP)? ((int_log2(e.n_gape + e.n_gapo)) >> 1) + 1 : e.n_gapo + e.n_gape;
							if (i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp)
							{
								dfs_push(&stack, done_push_types, i, k, l, e.n_mm, e.n_gapo, e.n_gape + 1, STATE_I, 1, current_stage+1);
//								printf("daughter,");
//								printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i, k, l, e.n_mm, e.n_gapo, e.n_gape+1,  STATE_I);
								current_stage++; //advance stage number by 1
								continue; //skip the rest and proceed to next stage
							}
						}
					}
				}
			}else if (e.state == STATE_D) //daughter node - extend a deletion entry
			{
				occ = l - k + 1;
				if (done_push_types[current_stage].z < no_of_eligible_cs)  //check if done before
				{
					if (e.n_gape < opt->max_gape) //skip if no of gap ext is beyond limit
					{
						if (score + opt->s_gape <=worst_tolerated_score) //skip if prospective entry is beyond worst tolerated
						{
							if (e.n_gape + e.n_gapo < max_diff || occ < opt->max_del_occ)
							{
								unsigned int tmp = (opt->mode & BWA_MODE_LOGGAP)? ((int_log2(e.n_gape + e.n_gapo)) >> 1) + 1 : e.n_gapo + e.n_gape;

								if (i >= opt->indel_end_skip + tmp && len - i >= opt->indel_end_skip + tmp)
								{
									int c = eligible_cs[current_stage][(done_push_types[current_stage].z)];
									done_push_types[current_stage].z++;
									dfs_push(&stack, done_push_types, i + 1, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo, e.n_gape + 1, STATE_D, 1, current_stage+1);
//									printf("daughter,");
//									printf("a: %d i: %d k: %u l: %u n_mm %d %d %d %d",a, i + 1, ks[current_stage][c], ls[current_stage][c], e.n_mm, e.n_gapo, e.n_gape + 1,  STATE_D);
									current_stage++; //advance stage number
									continue;
								}
							}
						}
					}
				}
			} //end else if (e.state == STATE_D)

		}//end if (!allow_diff)
		//printf("no daughter");
		current_stage--;

	} //end do while loop

	//printf("loop count: %d\n", loop_count);
	aln->no_of_alignments = n_aln;

	return;
}

