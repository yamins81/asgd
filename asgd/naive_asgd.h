#ifndef _NAIVE_ASGD_H_
#define _NAIVE_ASGD_H_

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum order {row_major, col_major};
typedef enum order order_t;

typedef struct matrix matrix_t;
struct matrix {
	order_t order; // row or col major
	size_t rows;
	size_t cols;
	size_t ld; // byte length of a major dim
	float *data;
};

matrix_t *matrix_init(order_t order, size_t rows, size_t cols, size_t ld);

void matrix_destr(matrix_t *m);

void matrix_copy(matrix_t *dst, const matrix_t *src);

matrix_t *matrix_clone(matrix_t *m);

float matrix_get(matrix_t *m, size_t i, size_t j);

void matrix_set(matrix_t *m, size_t i, size_t j, float val);

float *matrix_row(matrix_t *m, size_t i);

void mex_assert(bool cond, const char *mex);

typedef struct nb_asgd nb_asgd_t;
struct nb_asgd
{
	size_t n_feats;
	size_t n_iters;
	float l2_reg;
	bool feedback;

	matrix_t *sgd_weights;
	matrix_t *sgd_bias;
	float sgd_step_size;
	float sgd_step_size0;
	
	float sgd_step_size_scheduling_exp;
	float sgd_step_size_scheduling_mul;

	matrix_t *asgd_weights;
	matrix_t *asgd_bias;
	float asgd_step_size;
	float asgd_step_size0;

	uint64_t n_observs;
};

nb_asgd_t *nb_asgd_init(
	uint64_t n_feats,
	float sgd_step_size0,
	float l2_reg,
	uint64_t n_iters,
	bool feedback);

void nb_asgd_destr(
		nb_asgd_t *data);

#endif

