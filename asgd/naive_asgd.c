#include "naive_asgd.h"

#include <cblas.h>

matrix_t *matrix_init(
		order_t order,
		size_t rows,
		size_t cols,
		size_t ld)
{
	matrix_t *m = malloc(sizeof(*m));
	size_t major = order == row_major ? rows : cols;
	m->data = calloc(major*ld, sizeof(*m->data));
	return m;
}

void matrix_destr(matrix_t *m)
{
	free(m->data);
	free(m);
}

float matrix_get(matrix_t *m, size_t i, size_t j)
{
	if (__builtin_expect(m->order == row_major, true))
	{
		return m->data[i*m->ld + j];
	}
	else
	{
		return m->data[j*m->ld + i];
	}
}

void matrix_set(matrix_t *m, size_t i, size_t j, float val)
{
	if (__builtin_expect(m->order == row_major, true))
	{
		m->data[i*m->ld + j] = val;
	}
	else
	{
		m->data[j*m->ld + i] = val;
	}
}

float *matrix_row(matrix_t *m, size_t i)
{
	if (__builtin_expect(m->order == row_major, true))
	{
		return m->data + i*m->ld;
	}
	else
	{
		// TODO
		// pick all items in i-th vertical?
		return NULL;
	}
}

void matrix_copy(matrix_t *dst, const matrix_t *src)
{
	dst->order = src->order;
	dst->rows = src->rows;
	dst->cols = src->cols;
	dst->ld = src->ld;
	
	size_t major = src->order == row_major ? src->rows : src->cols;
	memcpy(dst->data, src->data, major * src->ld * sizeof(*src->data));
}

matrix_t *matrix_clone(matrix_t *m)
{
	matrix_t *r = malloc(sizeof(*r));
	memcpy(r, m, sizeof(*m));
	size_t major = m->order == row_major ? m->rows : m->cols;
	r->data = malloc(major * m->ld * sizeof(*m->data));
	memcpy(r->data, r->data, major * m->ld * sizeof(*m->data));
	return r;
}

static void swap(matrix_t *m, size_t j, size_t k, size_t x, size_t y)
{
	float buff = matrix_get(m, j, k);
	matrix_set(m, j, k, matrix_get(m, x, y));
	matrix_set(m, x, y, buff);
}

/**
 * Exists with an error unless a given condition is true
 * @param cond The condition to check
 * @param mex A string to print if the condition is false
 */
void mex_assert(bool cond, const char *mex)
{
	if (__builtin_expect(!cond, false)) {
		fprintf(stderr, "%s\n", mex);
		exit(EXIT_FAILURE);
	}
}

/**
 * Compute the inner product of two vectors
 * @param a The first vector
 * @param b The second vector
 * @param c The length of the vectors
 */
/*float inner_product(
		float *a,
		float *b,
		size_t length)
{
	float sum = 0.0;
	while (length--) {
		sum += a[length] * b[length];
	}
	return sum;
}*/

/*void scalar_product(
		float s,
		matrix_t *m)
{
	for (uint i = 0; i < m->rows; ++i) {
		for (uint j=0; j < m->cols; ++j) {
			m->data[i*m->rows+m->cols] *= s;
		}
	}
}*/

/*void scalar_sum(
		float s,
		matrix_t *m)
{
	for (uint i = 0; i < m->rows; ++i) {
		for (uint j=0; j < m->cols; ++j) {
			m->data[i*m->rows+m->cols] += s;
		}
	}
}*/

static void durstenfeld_shuffle(matrix_t *m)
{
	srand(time(NULL));
	for (size_t i = m->rows-1; i > 0; --i) {
		size_t j = rand() % (i+1);
		// flip current row with a random row among remaining ones
		for (size_t k = 0; k < m->cols; ++i) {
			swap(m, i, k, j, k);
		}
	}
}

/**
 * Constructor for the Binary ASGD structure
 */
nb_asgd_t *nb_asgd_init(
	uint64_t n_feats,
	float sgd_step_size0,
	float l2_reg,
	uint64_t n_iters,
	bool feedback)
{
	nb_asgd_t *data = malloc(sizeof(*data));
	mex_assert(data != NULL, "cannot allocate nb_asgd");
	data->n_feats = n_feats;
	data->n_iters = n_iters;
	data->feedback = feedback;

	mex_assert(__builtin_expect(l2_reg > 0, true), "invalid l2 regularization");
	data->l2_reg = l2_reg;

	data->sgd_weights = matrix_init(row_major, n_feats, 1, sizeof(*data->sgd_weights->data));
	data->sgd_bias = matrix_init(row_major, 1, 1, sizeof(*data->sgd_weights->data));
	data->sgd_step_size = sgd_step_size0;
	data->sgd_step_size0 = sgd_step_size0;

	data->sgd_step_size_scheduling_exp = 2. / 3.;
	data->sgd_step_size_scheduling_mul = l2_reg;

	data->asgd_weights = matrix_init(row_major, n_feats, 1, sizeof(*data->asgd_weights->data));
	data->asgd_bias = matrix_init(row_major, 1, 1, sizeof(*data->asgd_weights->data));
	data->asgd_step_size = 1;
	data->asgd_step_size0 = 1;

	data->n_observs = 0;
	return data;
}

/**
 * Destructor for the Binary ASGD structure
 */
void nb_asgd_destr(
		nb_asgd_t *data)
{
	matrix_destr(data->sgd_weights);
	matrix_destr(data->sgd_bias);
	matrix_destr(data->asgd_weights);
	matrix_destr(data->asgd_bias);
	free(data);
}

void partial_fit(
		nb_asgd_t *data,
		matrix_t *X,
		matrix_t *y)
{

	for (size_t i = 0; i < X->rows; ++i) {
		
		// compute margin
		float margin = matrix_get(y, i, 1) * // TODO this will become a matrix
			cblas_sdsdot(X->cols, matrix_get(data->sgd_bias, 1, 1),
				matrix_row(X, i), sizeof(*X->data),
				matrix_row(data->sgd_weights, i), data->sgd_weights->ld);

		// update sgd
		if (data->l2_reg != 0)
		{
			// TODO this will become a matrix
			cblas_sscal(data->sgd_weights->rows,
					1 - data->l2_reg * data->sgd_step_size,
					data->sgd_weights->data, sizeof(*data->sgd_weights->data));
		}

		if (margin < 1)
		{
			// TODO this will become a matrix
			cblas_saxpy(X->rows,
					data->sgd_step_size * matrix_get(y, i, 1),
					X->data, sizeof(*X->data),
					data->sgd_weights->data, sizeof(*data->sgd_weights->data));
			
			// TODO this will become a vector
			matrix_set(data->sgd_bias, 1, 1,
					data->sgd_step_size * matrix_get(y, i, 1));
		}

		// update asgd
		matrix_t *asgd_weights = matrix_clone(data->asgd_weights);
		cblas_sscal(asgd_weights->rows,
				1 - data->asgd_step_size,
				asgd_weights->data, sizeof(*asgd_weights->data));
		cblas_saxpy(asgd_weights->rows,
				data->asgd_step_size,
				data->asgd_weights->data, sizeof(*data->asgd_weights->data),
				asgd_weights->data, sizeof(*asgd_weights->data));

		matrix_t *asgd_bias = matrix_clone(data->asgd_bias);
		matrix_set(asgd_bias, 1, 1,
				1 - data->asgd_step_size * matrix_get(asgd_bias, 1, 1) +
				data->asgd_step_size * matrix_get(data->sgd_bias, 1, 1));
		
		// update step_sizes
		data->n_observs += 1;
		float sgd_step_size_scheduling = 1 + data->sgd_step_size0 * data->n_observs
			* data->sgd_step_size_scheduling_mul;
		data->sgd_step_size = data->sgd_step_size0 /
			powf(sgd_step_size_scheduling, data->sgd_step_size_scheduling_exp);
		data->asgd_step_size = 1.0f / data->n_observs;

		matrix_copy(data->asgd_weights, asgd_weights);
		matrix_copy(data->asgd_bias, asgd_bias);

		matrix_destr(asgd_weights);
		matrix_destr(asgd_bias);
	}
}

void fit(
	nb_asgd_t *data,
	matrix_t *X,
	matrix_t *y)
{
	mex_assert(X->rows > 1, "fit: X should be a matrix");
	mex_assert(y->cols == 1, "fit: y should be a column vector");

	for (uint64_t i = 0; i < data->n_iters; ++i) {
		durstenfeld_shuffle(X);
		durstenfeld_shuffle(y);
		//partial_fit(data, X, y);

		if (data->feedback) {
			matrix_copy(data->sgd_weights, data->asgd_weights);
			matrix_copy(data->sgd_bias, data->asgd_bias);
		}
	}
}

float predict(
	nb_asgd_t *data,
	matrix_t *X)
{
	// TODO
	// find appropriate BLAS
}

