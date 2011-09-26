#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// type and size parameters
// will need to autoset them later on
typedef unsigned int uint;

typedef struct matrix matrix_t;
struct matrix {
	size_t rows;
	size_t cols;
	double *data;
};

void swap(double *a, double *b)
{
	double x = *a;
	*a = *b;
	*b = x;
}

/**
 * Exists with an error unless a given condition is true
 * @param cond The condition to check
 * @param mex A string to print if the condition is false
 */
void aassert(bool cond, const char *mex)
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
double inner_product(
		double *a,
		double *b,
		size_t length)
{
	double sum = 0.0;
	while (length--) {
		sum += a[length] * b[length];
	}
	return sum;
}

void scalar_product(
		double s,
		matrix_t *m)
{
	for (uint i = 0; i < m->rows; ++i) {
		for (uint j=0; j < m->cols; ++j) {
			m->data[i*m->rows+m->cols] *= s;
		}
	}
}

void scalar_sum(
		double s,
		matrix_t *m)
{
	for (uint i = 0; i < m->rows; ++i) {
		for (uint j=0; j < m->cols; ++j) {
			m->data[i*m->rows+m->cols] += s;
		}
	}
}

void durstenfeld_shuffle(matrix_t *matrix)
{
	srand(time(NULL));
	for (uint i = matrix->rows-1; i > 0; --i) {
		uint j = rand() % (i+1);
		// flip current row with a random row among remaining ones
		for (uint k = 0; k < matrix->cols; ++i) {
			swap(matrix->data+(matrix->cols*i+k), matrix->data+(matrix->cols*j+k));
		}
	}
}

typedef struct native_binary_asgd native_binary_asgd_t;
struct native_binary_asgd
{
	uint n_features;
	uint n_iterations;
	double l2_regularization;
	double sgd_step_size;
	double sgd_step_size0;
	bool feedback;

	matrix_t sgd_weights; // n_features long
	
	double sgd_bias;
	double sgd_step_size_scheduling_exponent;
	double sgd_step_size_scheduling_multiplier;

	matrix_t asgd_weights; // n_features long
	
	double asgd_bias;
	double asgd_step_size;
	double asgd_step_size0;

	uint n_observations;
};

/**
 * Constructor for the Binary ASGD structure
 */
void native_binary_asgd_init(
	native_binary_asgd_t *data,
	uint n_features,
	double sgd_step_size0,
	double l2_regularization,
	uint n_iterations,
	bool feedback)
{
	data->n_features = n_features;
	data->n_iterations = n_iterations;
	data->feedback = feedback;

	if (__builtin_expect(l2_regularization <= 0, false))
	{
		fprintf(stderr, "invalid l2_regularization, quitting");
		exit(EXIT_FAILURE);
	}
	data->l2_regularization = l2_regularization;

	data->sgd_weights.data = malloc(n_features*sizeof(*data->sgd_weights.data));
	aassert(data->sgd_weights.data != NULL, "could not allocate sgd_weights");
	data->sgd_weights.rows = 1;
	data->sgd_weights.cols = n_features;

	data->sgd_bias = 0;
	data->sgd_step_size = sgd_step_size0;
	data->sgd_step_size0 = sgd_step_size0;
	data->sgd_step_size_scheduling_exponent = 2. / 3.;
	data->sgd_step_size_scheduling_multiplier = l2_regularization;

	data->asgd_weights.data = malloc(n_features*sizeof(*data->asgd_weights.data));
	aassert(data->asgd_weights.data != NULL, "could not allocate asgd_weights");
	data->asgd_weights.rows = 1;
	data->asgd_weights.cols = n_features;

	data->asgd_bias = 0;
	data->asgd_step_size = 1;
	data->asgd_step_size0 = 1;

	data->n_observations = 0;
}

/**
 * Destructor for the Binary ASGD structure
 */
void native_binary_asgd_destr(
		native_binary_asgd_t *data)
{
	free(data->sgd_weights.data);
	free(data->asgd_weights.data);
}

void partial_fit(
		native_binary_asgd_t *data,
		matrix_t *X,
		matrix_t *y)
{
	matrix_t asgd_weights = data->asgd_weights;
	double asgd_bias = data->asgd_bias;
	uint asgd_step_size = data->asgd_step_size;

	for (uint i = 0; i < X->cols; ++i) {) {

		for (uint j = 0; j < y->cols; ++j) {
			
			double margin = y->data[j] *
				(inner_product(X->data+i, data->sgd_weights.data+j, X->cols) + data->sgd_bias);

			if (data->l2_regularization == 0) {
				scalar_product(1 - data->l2_regularization * data->sgd_step_size,
						data->sgd_weights.data,
						data->sgd_weights.cols);
			}

			if (margin < 1) {
				// TODO use proper dimensions
			}

		}
	}
}

void fit(
	native_binary_asgd_t *data,
	matrix_t *X,
	matrix_t *y)
{
	aassert(X->rows == 2, "fit: X should have 2 rows");
	aassert(y->rows == 1, "fit: y should have 2 rows");
	aassert(X->rows == data->n_features, "fit: X has wrong col num");
	aassert(X->cols == y->cols, "fit: X-y dim mismatch");

	for (uint i = 0; i < data->n_iterations; ++ i) {
		durstenfeld_shuffle(X);
		durstenfeld_shuffle(y); // FIXME establish correct dimension
		partial_fit(data, X, y);

		if (data->feedback) {
			memcpy(&data->sgd_weights, &data->asgd_weights, sizeof(data->sgd_weights));
			data->sgd_bias = data->asgd_bias;
		}
	}
}

double predict(
	native_binary_asgd_t *data,
	double *X)
{
	double dot = inner_product(data->asgd_weights.data, X, data->asgd_weights.cols);
	return copysign(1.0, dot + data->asgd_bias);
}

