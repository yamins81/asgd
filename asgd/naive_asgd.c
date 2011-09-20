#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// type and size parameters
// will need to autoset them later on
typedef uint64_t uint_t;
typedef double float_t;

typedef struct native_binary_asgd native_binary_asgd_t;
struct native_binary_asgd
{
	uint_t n_features;
	uint_t n_iterations;
	float_t l2_regularization;
	float_t sgd_step_size;
	float_t sgd_step_size0;
	bool feedback;

	size_t sgd_weights_size;
	float_t *sgd_weights; // n_features long
	
	float_t sgd_bias;
	float_t sgd_step_size_scheduling_exponent;
	float_t sgd_step_size_scheduling_multiplier;

	size_t asgd_weights_size;
	float_t *asgd_weights; // n_features long
	
	float_t asgd_bias;
	float_t asgd_step_size;
	float_t asgd_step_size;
};

/**
 * Constructor for the Binary ASGD structure
 */
void native_binary_asgd_init(
	native_binary_asgd_t *data,
	uint_t n_features,
	float_t sgd_step_size0,
	float_t l2_regularization,
	uint_t n_iterations,
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

	data->sgd_weights = malloc(n_features*sizeof(*data->sgd_weights));
	if (__builtin_expect(data->sgd_weights == NULL, false))
	{
		fprintf(stderr, "could not allocate sgd_weights, quitting");
		exit(EXIT_FAILURE);
	} else {
		data->sgd_weights_size = n_features;
	}

	data->sgd_bias = 0;
	data->sgd_step_size = sgd_step_size0;
	data->sgd_step_size0 = sgd_step_size0;
	data->sgd_step_size_schedulig_exponent = 2. / 3.;
	data->sgd_step_size_scheduling_multiplier = l2_regularization;

	data->asgd_weights = malloc(n_features*sizeof(*data->asgd_weights));
	if (__builtin_expect(data->asgd_weights == NULL, false))
	{
		fprintf(stderr, "could not allocate asgd_weights, quitting");
		exit(EXIT_FAILURE);
	} else {
		data->asgd_weights_size = n_features;
	}

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
	free(data->sgd_weights);
	free(data->asgd_weights);
}

