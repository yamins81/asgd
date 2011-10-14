#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "blas_asgd.h"

int main(
	int argc,
	char *argv[])
{
	srand(time(NULL));
	size_t n_points = 1000;
	size_t n_feats = 100;

	matrix_t *X = matrix_init(n_points, n_feats, 0.0f);
	matrix_t *y = matrix_init(n_points, 1, 0.0f);
	matrix_t *Xtst = matrix_init(n_points, n_feats, 0.0f);
	matrix_t *ytst = matrix_init(n_points, 1, 0.0f);
	for (size_t i = 0; i < n_points; ++i)
	{
		float last0 = powf(-1.0f, rand());
		float last1 = powf(-1.0f, rand());
		matrix_set(y, i, 0, last0);
		matrix_set(ytst, i, 0, last1);
		for (size_t j = 0; j < n_feats; ++j)
		{
			float val0 = 1.0f * rand() / RAND_MAX;
			float val1 = 1.0f * rand() / RAND_MAX;
			val0 = last0 == 1.0f ? val0 + 0.1f : val0;
			val1 = last1 == 1.0f ? val1 + 0.1f : val1;
			matrix_set(X, i, j, val0);
			matrix_set(Xtst, i, j, val1);
		}
	}
	
	nb_asgd_t *clf = nb_asgd_init(n_feats, 1e-3f, 1e-6f, 4, false);
	fit(clf, X, y);
	matrix_t *ytrn_preds = matrix_init(1, n_points, 1.0f);
	predict(clf, X, ytrn_preds);
	matrix_t *ytst_preds = matrix_init(1, n_points, 1.0f);
	predict(clf, Xtst, ytst_preds);

	nb_asgd_destr(clf);
	matrix_destr(ytrn_preds);
	matrix_destr(ytst_preds);
	matrix_destr(X);
	matrix_destr(y);
	matrix_destr(Xtst);
	matrix_destr(ytst);
}

