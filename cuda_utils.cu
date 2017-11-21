#include "cuda_utils.h"
#include <thrust/inner_product.h>
#include <thrust/gather.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

using namespace thrust;

static int host_cell_rows = -1;
static int host_cell_cols = -1;
static double host_x_step = -1;
static double host_y_step = -1;

/**
 * A column can not be copied directly from a device to a host
 * memory, and it must done in two steps:
 * 1) gather a column points into a preallocated device buffer;
 * 2) copy buffer to a host.
 * This vector is the buffer to gather column points during
 * matrix transformation. It allows to avoid separate gathering.
 */
static dvector *left_column_buffer, *right_column_buffer;

/**
 * Variables in a device memory which values will not change
 * afer initialization in cuda_init().
 */
/** Size of a cell owned by the current process. */
__constant__ static int cell_rows = -1;
__constant__ static int cell_cols = -1;
/**
 * Each process owns a set of points, grouped by rows and columns.
 * Border rows and columns can be the borders of an entire grid.
 * But most of operations do not use grid borders
 * (scalar, laplas). These four values stores for each process
 * its subgrid with only grid internal points.
 * Example:
 *     |  ...
 *     +---+---+    <- last_internal_row
 *     |   |   | ...
 *     +---+---+    <- first_internal_row
 *     |   |   | ...
 *     +---+---+---
 *         ^
 *  first_internal_col
 */
__constant__ static int first_internal_row = -1;
__constant__ static int last_internal_row = -1;
__constant__ static int first_internal_col = -1;
__constant__ static int last_internal_col = -1;
/** Precalculated 1/(x_step^2) and 1/(y_step^2). */
__constant__ static double h1_in_2_reverted = -1;
__constant__ static double h2_in_2_reverted = -1;
/** Index of a start point by X axis. */
__constant__ static int start_x_i = -1;
/** Index of a start point by Y axis. */
__constant__ static int start_y_i = -1;
/** First X in a global grid. */
__constant__ static double x_1 = -1;
/** First Y in a global grid. */
__constant__ static double y_1 = -1;
/** Precalculated step by X on each i. */
__constant__ static double x_step = -1;
/** Precalculated step by Y on each i. */
__constant__ static double y_step = -1;
/** Offset in a matrix array to a borders. */
static int border_offsets[4];

/** Stream to move borders between host and device memory. */
static cudaStream_t borders_stream;

/**
 * Macros to copy a value from a host memory to a const memory of
 * a device.
 */
#define copy_to_device(dst, src) \
	cudaMemcpyToSymbol(dst, &src, sizeof(dst), 0, cudaMemcpyHostToDevice)

void
cuda_init(int h_cell_rows, int h_cell_cols, int h_first_internal_row,
	  int h_last_internal_row, int h_first_internal_col,
	  int h_last_internal_col, int h_start_x_i, int h_start_y_i,
	  double h_x_1, double h_y_1, double h_x_step, double h_y_step,
	  int *border_size)
{
	host_cell_rows = h_cell_rows;
	host_cell_cols = h_cell_cols;
	host_x_step = h_x_step;
	host_y_step = h_y_step;
	copy_to_device(cell_rows, h_cell_rows);
	copy_to_device(cell_cols, h_cell_cols);
	copy_to_device(first_internal_row, h_first_internal_row);
	copy_to_device(last_internal_row, h_last_internal_row);
	copy_to_device(first_internal_col, h_first_internal_col);
	copy_to_device(last_internal_col, h_last_internal_col);
	double tmp = 1 / (h_x_step * h_x_step);
	copy_to_device(h1_in_2_reverted, tmp);
	tmp = 1 / (h_y_step * h_y_step);
	copy_to_device(h2_in_2_reverted, tmp);
	copy_to_device(start_x_i, h_start_x_i);
	copy_to_device(start_y_i, h_start_y_i);
	copy_to_device(x_1, h_x_1);
	copy_to_device(y_1, h_y_1);
	copy_to_device(x_step, h_x_step);
	copy_to_device(y_step, h_y_step);
	int offset = h_cell_cols * h_cell_rows;
	for (int i = 0; i < 4; ++i) {
		if (border_size[i] == -1) {
			border_offsets[i] = -1;
		} else {
			border_offsets[i] = offset;
			offset += border_size[i];
		}
	}

	left_column_buffer = dvector_new(host_cell_rows);
	right_column_buffer = dvector_new(host_cell_rows);
	cudaStreamCreate(&borders_stream);
}

void
cuda_exit()
{
	delete left_column_buffer;
	delete right_column_buffer;
	cudaStreamDestroy(borders_stream);
}

#define X(i) (x_1 + (start_x_i + (i)) * x_step)
#define Y(i) (y_1 + (start_y_i + (i)) * y_step)

__host__ dvector *
dvector_new(int size)
{
	return new dvector(size);
}

__host__ void
dvector_delete(dvector *vector)
{
	delete vector;
}

/**
 * The algorithm solves the following system:
 * -laplas(u) = F(x, y),  (x, y) are internal points.
 * u(x, y) = phi(xm y),  (x, y) are border points.
 *
 * The phi() and F() functions are defined below.
 */
__device__ static inline double
cuda_phi(double x, double y)
{
	double t1 = 1 - x * x;
	double t2 = 1 - y * y;
	return t1 * t1 + t2 * t2;
}

__device__ static inline double
cuda_F(double x, double y)
{
	return 4 * (2 - 3 * x * x - 3 * y * y);
}

/** Analitycal decision of the task. Used to calculate error. */
__device__ static inline double
cuda_ethalon(double x, double y)
{
	return cuda_phi(x, y);
}

/**
 * Matrices are flattened into a vectors. And index of a matrix
 * [i][j] must be translated into an index of an array.
 * These two functions incapsulates this logic.
 */
__device__ static inline double
cuda_cell(const double *matrix, int row, int col)
{
	int idx = row * cell_cols + col;
	assert(idx <= cell_cols * cell_rows);
	return matrix[idx];
}

__device__ static inline double
cuda_laplas_5(const double *matrix, int row, int col, const double *top_border,
	      const double *bottom_border, const double *left_border,
	      const double *right_border)
{
	assert(row >= first_internal_row && row <= last_internal_row);
	assert(col >= first_internal_col && col <= last_internal_col);
	double a_11 = cuda_cell(matrix, row, col);
	double a_01;
	if (row > 0)
		a_01 = cuda_cell(matrix, row - 1, col);
	else
		a_01 = bottom_border[col];
	double a_21;
	if (row + 1 < cell_rows) {
		a_21 = cuda_cell(matrix, row + 1, col);
	} else {
		assert(top_border != NULL);
		a_21 = top_border[col];
	}
	double a_10;
	if (col > 0) {
		a_10 = cuda_cell(matrix, row, col - 1);
	} else {
		assert(left_border != NULL);
		a_10 = left_border[row];
	}
	double a_12;
	if (col + 1 < cell_cols) {
		a_12 = cuda_cell(matrix, row, col + 1);
	} else {
		assert(right_border != NULL);
		a_12 = right_border[row];
	}

	double tmp1 = h1_in_2_reverted * (2 * a_11 - a_01 - a_21);
	double tmp2 = h2_in_2_reverted * (2 * a_11 - a_10 - a_12);
	return -tmp1 - tmp2;
}

struct functor_with_borders
{
	/**
	 * Neighbour borders of a specified matrix from another
	 * processes.
	 */
	const double *top_border;
	const double *bottom_border;
	const double *left_border;
	const double *right_border;

	functor_with_borders(const dvector *matrix) {
		const double *data = matrix->data().get();
		if (border_offsets[TOP_BORDER] == -1)
			top_border = NULL;
		else
			top_border = data + border_offsets[TOP_BORDER];
		if (border_offsets[BOTTOM_BORDER] == -1)
			bottom_border = NULL;
		else
			bottom_border = data + border_offsets[BOTTOM_BORDER];
		if (border_offsets[LEFT_BORDER] == -1)
			left_border = NULL;
		else
			left_border = data + border_offsets[LEFT_BORDER];
		if (border_offsets[RIGHT_BORDER] == -1)
			right_border = NULL;
		else
			right_border = data + border_offsets[RIGHT_BORDER];
	}
};

/** Functor to calculate -laplas from a specified matrix. */
struct laplas_5_functor : public functor_with_borders
{
	/** Raw data of a source matrix. */
	const double *src;

	laplas_5_functor(const dvector *src)
		: functor_with_borders(src), src(src->data().get()) { }

	__device__ double
	operator() (int idx) const {
		int row = idx / cell_cols;
		int col = idx % cell_cols;
		if (row < first_internal_row || row > last_internal_row ||
		    col < first_internal_col || col > last_internal_col) {
			return cuda_cell(src, row, col);
		} else {
			return -cuda_laplas_5(src, row, col, top_border,
					      bottom_border, left_border,
					      right_border);
		}
	}
};

void
cuda_laplas_5_matrix(const dvector *src, dvector *dst)
{
	struct laplas_5_functor func(src);
	counting_iterator<int> first(0);
	transform(first, first + host_cell_rows * host_cell_cols, dst->begin(), func);
}

double
cuda_scalar(const dvector *a, const dvector *b)
{
	assert(a->size() == b->size());
	return inner_product(a->begin(), a->begin() + host_cell_rows * host_cell_cols,
			     b->begin(), (double)0) * host_x_step * host_y_step;
}

/** Calculate a linear combination of two matrices. */
struct calculator_LinearComb
{
	double alpha;

	calculator_LinearComb(double alpha)
		: alpha(alpha) { }

	__device__ double
	operator ()(double r, double g) {
		return r - alpha * g;
	}
};

void
cuda_calculate_next_G(dvector *g, const dvector *r, double alpha)
{
	assert(g->size() == r->size());
	calculator_LinearComb calc(alpha);
	transform(r->begin(), r->end(), g->begin(), g->begin(), calc);
}

/**
 * Functor to calculate a next P and save increments in each
 * point.
 */
struct calculator_P
{
	double tau;
	const double *disp;
	double *increments;

	calculator_P(double tau, const dvector *disp, dvector *increments)
		: tau(tau), disp(disp->data().get()),
		  increments(increments->data().get()) { }

	__device__ double
	operator ()(int idx, double p) {
		double new_p = p - tau * disp[idx];
		double val = new_p - p;
		increments[idx] = val * val;
		return new_p;
	}
};

double
cuda_calculate_next_P(dvector *p, double tau, const dvector *disp,
		      dvector *increments)
{
	counting_iterator<int> first(0);
	calculator_P calc(tau, disp, increments);
	transform(first, first + p->size(), p->begin(), p->begin(), calc);
	return reduce(increments->begin(), increments->begin() + host_cell_rows * host_cell_cols);
}

/** Functor to collect errors of a P in each point. */
struct errors_collector {
	double *errors;

	errors_collector(dvector *errors)
		: errors(errors->data().get()) { }

	__device__ double
	operator ()(int idx, double p) {
		int row = idx / cell_cols;
		int col = idx % cell_cols;
		double diff = cuda_ethalon(X(col), Y(row)) - p;
		errors[idx] = diff * diff;
		return p;
	}
};

double
cuda_calculate_P_error(dvector *p, dvector *errors)
{
	counting_iterator<int> first(0);
	transform(first, first + host_cell_rows * host_cell_cols, p->begin(), p->begin(),
		  errors_collector(errors));
	return reduce(errors->begin(), errors->begin() + host_cell_rows * host_cell_cols);
}

/** Functor to calculate a next R. */
struct calculator_R : public functor_with_borders
{
	const double *p;

	double *left_column;
	double *right_column;

	calculator_R(const dvector *p)
		: functor_with_borders(p), p(p->data().get()) {
		left_column = left_column_buffer->data().get();
		right_column = right_column_buffer->data().get();
	}

	__device__ double
	operator ()(int idx) {
		int row = idx / cell_cols;
		int col = idx % cell_cols;
		if (row < first_internal_row || row > last_internal_row ||
		    col < first_internal_col || col > last_internal_col)
			return 0;
		double ret = -cuda_laplas_5(p, row, col, top_border,
					    bottom_border, left_border,
					    right_border) -
			     cuda_F(X(col), Y(row));
		if (col == 0 && first_internal_col == col)
			left_column[row] = ret;
		if (col == cell_cols - 1 && last_internal_col == col)
			right_column[row] = ret;
		return ret;
	}
};

static inline void
cuda_copy_borders_to_host(const dvector *src, double **borders)
{
	int border_cnt = 0;
	if (borders[BOTTOM_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(borders[BOTTOM_BORDER],
				src->data().get(), host_cell_cols * sizeof(double),
				cudaMemcpyDeviceToHost, borders_stream);
	}
	if (borders[TOP_BORDER] != NULL) {
		border_cnt++;
		int last_row_idx = (host_cell_rows - 1) * host_cell_cols;
		cudaMemcpyAsync(borders[TOP_BORDER],
				src->data().get() + last_row_idx,
				host_cell_cols * sizeof(double), cudaMemcpyDeviceToHost,
				borders_stream);
	}
	if (borders[LEFT_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(borders[LEFT_BORDER],
				left_column_buffer->data().get(),
				host_cell_rows * sizeof(double), cudaMemcpyDeviceToHost,
				borders_stream);
	}
	if (borders[RIGHT_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(borders[RIGHT_BORDER],
				right_column_buffer->data().get(),
				host_cell_rows * sizeof(double), cudaMemcpyDeviceToHost,
				borders_stream);
	}
	if (border_cnt > 0)
		cudaStreamSynchronize(borders_stream);
}

void
cuda_calculate_next_R(const dvector *p, dvector *r, double **r_border_buffers)
{
	counting_iterator<int> first(0);
	transform(first, first + p->size(), r->begin(),
		  calculator_R(p));
	cuda_copy_borders_to_host(r, r_border_buffers);
}

/**
 * Functor to initialize a P with 0 in internal points and
 * phi() on grid border points.
 */
struct constructor_P
{
	double *left_column;
	double *right_column;

	constructor_P() {
		left_column = left_column_buffer->data().get();
		right_column = right_column_buffer->data().get();
	}

	__device__ double
	operator ()(int idx) {
		int row = idx / cell_cols;
		int col = idx % cell_cols;
		double ret;
		if (row >= first_internal_row && row <= last_internal_row &&
		    col >= first_internal_col && col <= last_internal_col)
			ret = 0;
		else
			ret = cuda_phi(X(col), Y(row));
		if (col == 0 && first_internal_col == 0)
			left_column[row] = ret;
		if (col == cell_cols - 1 && last_internal_col == col)
			right_column[row] = ret;
		return ret;
	}
};

void
cuda_init_P(dvector *p, double **p_borders)
{
	counting_iterator<int> first(0);
	transform(first, first + host_cell_rows * host_cell_cols,
		  p->begin(), constructor_P());
	cuda_copy_borders_to_host(p, p_borders);
}

void
cuda_init_G(dvector *r, dvector *g)
{
	copy(r->begin(), r->end(), g->begin());
}

void
cuda_copy_borders_from_host(const double * const *borders, dvector *mat)
{
	int border_cnt = 0;
	double *data = mat->data().get();
	if (borders[TOP_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(data + border_offsets[TOP_BORDER],
				borders[TOP_BORDER], host_cell_cols * sizeof(double),
				cudaMemcpyHostToDevice, borders_stream);
	}
	if (borders[BOTTOM_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(data + border_offsets[BOTTOM_BORDER],
				borders[BOTTOM_BORDER], host_cell_cols * sizeof(double),
				cudaMemcpyHostToDevice, borders_stream);
	}
	if (borders[LEFT_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(data + border_offsets[LEFT_BORDER],
				borders[LEFT_BORDER], host_cell_rows * sizeof(double),
				cudaMemcpyHostToDevice, borders_stream);
	}
	if (borders[RIGHT_BORDER] != NULL) {
		border_cnt++;
		cudaMemcpyAsync(data + border_offsets[RIGHT_BORDER],
				borders[RIGHT_BORDER], host_cell_rows * sizeof(double),
				cudaMemcpyHostToDevice, borders_stream);
	}
	if (border_cnt > 0)
		cudaStreamSynchronize(borders_stream);
}

