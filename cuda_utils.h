#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifndef __cplusplus
typedef struct dvector dvector;
#else
#include <thrust/device_vector.h>
typedef thrust::device_vector<double> dvector;

extern "C" {
#endif

/** Indexes to access border arrays. */
enum {
	BOTTOM_BORDER, TOP_BORDER, LEFT_BORDER, RIGHT_BORDER
};

/**
 * Initialize cuda global device variables with a specified
 * values.
 */
void
cuda_init(int h_cell_rows, int h_cell_cols, int h_first_internal_row,
	  int h_last_internal_row, int h_first_internal_col,
	  int h_last_internal_col, int h_start_x_i, int h_start_y_i,
	  double h_x_1, double h_y_1, double h_x_step, double h_y_step,
	  int *border_size);

/** Free internal resources. */
void
cuda_exit();

/**
 * C wrappers to create thrust vectors and use them as
 * matrices.
 */
dvector *
dvector_new(int size);

void
dvector_delete(dvector *vector);

/**
 * Calculate a -laplas(src) and save it to @a dst.
 * @param src Source matrix.
 * @param borders Borders from a neighbour processes of @a src.
 * @param dst Destination matrix to which to save a result.
 */
void
cuda_laplas_5_matrix(const dvector *src, dvector *dst);

/** Calculate an inner product of two matrices. */
double
cuda_scalar(const dvector *a, const dvector *b);

/**
 * Calculate a next G matrix using the formula:
 * G = R - alpha * G.
 * @param g G matrix.
 * @param r R matrix.
 * @param alpha Coefficient for a current G.
 */
void
cuda_calculate_next_G(dvector *g, const dvector *r, double alpha);

/**
 * Calculate a new P matrix using the formula:
 * P = P - tau * disp.
 * @param p P matrix.
 * @param tau Coefficient for @a disp.
 * @param disp Either R or G.
 * @param increments Buffer to calculate increment from a current
 *        P.
 *
 * @retval Increment of @a P local for a current process.
 */
double
cuda_calculate_next_P(dvector *p, double tau, const dvector *disp,
		      dvector *increments);

/**
 * Calculate a new R matrux using the formula:
 * R = -laplas(P) - F.
 * Borders of a result R are saved into @a r_border_buffers. It
 * is used to do not copy them separately after R calculation and
 * avoid many copying from a device.
 * @param p P matrix.
 * @param r R matrix.
 * @param r_border_buffers Array of borders to save R borders to.
 */
void
cuda_calculate_next_R(const dvector *p, dvector *r, double **r_border_buffers);

/**
 * Calculate an error of result P and ethalon decision.
 * @param p Last calculated P.
 * @param errors Buffer for error values in each point.
 *
 * @retval Error of @a P local for a current process.
 */
double
cuda_calculate_P_error(dvector *p, dvector *errors);

/**
 * Initialize P with phi() on borders of a global grid and 0 in
 * internal points. Save result border values in @a p_borders.
 * @param p P matrix.
 * @param p_borders Array of borders of @a P.
 */
void
cuda_init_P(dvector *p, double **p_borders);

/**
 * Initialize G as a copy of R.
 * @param r R matrix.
 * @param g G matrix.
 */
void
cuda_init_G(dvector *r, dvector *g);

/**
 * Copy a host memory borders into a device memory.
 * @param borders Host memory borders.
 * @param mat_borders Device borders.
 */
void
cuda_copy_borders_from_host(const double * const *borders, dvector *mat);

#ifdef __cplusplus
}
#endif

#endif

