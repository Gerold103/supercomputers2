#include "io_utils.h"
#include <stdlib.h>
#include <string.h>

/** Rectangle points. */
static const double x_1 = 0, x_2 = 1, y_1 = 0, y_2 = 1;
/** Steps over X and Y axes. */ 
static double x_step = -1;
static double y_step = -1;

/** Needed precision. */
static double epsilon = 0.0001;

/** Border values of neighbour processes. */
static double *P_neigh_buffer[4];
static dvector *P = NULL;
static double *G_neigh_buffer[4];
static dvector *G = NULL;
static double *R_neigh_buffer[4];
static dvector *R = NULL;
static dvector *errors = NULL;
static dvector *increments = NULL;
/** Buffer to store some temporary computation results. */
static dvector *matrix_buffer = NULL;

/** Iterations count of the algorithm. */
static int iterations_count = 0;
/** Maximal iterations count. Useful to see the progress. */
static const int max_iterations_count = 100000;

/** Init neighbour border buffers with 0. */
static inline void
create_border_buffers(int border_id)
{
	int elem_count = border_size[border_id];
	if (elem_count > 0) {
		assert(elem_count != -1);
		double *mem = (double *) calloc(elem_count * 3, sizeof(double));
		assert(mem != NULL);
		P_neigh_buffer[border_id] = mem;
		G_neigh_buffer[border_id] = mem + elem_count;
		R_neigh_buffer[border_id] = mem + elem_count * 2;
	} else {
		P_neigh_buffer[border_id] = NULL;
		G_neigh_buffer[border_id] = NULL;
		R_neigh_buffer[border_id] = NULL;
	}
}

/**
 * Initialize matrices P and R, allocate buffers, G.
 * Calculate first and last internal rows and columns.
 */
static inline void
create_matrices()
{
	/* Make step 0. */
	int neigh_cell_count = 0;
	for (int i = 0; i < 4; ++i) {
		if (border_size[i] > 0)
			neigh_cell_count += border_size[i];
	}
	int size = cell_rows * cell_cols + neigh_cell_count;
	P = dvector_new(size);
	R = dvector_new(size);
	G = dvector_new(size);
	errors = dvector_new(size);
	increments = dvector_new(size);
	matrix_buffer = dvector_new(size);
	create_border_buffers(BOTTOM_BORDER);
	create_border_buffers(TOP_BORDER);
	create_border_buffers(LEFT_BORDER);
	create_border_buffers(RIGHT_BORDER);

	int first_internal_row, first_internal_col;
	int last_internal_row, last_internal_col;
	if (is_bottom)
		first_internal_row = 1;
	else
		first_internal_row = 0;
	if (is_top)
		last_internal_row = cell_rows - 2;
	else
		last_internal_row = cell_rows - 1;
	if (is_left)
		first_internal_col = 1;
	else
		first_internal_col = 0;
	if (is_right)
		last_internal_col = cell_cols - 2;
	else
		last_internal_col = cell_cols - 1;

	cuda_init(cell_rows, cell_cols, first_internal_row, last_internal_row,
		  first_internal_col, last_internal_col, start_x_i, start_y_i,
		  x_1, y_1, x_step, y_step, border_size);
	cuda_init_P(P, P_neigh_buffer);
	MPI_Request req[4];
	for (int i = 0; i < 4; ++i)
		req[i] = MPI_REQUEST_NULL;

	send_borders(P_neigh_buffer);
	receive_borders(P_neigh_buffer, req);
	sync_receive_borders(req, 4);
	cuda_copy_borders_from_host(P_neigh_buffer, P);

	cuda_calculate_next_R(P, R, R_neigh_buffer);

	send_borders(R_neigh_buffer);
	receive_borders(R_neigh_buffer, req);
	sync_receive_borders(req, 4);
	cuda_copy_borders_from_host(R_neigh_buffer, R);
}

/**
 * Calculate algorithm.
 */
static inline void
calculate()
{
	/* Make step 1. */
	iterations_count++;

	/* Calculate tau = (r, r) / (-laplas(r), r). */
	double numerator = cuda_scalar(R, R);
	cuda_laplas_5_matrix(R, matrix_buffer);
	double denominator = cuda_scalar(matrix_buffer, R);
	double tau = global_scalar_fraction(numerator, denominator);

	/* Calculate error. */
	double local_inc = cuda_calculate_next_P(P, tau, R, increments);
	double global_inc = global_increment(local_inc, x_step * y_step);
	if (proc_rank == 0)
		printf("global_increment = %lf\n", global_inc);

	cuda_init_G(R, G);

	MPI_Request req[4];
	for (int i = 0; i < 4; ++i)
		req[i] = MPI_REQUEST_NULL;

	/* Make step i. */
	while (global_inc > epsilon &&
	       iterations_count < max_iterations_count) {
	       	++iterations_count;
		/*
		 * Calcuilate
		 * 1) R_i
		 * 2) alpha_i = (-laplas(R_i), G_i-1) /
		 * 		(-laplas(G_i-1), G_i-1)
		 * 3) G_i = R_i - alpha * G_i-1
		 * 4) tau_i+1 = (R_i, G_i) / (-laplas(G_i), G_i)
		 * 5) P_i+1 = P_i - tau_i+1 * G_i
		 */
		cuda_calculate_next_R(P, R, R_neigh_buffer);
		/*
		 * Send now, but sync later, when the borders are
		 * actually needed.
		 */
		send_borders(R_neigh_buffer);
		receive_borders(R_neigh_buffer, req);
		cuda_laplas_5_matrix(G, matrix_buffer);
		double denominator = cuda_scalar(matrix_buffer, G);
		/* Receive R borders. */
		/* (1) */
		sync_receive_borders(req, 4);
		cuda_copy_borders_from_host(R_neigh_buffer, R);

		cuda_laplas_5_matrix(R, matrix_buffer);
		double numerator = cuda_scalar(matrix_buffer, G);
		/* (2) */
		double alpha = global_scalar_fraction(numerator, denominator);

		/* (3) */
		cuda_calculate_next_G(G, R, alpha);

		numerator = cuda_scalar(R, G);
		cuda_laplas_5_matrix(G, matrix_buffer);
		denominator = cuda_scalar(matrix_buffer, G);
		/* (4) */
		double tau = global_scalar_fraction(numerator, denominator);

		/* (5) */
		double local_inc = cuda_calculate_next_P(P, tau, G, increments);
		global_inc = global_increment(local_inc, x_step * y_step);
		if (proc_rank == 0 && iterations_count % 10 == 0)
			printf("global_increment = %lf\n", global_inc);
	}
	double local_error = cuda_calculate_P_error(P, errors);
	double global_error = global_increment(local_error, x_step * y_step);
	if (proc_rank == 0) {
		printf("result global_increment = %lf, global_error = %lf\n", global_inc,
	               global_error);
		printf("finished in %d iterations\n", iterations_count);
	}
}

int
main(int argc, char **argv)
{
	int table_height, table_width;
	if (argc >= 3) {
		table_height = atoi(argv[1]);
		if (table_height == 0) {
			printf("Incorrect table height\n");
			return -1;
		}
		table_width = atoi(argv[2]);
		if (table_width == 0) {
			printf("Incorrect table width\n");
			return -1;
		}
	} else {
		table_height = 1000;
		table_width = 1000;
	}
	x_step = (x_2 - x_1) / (table_width - 1);
	y_step = (y_2 - y_1) / (table_height - 1);
	double start_time = 0, end_time = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	int proc_count;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
	if (calculate_cells(table_height, table_width, proc_count) != 0) {
		printf("Cannot split table in a specified process count\n");
		goto error;
	}
	create_matrices();

	start_time = MPI_Wtime();
	calculate();
	end_time = MPI_Wtime();
	if (proc_rank == 0)
		printf("time = %lf\n", end_time - start_time);

	dvector_delete(P);
	dvector_delete(R);
	dvector_delete(G);
	dvector_delete(errors);
	dvector_delete(increments);
	dvector_delete(matrix_buffer);
	for (int i = 0; i < 4; ++i)
		free(P_neigh_buffer[i]);

	cuda_exit();

	MPI_Finalize();
	return 0;
error:
	MPI_Finalize();
	return -1;
}
