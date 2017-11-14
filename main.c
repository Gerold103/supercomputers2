#include "io_utils.h"
#include <stdlib.h>
#include <string.h>

/** Rectangle points. */
static const double x_1 = 0, x_2 = 1, y_1 = 0, y_2 = 1;
/** Steps over X and Y axes. */ 
static double x_step = -1;
static double y_step = -1;
/** Buffered result of 1/(x_step^2) and 1/(y_step^2). */
static double h1_in_2_reverted = -1;
static double h2_in_2_reverted = -1;

/** Needed precision. */
static double epsilon = 0.0001;

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
 *     
 */
static int first_internal_row = -1;
static int last_internal_row = -1;
static int first_internal_col = -1;
static int last_internal_col = -1;

/** Border values of neighbour processes. */
static double *P_neigh[4];
static double *P = NULL;
static double *G_neigh[4];
static double *G = NULL;
static double *R_neigh[4];
static double *R = NULL;
/** Buffer to store some temporary computation results. */
static double *matrix_buffer = NULL;

/** Iterations count of the algorithm. */
static int iterations_count = 0;
/** Maximal iterations count. Useful to see the progress. */
static const int max_iterations_count = 2000000;

/**
 * The algorithm solves the following system:
 * -laplas(u) = F(x, y),  (x, y) are internal points.
 * u(x, y) = phi(xm y),  (x, y) are border points.
 *
 * The phi() and F() functions are defined below.
 */
static inline double
phi(double x, double y)
{
	double t1 = 1 - x * x;
	double t2 = 1 - y * y;
	return t1 * t1 + t2 * t2;
}

static inline double
F(double x, double y)
{
	return 4 * (2 - 3 * x * x - 3 * y * y);
}

/** Init neighbour borders with 0. */
static inline void
create_borders(int border_id, bool exists)
{
	if (exists) {
		int elem_count = border_size[border_id];
		int size = elem_count * 3;
		double *mem = (double *) calloc(size, sizeof(double));
		assert(mem != NULL);
		P_neigh[border_id] = mem;
		G_neigh[border_id] = mem + elem_count;
		R_neigh[border_id] = mem + elem_count * 2;
	} else {
		P_neigh[border_id] = NULL;
		G_neigh[border_id] = NULL;
		R_neigh[border_id] = NULL;
	}
}

/**
 * 5-point approximation of the laplas operator.
 * @param matrix Source matrix.
 * @param row Element row.
 * @param col Element column.
 * @param borders Matrix neighbour processes borders.
 *
 * @retval 5-point laplas operator in point matrix[row][col].
 */
static inline double
laplas_5(double *matrix, int row, int col, double **borders)
{
	assert(row >= first_internal_row && row <= last_internal_row);
	assert(col >= first_internal_col && col <= last_internal_col);
	double a_11 = get_cell(matrix, row, col);
	double a_01;
	if (row > 0)
		a_01 = get_cell(matrix, row - 1, col);
	else
		a_01 = borders[BOTTOM_BORDER][col];
	double a_21;
	if (row + 1 < cell_rows) {
		a_21 = get_cell(matrix, row + 1, col);
	} else {
		assert(borders[TOP_BORDER] != NULL);
		a_21 = borders[TOP_BORDER][col];
	}
	double a_10;
	if (col > 0) {
		a_10 = get_cell(matrix, row, col - 1);
	} else {
		assert(borders[LEFT_BORDER] != NULL);
		a_10 = borders[LEFT_BORDER][row];
	}
	double a_12;
	if (col + 1 < cell_cols) {
		a_12 = get_cell(matrix, row, col + 1);
	} else {
		assert(borders[RIGHT_BORDER] != NULL);
		a_12 = borders[RIGHT_BORDER][row];
	}

	double tmp1 = h1_in_2_reverted * (2 * a_11 - a_01 - a_21);
	double tmp2 = h2_in_2_reverted * (2 * a_11 - a_10 - a_12);
	return tmp1 + tmp2;
}

/**
 * Calculate 5-point laplas for internal elements of @a src
 * matrix and store them in @a dst matrix.
 * @param src Source matrix.
 * @param src_borders Neighbour processes borders of @a src.
 * @param[out] dst Destination matrix.
 */
static inline void
laplas_5_matrix(double *src, double **src_borders, double *dst)
{
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j)
			set_cell(dst, i, j, -laplas_5(src, i, j, src_borders));
	}
}

/**
 * Calculate a (a, b) scalar product in for a local process
 * matrices. For a global scalar see global_scalar function().
 * @param a Matrix.
 * @param b Matrix.
 * @retval Scalar product.
 */
static inline double
local_scalar(double *a, double *b)
{
	double ret = 0;
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j)
			ret += get_cell(a, i, j) * get_cell(b, i, j);
	}
	return ret * x_step * y_step;
}

/**
 * Next G is calculated using formula:
 * G = R - alpha * G
 * @param alpha Coefficient for the old G.
 */
static inline void
calculate_next_G(double alpha)
{
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j) {
			set_cell(G, i, j, get_cell(R, i, j) -
				 alpha * get_cell(G, i, j));
		}
	}
	for (int i = 0; i < 4; ++i) {
		if (G_neigh[i] == NULL)
			continue;
		assert(R_neigh[i] != NULL);
		int count = border_size[i];
		for (int j = 0; j < count; ++j)
			G_neigh[i][j] = R_neigh[i][j] - alpha * G_neigh[i][j];
	}
}

/**
 * Next P is calculated using formula:
 * P = P - tau * discrepancy_matrix.
 * @param tau Coefficient for @a discrepancy_matrix.
 * @param discrepancy_matrix R on step 1 and G on other steps.
 * @param discrepancy_borders Neighbour processes borders of
 *        @a discrepancy_matrix.
 *
 * @retval Difference between old and new P for a local process.
 *         Calculated using formula:
 *         (P_new - P_old, P_new - P_old). For a next step see
 *         global_increment() function.
 */
static inline double
calculate_next_P(double tau, double *discrepancy_matrix,
		 double **discrepancy_borders)
{
	double increment = 0;
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j) {
			double old = get_cell(P, i, j);
			set_cell(P, i, j, old -
				 tau * get_cell(discrepancy_matrix, i, j));
			double local_increment = get_cell(P, i, j) - old;
			increment += local_increment * local_increment;
		}
	}
	for (int i = 0; i < 4; ++i) {
		if (P_neigh[i] == NULL)
			continue;
		assert(discrepancy_borders[i] != NULL);
		int count = border_size[i];
		for (int j = 0; j < count; ++j) {
			double old = P_neigh[i][j];
			P_neigh[i][j] -= tau * discrepancy_borders[i][j];
			double local_increment = P_neigh[i][j] - old;
			increment += local_increment * local_increment;
		}
	}
	return increment;
}

/**
 * Next R is calculated using formula:
 * R = -laplas_5(P) - F() for internal points and
 * R = 0 for grid border points.
 */
static inline void
calculate_next_R()
{
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j) {
			set_cell(R, i, j, -laplas_5(P, i, j, P_neigh)
				 - F(X(j), Y(i)));
		}
	}
	if (is_bottom) {
		for (int i = 0; i < cell_cols; ++i)
			set_cell(R, 0, i, 0);
	}
	if (is_top) {
		for (int i = 0; i < cell_cols; ++i)
			set_cell(R, cell_rows - 1, i, 0);
	}
	if (is_left) {
		for (int i = 0; i < cell_rows; ++i)
			set_cell(R, i, 0, 0);
	}
	if (is_right) {
		for (int i = 0; i < cell_rows; ++i)
			set_cell(R, i, cell_cols - 1, 0);
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

	int max_count = cell_rows > cell_cols ? cell_rows : cell_cols;
	int size = cell_rows * cell_cols;
	double *mem = (double *) calloc(size * 4 + max_count * 2,
					sizeof(double));
	assert(mem != NULL);
	P = mem;
	R = mem + size;
	G = mem + size * 2;
	matrix_buffer = mem + size * 3;
	create_borders(BOTTOM_BORDER, ! is_bottom);
	create_borders(TOP_BORDER, ! is_top);
	create_borders(LEFT_BORDER, ! is_left);
	create_borders(RIGHT_BORDER, ! is_right);
	border_buffer_left = mem + size * 4;
	border_buffer_right = mem + size * 4 + max_count;

	if (is_bottom) {
		for (int i = 0; i < cell_cols; ++i)
			set_cell(P, 0, i, phi(X(i), Y(0)));
	}
	if (is_top) {
		for (int i = 0; i < cell_cols; ++i) {
			set_cell(P, cell_rows - 1, i,
				 phi(X(i), Y(cell_rows - 1)));
		}
	}
	if (is_right) {
		for (int i = 0; i < cell_rows; ++i) {
			set_cell(P, i, cell_cols - 1,
				 phi(X(cell_cols - 1), Y(i)));
		}
	}
	if (is_left) {
		for (int i = 0; i < cell_rows; ++i)
			set_cell(P, i, 0, phi(X(0), Y(i)));
	}
	MPI_Request req[4];
	for (int i = 0; i < 4; ++i)
		req[i] = MPI_REQUEST_NULL;
	send_borders(P);
	receive_borders(P_neigh, req);
	sync_receive_borders(req, 4);

	/* Create R internal points. */
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
	for (int i = first_internal_row; i <= last_internal_row; ++i) {
		for (int j = first_internal_col; j <= last_internal_col; ++j) {
			set_cell(R, i, j, -laplas_5(P, i, j, P_neigh) -
				 F(X(j), Y(i)));
		}
	}
	send_borders(R);
	receive_borders(R_neigh, req);
	sync_receive_borders(req, 4);
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
	double numerator = local_scalar(R, R);
	laplas_5_matrix(R, R_neigh, matrix_buffer);
	double denominator = local_scalar(matrix_buffer, R);
	double tau = global_scalar_fraction(numerator, denominator);

	/* Calculate error. */
	double local_inc = calculate_next_P(tau, R, R_neigh);
	double global_inc = global_increment(local_inc);
	if (proc_rank == 0)
		printf("global_increment = %lf\n", global_inc);

	/* Prepare G for a next steps. */
	memcpy(G, R, cell_rows * cell_cols * sizeof(double));
	for (int i = 0; i < 4; ++i) {
		if (R_neigh[i] == NULL)
			continue;
		int count = border_size[i];
		memcpy(G_neigh[i], R_neigh[i], count * sizeof(double));
	}

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
		calculate_next_R();
		/*
		 * Send now, but sync later, when the borders are
		 * actually needed.
		 */
		send_borders(R);
		receive_borders(R_neigh, req);
		laplas_5_matrix(G, G_neigh, matrix_buffer);
		double denominator = local_scalar(matrix_buffer, G);
		/* Receive R borders. */
		/* (1) */
		sync_receive_borders(req, 4);

		laplas_5_matrix(R, R_neigh, matrix_buffer);
		double numerator = local_scalar(matrix_buffer, G);
		/* (2) */
		double alpha = global_scalar_fraction(numerator, denominator);

		/* (3) */
		calculate_next_G(alpha);

		numerator = local_scalar(R, G);
		laplas_5_matrix(G, G_neigh, matrix_buffer);
		denominator = local_scalar(matrix_buffer, G);
		/* (4) */
		double tau = global_scalar_fraction(numerator, denominator);

		/* (5) */
		double local_inc = calculate_next_P(tau, G, G_neigh);
		global_inc = global_increment(local_inc);
		if (proc_rank == 0 && iterations_count % 10 == 0)
			printf("global_increment = %lf\n", global_inc);
	}
	printf("finished in %d iterations\n", iterations_count);
}

/**
 * Write a matrix into a file in JSON format.
 * @param matrix Matrix to write.
 * @param name File name.
 * @param borders Neighbour processes borders or NULL.
 */
static void
dump_matrix(double *matrix, const char *name, double **borders)
{
	char filename[1024];
	snprintf(filename, sizeof(filename), "%s_vers%02d_proc%05d", name,
		 iterations_count, proc_rank);
	FILE *f = fopen(filename, "w");
	assert(f != NULL);
	fprintf(f, "[[");
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j)
			if (i + 1 != cell_rows || j + 1 != cell_cols)
				fprintf(f, "%lf, ", X(j));
			else
				fprintf(f, "%lf", X(j));
	}
	fprintf(f, "],[\n");
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j)
			if (i + 1 != cell_rows || j + 1 != cell_cols)
				fprintf(f, "%lf, ", Y(i));
			else
				fprintf(f, "%lf", Y(i));
	}
	fprintf(f, "],[\n");
	for (int i = 0; i < cell_rows; ++i) {
		for (int j = 0; j < cell_cols; ++j)
			if (i + 1 != cell_rows || j + 1 != cell_cols)
				fprintf(f, "%lf, ", get_cell(matrix, i, j));
			else
				fprintf(f, "%lf", get_cell(matrix, i, j));
	}
	fprintf(f, "]]");
	if (borders != NULL) {
		fprintf(f, "\n");
		if (borders[BOTTOM_BORDER] != NULL) {
			fprintf(f, "Bottom: ");
			for (int i = 0; i < cell_cols; ++i)
				fprintf(f, "%lf, ", borders[BOTTOM_BORDER][i]);
			fprintf(f, "\n");
		}
		if (borders[TOP_BORDER] != NULL) {
			fprintf(f, "Top: ");
			for (int i = 0; i < cell_cols; ++i)
				fprintf(f, "%lf, ", borders[TOP_BORDER][i]);
			fprintf(f, "\n");
		}
		if (borders[LEFT_BORDER] != NULL) {
			fprintf(f, "Left: ");
			for (int i = 0; i < cell_rows; ++i)
				fprintf(f, "%lf, ", borders[LEFT_BORDER][i]);
			fprintf(f, "\n");
		}
		if (borders[RIGHT_BORDER] != NULL) {
			fprintf(f, "Right: ");
			for (int i = 0; i < cell_rows; ++i)
				fprintf(f, "%lf, ", borders[RIGHT_BORDER][i]);
			fprintf(f, "\n");
		}
	}
	fclose(f);
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
	h1_in_2_reverted = 1 / (x_step * x_step);
	h2_in_2_reverted = 1 / (y_step * y_step);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
	int proc_count;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
	if (calculate_cells(table_height, table_width, proc_count) != 0) {
		printf("Cannot split table in a specified process count\n");
		goto error;
	}
	create_matrices();

	double start_time = MPI_Wtime();
	calculate();
	double end_time = MPI_Wtime();
	if (proc_rank == 0)
		printf("time = %lf\n", end_time - start_time);

	dump_matrix(P, "P", NULL);

	free(P);
	for (int i = 0; i < 4; ++i)
		free(P_neigh[i]);

	MPI_Finalize();
	return 0;
error:
	MPI_Finalize();
	return -1;
}
