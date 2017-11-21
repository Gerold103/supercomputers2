#ifndef HW2_IO_UTILS_H
#define HW2_IO_UTILS_H

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include "cuda_utils.h"

/** Mpi process rank. */
static int proc_rank = -1;
/**
 * The process set is split into a rows and columns. These
 * values specifies, in which cell in the process matrix the
 * current process is placed.
 */
static int proc_column = -1;
static int proc_row = -1;
/** Ranks of a neighbour processes. */
static int ranks_neigh[4];
/** Count of existing borders. */
static int border_count = -1;
/**
 * Each process in a global processes grid owns an XY coordinates
 * matrix. These values specifies the size of this matrix.
 */
static int cell_rows = -1;
static int cell_cols = -1;

/** Sizes of a borders. */
static int border_size[4];
/** From with x_i and y_i the current process owns points grid. */
static int start_x_i = -1;
static int start_y_i = -1;

/** True, if a process is on global grid border. */
static bool is_top = false;
static bool is_bottom = false;
static bool is_left = false;
static bool is_right = false;

/**
 * Scalar functions are calculated in two steps:
 * 1) local calculation in each process;
 * 2) sqrt of global sum of local scalars.
 * But in all places of the algorithm scalars are not calculated
 * single. In all steps we can see only scalar_1 / scalar_2. So
 * lets calculate local scalar_1, local scalar_2 and then in a
 * single reduce calculate global scalar_1 and scalar_2. It
 * reduces count of network messages twice.
 */
static inline double
global_scalar_fraction(double local_numerator, double local_denominator)
{
	double local_buf[2], global_buf[2];
	/* Send two local scalar in a single message. */
	local_buf[0] = local_numerator;
	local_buf[1] = local_denominator;
	int rc = MPI_Allreduce(local_buf, global_buf, 2, MPI_DOUBLE, MPI_SUM,
			       MPI_COMM_WORLD);
	(void) rc; /* Prune warning in release mode. */
	assert(rc == MPI_SUCCESS);
	return global_buf[0] / global_buf[1];
}

/**
 * Calculate global increment of P using local increment. Global
 * increment is calculated by formula:
 * || P_i+1 - P_i ||.
 * @param local_increment Local increment of P.
 * @param step Grid step (step by X * step by Y).
 *
 * @retval Global increment.
 */
static inline double
global_increment(double local_increment, double step)
{
	double global_increment;
	int rc = MPI_Allreduce(&local_increment, &global_increment, 1,
			       MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	(void) rc; /* Prune warning in release mode. */
	assert(rc == MPI_SUCCESS);
	return sqrt(global_increment * step);
}

/**
 * Async send borders of a matrix to a neighbour processes. Each
 * neighbour process will store borders in its P, R or G borders
 * section.
 * @param border_buffers Buffers to send.
 */
static inline void
send_borders(const double * const *border_buffers)
{
	for (int i = 0; i < 4; ++i) {
		MPI_Request req;
		if (ranks_neigh[i] == -1)
			continue;
		int size = border_size[i];
		int rc = MPI_Isend(border_buffers[i], size, MPI_DOUBLE,
				   ranks_neigh[i], i, MPI_COMM_WORLD, &req);
		(void) rc; /* Prune warning in release mode. */
		assert(rc == MPI_SUCCESS);
		MPI_Request_free(&req);
	}
}

/**
 * Async receive borders from a neighbour processes.
 * @param border_buffers Input border buffers.
 * @param[out] reqs Result array of requests to sync by them.
 */
static inline void
receive_borders(double **border_buffers, MPI_Request *reqs)
{
	for (int i = 0; i < 4; ++i) {
		if (ranks_neigh[i] == -1)
			continue;
		int type_from;
		if (i == BOTTOM_BORDER || i == TOP_BORDER) {
			if (i == BOTTOM_BORDER)
				type_from = TOP_BORDER;
			else
				type_from = BOTTOM_BORDER;
		} else {
			if (i == LEFT_BORDER)
				type_from = RIGHT_BORDER;
			else
				type_from = LEFT_BORDER;
		}
		int size = border_size[i];
		int rc = MPI_Irecv(border_buffers[i], size, MPI_DOUBLE,
				   ranks_neigh[i], type_from, MPI_COMM_WORLD,
				   &reqs[i]);
		(void) rc; /* Prune warning in release mode. */
		assert(rc == MPI_SUCCESS);
	}
}

/**
 * Wait until @a count requests are received.
 * Example:
 * send_borders();
 * // Do some local work.
 * receive_borders();
 * // Do some local work, until borders are needed.
 * sync_receive_borders().
 */
static inline void
sync_receive_borders(MPI_Request *reqs, int count)
{
	int rc = MPI_Waitall(count, reqs, MPI_STATUSES_IGNORE);
	(void) rc; /* Prune warning in release mode. */
	assert(rc == MPI_SUCCESS);
}

/** Next denominator of a value staring from a start+1. */
static inline int
next_denominator(int value, int start)
{
	++start;
	while (start < value && value % start != 0)
		++start;
	return start;
}

/**
 * Calculate cell size of a process using its rank, process count
 * and global grid size. Note, that global grid can be not square.
 * @param table_height Height of a global grid
 *        (count of points on Y).
 * @param table_width Width of a global grid
 *        (count of points on X).
 * @param proc_count Process count.
 *
 * @retval  0 Success.
 * @retval -1 Can not calculate sizes using specified parameters.
 *         It can happen, for example, if it is not possible to
 *         split grid on cells with not breaking condition:
 *         2 * cell_height >= cell_width and
 *         2 * cell_width >= cell_height.
 */
static inline int
calculate_cells(int table_height, int table_width, int proc_count)
{
	int rows = 1;
	int ch = table_height;
	int max_ch = table_height;
	int cols, cw, max_cw;

	do {
		cols = proc_count / rows;
		assert(proc_count % rows == 0);
		cw = table_width / cols;
		if (table_width % cols != 0)
			max_cw = cw + 1;
		else
			max_cw = cw;

		if (2 * max_ch >= cw && 2 * max_cw >= ch)
			break;

		rows = next_denominator(proc_count, rows);
		if (rows > proc_count)
			return -1;

		ch = table_height / rows;
		if (table_height % rows != 0)
			max_ch = ch + 1;
		else
			max_ch = ch;

	} while(true);

	proc_row = proc_rank / cols;
	proc_column = proc_rank % cols;
	cell_rows = ch;
	cell_cols = cw;

	/*
	 * If can not split table on equal cells, then
	 * spread rest of points 1 by 1 on a first cells.
	 */
	int row_points_rest = table_height - ch * rows;
	int col_points_rest = table_width - cw * cols;
	assert(row_points_rest >= 0);
	assert(col_points_rest >= 0);
	if (proc_row + 1 <= row_points_rest)
		cell_rows++;
	if (proc_column + 1 <= col_points_rest)
		cell_cols++;

	start_x_i = cell_cols * proc_column;
	start_y_i = cell_rows * proc_row;
	if (proc_row + 1 > row_points_rest)
		start_y_i += row_points_rest;
	if (proc_column + 1 > col_points_rest)
		start_x_i += col_points_rest;

	border_count = 4;
	if (proc_row + 1 == rows) {
		is_top = true;
		--border_count;
	}
	if (proc_row == 0) {
		is_bottom = true;
		--border_count;
	}

	if (proc_column == 0) {
		is_left = true;
		--border_count;
	}
	if (proc_column + 1 == cols) {
		is_right = true;
		--border_count;
	}

	if (! is_top) {
		ranks_neigh[TOP_BORDER] = proc_rank + cols;
		border_size[TOP_BORDER] = cell_cols;
	} else {
		ranks_neigh[TOP_BORDER] = -1;
		border_size[TOP_BORDER] = -1;
	}
	if (! is_bottom) {
		ranks_neigh[BOTTOM_BORDER] = proc_rank - cols;
		border_size[BOTTOM_BORDER] = cell_cols;
	} else {
		ranks_neigh[BOTTOM_BORDER] = -1;
		border_size[BOTTOM_BORDER] = -1;
	}
	if (! is_left) {
		ranks_neigh[LEFT_BORDER] = proc_rank - 1;
		border_size[LEFT_BORDER] = cell_rows;
	} else {
		ranks_neigh[LEFT_BORDER] = -1;
		border_size[LEFT_BORDER] = -1;
	}
	if (! is_right) {
		ranks_neigh[RIGHT_BORDER] = proc_rank + 1;
		border_size[RIGHT_BORDER] = cell_rows;
	} else {
		ranks_neigh[RIGHT_BORDER] = -1;
		border_size[RIGHT_BORDER] = -1;
	}
	return 0;
}

#endif
