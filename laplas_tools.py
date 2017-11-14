# Grid settings. Replace them with your values.
# Grid size.
cols = 6
rows = 6
# X and Y steps.
x_step = 0.2
y_step = 0.2

# Replace it with your Phi() function.
def phi(x, y):
	t1 = 1 - x * x
	t2 = 1 - y * y
	return t1 * t1 + t2 * t2

def X(i):
	return i * x_step

def Y(i):
	return i * y_step

# Replace it with your F() function.
def F(x, y):
	return 4 * (2 - 3 * x * x - 3 * y * y)

def cell(m, row, col):
	return m[row * cols + col]

def set_cell(m, row, col, value):
	m[row * cols + col] = value

# Scalar product of matrices a and b in internal points.
def scalar(a, b):
	ret = 0
	for i in range(1, rows - 1):
		for j in range(1, cols - 1):
			ret += x_step * y_step * cell(a, i, j) * cell(b, i, j)
	return ret

# Negative result of Laplas(Matrix m). Creates a new matrix.
def laplas_5_matrix(m):
	ret = m.copy()
	for i in range(1, rows - 1):
		for j in range(1, cols - 1):
			a11 = cell(m, i, j)
			a21 = cell(m, i + 1, j)
			a01 = cell(m, i - 1, j)
			a10 = cell(m, i, j - 1)
			a12 = cell(m, i, j + 1)
			tmp1 = 1.0/(x_step * x_step) * (2*a11 - a01 - a21)
			tmp2 = 1.0/(y_step * y_step) * (2*a11 - a10 - a12)
			set_cell(ret, i, j, (tmp1 + tmp2))
	return ret

# Init P and R.
P = [0] * (cols * rows)
R = P.copy()
for i in range(0, cols):
	set_cell(P, 0, i, phi(X(i), Y(0)))
	set_cell(P, rows - 1, i, phi(X(i), Y(rows - 1)))

for i in range(0, rows):
	set_cell(P, i, 0, phi(X(0), Y(i)))
	set_cell(P, i, cols - 1, phi(X(cols - 1), Y(i)))

# Calculate R_i using formula R_i = -laplas(P_i) - F.
def calculate_next_R():
	matrix_buffer = laplas_5_matrix(P)
	for i in range(1, rows - 1):
		for j in range(1, cols - 1):
			set_cell(R, i, j, cell(matrix_buffer, i, j) - F(X(j), Y(i)))
	for i in range(0, cols):
		set_cell(R, 0, i, 0)
		set_cell(R, rows - 1, i, 0)
	for i in range(0, rows):
		set_cell(R, i, 0, 0)
		set_cell(R, i, cols - 1, 0)

# Print matrix in a readable format.
def print_matrix(m):
	for i in range(rows - 1, -1, -1):
		buf = []
		for j in range(cols):
			buf.append(cell(m, i, j))
		print(buf)

# Calculate P_i+1 using formula P_i+1 = P_i - tau * r_or_g_i.
def calculate_next_P(tau, r_or_g):
	for i in range(rows):
		for j in range(cols):
			old = cell(P, i, j)
			set_cell(P, i, j, old - tau * cell(r_or_g, i, j))

# Calculate G_i using formula G_i = R_i - alpha * G_i-1.
def calculate_next_G(alpha):
	for i in range(rows):
		for j in range(cols):
			old = cell(G, i, j)
			set_cell(G, i, j, cell(R, i, j) - alpha * old)

# Make step 0.
calculate_next_R()

# Make step 1.
G = R.copy()
tmp1 = scalar(R, R)
matrix_buffer = laplas_5_matrix(R)
tmp2 = scalar(matrix_buffer, R)
tau = tmp1 / tmp2
calculate_next_P(tau, R)

# Make step 2.
calculate_next_R()
matrix_buffer = laplas_5_matrix(R)
tmp1 = scalar(matrix_buffer, G)
matrix_buffer = laplas_5_matrix(G)
tmp2 = scalar(matrix_buffer, G)
alpha = tmp1 / tmp2
# ...
