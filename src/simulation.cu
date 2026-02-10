#include "simulation.cuh"
#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <vector>

Grid::Grid(uint32_t width, uint32_t height)
	: m_width{ width }
	, m_height{ height }
	, m_cells(width * height)
{
	this->clear();
}

uint8_t Grid::get(uint32_t x, uint32_t y) const
{
	assert(x < m_width && y < m_height);

	size_t index = static_cast<size_t>(y * m_width + x);
	return m_cells[index];
}

void Grid::set(uint32_t x, uint32_t y, uint8_t value)
{
	assert(x < m_width && y < m_height);

	size_t index = static_cast<size_t>(y * m_width + x);
	m_cells[index] = value;
}

void Grid::clear()
{
	std::fill(m_cells.begin(), m_cells.end(), 0);
}

uint32_t Grid::width() const
{
	return m_width;
}
uint32_t Grid::height() const
{
	return m_height;
}

void Grid::fill(uint32_t x, uint32_t y, uint8_t value, uint32_t radius)
{
	for (int32_t y_offset = -radius; y_offset < static_cast<int32_t>(radius); y_offset++) {
		int32_t y_offsetted = y + y_offset;

		if (y_offsetted < 0 || y_offsetted >= m_height) {
			continue;
		}

		for (int32_t x_offset = -radius; x_offset < static_cast<int32_t>(radius); x_offset++) {
			// we want a nice circle
			if (x_offset * x_offset + y_offset * y_offset > radius * radius) {
				continue;
			}

			int32_t x_offsetted = x + x_offset;
			if (x_offsetted < 0 || x_offsetted >= m_width) {
				continue;
			}

			this->set(static_cast<uint32_t>(x_offsetted), static_cast<uint32_t>(y_offsetted), value);
		}
	}
}

uint8_t *Grid::data()
{
	return m_cells.data();
}

__global__ void render_grid(const uint8_t *cells, uint32_t width, uint32_t height, uint32_t *pixbuf)
{
	size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
	size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
	size_t idx_cell = idx_y * static_cast<size_t>(width) + idx_x;

	if (idx_cell >= static_cast<size_t>(width) * static_cast<size_t>(height)) {
		return;
	}

	if (cells[idx_cell] > 0) {
		// ARGB
		pixbuf[idx_cell] = 0x2A94F5FF;
	} else {
		pixbuf[idx_cell] = 0x00000000;
	}
}

__global__ void step_simulation(uint8_t *cells_new, const uint8_t *cells, uint32_t width, uint32_t height)
{
	size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
	size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;

	const std::ptrdiff_t coord_offsets[] = {
		-1, -1, -1, 0, -1, 1, 0, -2, 0, -1, 0, 0, 0, 1, 0, 2, 1, -1, 1, 0, 1, 1
	};

	uint16_t neighborhood_value = 0;
	for (size_t coord_offset_idx = 0; coord_offset_idx < sizeof(coord_offsets) / sizeof(coord_offsets[0]) / 2;
	     coord_offset_idx++) {
		size_t idx_offsetted_y = idx_y + coord_offsets[2 * coord_offset_idx + 0];
		size_t idx_offsetted_x = idx_x + coord_offsets[2 * coord_offset_idx + 1];

		uint8_t cell_value =
			get_cell_value_checked(cells, width, height, idx_offsetted_x, idx_offsetted_y) > 0 ? 1 : 0;
		neighborhood_value |= cell_value << coord_offset_idx;
	}

	set_cell_value_checked(cells_new, width, height, idx_x, idx_y, neighborhood_value > 0 ? 1 : 0);
}

__device__ uint8_t get_cell_value_checked(const uint8_t *cells, uint32_t width, uint32_t height, size_t x, size_t y)
{
	if (x >= static_cast<size_t>(width) || y >= static_cast<size_t>(height)) {
		// boundary condition: zero
		return 0;
	}

	size_t idx = y * static_cast<size_t>(width) + x;
	return cells[idx];
}

__device__ void set_cell_value_checked(uint8_t *cells, uint32_t width, uint32_t height, size_t x, size_t y,
				       uint8_t value)
{
	if (x >= static_cast<size_t>(width) || y >= static_cast<size_t>(height)) {
		// boundary condition: zero
		return;
	}

	size_t idx = y * width + x;
	cells[idx] = value;
}
