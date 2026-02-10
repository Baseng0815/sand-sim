#include "simulation.cuh"
#include <algorithm>
#include <assert.h>
#include <cstdio>
#include <cstring>
#include <vector>

Grid::Grid(uint32_t width, uint32_t height)
	: m_width{ width }
	, m_height{ height }
{
	assert(width * height % 8 == 0);

	m_cells = std::vector<uint8_t>(width * height / 8);
	this->clear();
}

bool Grid::is_set(uint32_t x, uint32_t y) const
{
	assert(x < m_width && y < m_height);

	size_t index = static_cast<size_t>(y * m_width + x);
	size_t byte_index = index / 8;
	size_t bit_index = index % 8;

	uint8_t byte = m_cells[byte_index];
	return ((byte >> bit_index) & 1) == 1;
}

void Grid::set(uint32_t x, uint32_t y)
{
	assert(x < m_width && y < m_height);

	size_t index = static_cast<size_t>(y * m_width + x);
	size_t byte_index = index / 8;
	size_t bit_index = index % 8;

	m_cells[byte_index] |= (1 << bit_index);
}

void Grid::unset(uint32_t x, uint32_t y)
{
	assert(x < m_width && y < m_height);

	size_t index = static_cast<size_t>(y * m_width + x);
	size_t byte_index = index / 8;
	size_t bit_index = index % 8;

	m_cells[byte_index] &= ~(1 << bit_index);
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

void Grid::fill(uint32_t x, uint32_t y, uint32_t radius)
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

			this->set(static_cast<uint32_t>(x_offsetted), static_cast<uint32_t>(y_offsetted));
		}
	}
}

const uint8_t *Grid::data() const
{
	return m_cells.data();
}

__global__ void step_once(const uint8_t *cells, uint32_t width, uint32_t height, uint32_t *pixel_data)
{
	size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
	size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
	size_t idx_cell = idx_y * static_cast<size_t>(width) + idx_x;

	if (idx_cell >= static_cast<size_t>(width) * static_cast<size_t>(height)) {
		return;
	}

	size_t idx_byte = idx_cell / 8;
	size_t idx_bit = idx_cell % 8;

	uint8_t cell_value = (cells[idx_byte] >> idx_bit) & 1;
	if (cell_value > 0) {
		// ARGB
		pixel_data[idx_cell] = 0xFFFFFFFF;
	} else {
		pixel_data[idx_cell] = 0x00000000;
	}
}
