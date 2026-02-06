#include "simulation.h"
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
