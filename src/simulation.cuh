#include <stdint.h>
#include <stddef.h>
#include <vector>

class Grid {
    public:
	Grid(uint32_t width, uint32_t height);

	uint8_t get(uint32_t x, uint32_t y) const;
	void set(uint32_t x, uint32_t y, uint8_t value);
	void clear();

	uint32_t width() const;
	uint32_t height() const;

	void fill(uint32_t x, uint32_t y, uint8_t value, uint32_t radius);

	uint8_t *data();

    private:
	uint32_t m_width;
	uint32_t m_height;
	std::vector<uint8_t> m_cells;
};

__global__ void render_grid(const uint8_t *cells, uint32_t width, uint32_t height, uint32_t *pixel_data);
__global__ void step_simulation(uint8_t *cells_new, const uint8_t *cells, uint32_t width, uint32_t height);

__device__ uint8_t get_cell_value_checked(const uint8_t *cells, uint32_t width, uint32_t height, size_t x, size_t y);
__device__ void set_cell_value_checked(uint8_t *cells, uint32_t width, uint32_t height, size_t x, size_t y,
				       uint8_t value);
