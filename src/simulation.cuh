#include <stdint.h>
#include <stddef.h>
#include <vector>

class Grid {
    public:
	Grid(uint32_t width, uint32_t height);

	bool is_set(uint32_t x, uint32_t y) const;
	void set(uint32_t x, uint32_t y);
	void unset(uint32_t x, uint32_t y);
	void clear();

	uint32_t width() const;
	uint32_t height() const;

        void fill(uint32_t x, uint32_t y, uint32_t radius);

        const uint8_t *data() const;

    private:
	uint32_t m_width;
	uint32_t m_height;
	std::vector<uint8_t> m_cells;
};

__global__ void step_once(const uint8_t *cells, uint32_t width, uint32_t height, uint32_t *pixel_data);
