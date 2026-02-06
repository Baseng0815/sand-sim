#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_surface.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>
#define SDL_MAIN_USE_CALLBACKS

#include "simulation.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "SDL3/SDL_error.h"
#include "SDL3/SDL_log.h"
#include "SDL3/SDL_pixels.h"
#include "SDL3/SDL_render.h"
#include "SDL3/SDL_scancode.h"
#include <SDL3_ttf/SDL_ttf.h>
#include <stdio.h>
#include <stdlib.h>

#include "SDL3/SDL_events.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define CUDA_CHECK(expr_to_check)                                                                          \
	do {                                                                                               \
		cudaError_t result = expr_to_check;                                                        \
		if (result != cudaSuccess) {                                                               \
			fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result, \
				cudaGetErrorString(result));                                               \
		}                                                                                          \
	} while (0)

const int SAND_DRAW_RADIUS = 16;

class AppState {
    public:
	SDL_Window *window;
	SDL_Renderer *renderer;
	Grid grid;
	int cuda_device;
	TTF_Font *font;
	std::chrono::time_point<std::chrono::high_resolution_clock> last_iteration_start;
};

void render_text(AppState *app_state, std::string &string, int pos_x, int pos_y);

int enumerate_cuda_devices()
{
	int device_count;
	CUDA_CHECK(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr, "No CUDA-enabled devices found, aborting\n");
		exit(1);
	}

	printf("Found %d CUDA devices\n", device_count);

	int chosen_device = 0;
	for (int device_i = 0; device_i < device_count; device_i++) {
		chosen_device = device_i;

		cudaDeviceProp device_prop;
		CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_i));

		printf("Device %d properties:\n", device_i);
		printf("    name                     : %s\n", device_prop.name);
		printf("    total global mem         : %lu bytes\n", device_prop.totalGlobalMem);
		printf("    total const mem          : %lu bytes\n", device_prop.totalConstMem);
		printf("    shared mem per block     : %lu bytes\n", device_prop.sharedMemPerBlock);
		printf("    registers per block      : %d\n", device_prop.regsPerBlock);
		printf("    warp size                : %d\n", device_prop.warpSize);
		printf("    max threads per block    : %d\n", device_prop.maxThreadsPerBlock);
		printf("    max threads dim          : [%d, %d, %d]\n", device_prop.maxThreadsDim[0],
		       device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
		printf("    max grid size            : [%d, %d, %d]\n", device_prop.maxGridSize[0],
		       device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
		printf("    clock rate               : %dkHz\n", device_prop.clockRate);
		printf("    major/minor capabilities : %d/%d\n", device_prop.major, device_prop.minor);
		printf("    multiprocessor count     : %d\n", device_prop.multiProcessorCount);
		printf("    max threads per SM       : %d\n", device_prop.maxThreadsPerMultiProcessor);
		printf("    shared memory per SM     : %lu bytes\n", device_prop.sharedMemPerMultiprocessor);
		printf("    regs per SM              : %d\n", device_prop.regsPerMultiprocessor);
		printf("    memory clock rate        : %dkHz\n", device_prop.memoryClockRate);
		printf("    memory bus width         : %d bits\n", device_prop.memoryBusWidth);
		printf("    ecc enabled              : %d\n", device_prop.ECCEnabled);
		printf("    L2 cache size            : %d bytes\n", device_prop.l2CacheSize);
	}

	printf("Choosing device %d\n", chosen_device);
	return chosen_device;
}

SDL_AppResult SDL_AppInit(void **raw_appstate, int argc, char **argv)
{
	Grid grid = Grid{ 1280, 720 };

	int cuda_device = enumerate_cuda_devices();

	SDL_SetAppMetadata("fluid-sim", "0.9.0", "xyz.bengel.fluidsim");

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		SDL_Log("Couldn't initialize SDL: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	SDL_Window *window;
	SDL_Renderer *renderer;
	if (!SDL_CreateWindowAndRenderer("fluid-sim", grid.width(), grid.height(), SDL_WINDOW_RESIZABLE, &window,
					 &renderer)) {
		SDL_Log("Couldn't create renderer or window: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	SDL_SetRenderLogicalPresentation(renderer, grid.width(), grid.height(), SDL_LOGICAL_PRESENTATION_LETTERBOX);

	if (!TTF_Init()) {
		SDL_Log("Couldn't initialize SDL_ttf: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	TTF_Font *font = TTF_OpenFont("assets/SourceCodePro-Regular.ttf", 18.0f);
	if (!font) {
		SDL_Log("Couldn't open font: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	AppState *app_state = new AppState{ window, renderer, grid, cuda_device, font };
	*raw_appstate = (void *)app_state;

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void *raw_appstate)
{
	AppState *app_state = (AppState *)raw_appstate;

	auto curr_iter_start = std::chrono::high_resolution_clock::now();
	auto last_iter_dur = curr_iter_start - app_state->last_iteration_start;
	auto last_iter_us = std::chrono::duration_cast<std::chrono::microseconds>(last_iter_dur);
	app_state->last_iteration_start = curr_iter_start;

	// event polling
	float mouse_x, mouse_y;
	SDL_MouseButtonFlags mouse_flags = SDL_GetMouseState(&mouse_x, &mouse_y);
	if (mouse_flags & SDL_BUTTON_LEFT) {
		uint32_t x = static_cast<uint32_t>(mouse_x);
		uint32_t y = static_cast<uint32_t>(mouse_y);

		if (x < app_state->grid.width() && y < app_state->grid.height()) {
			app_state->grid.set(x, y);
		}
	}

	// rendering
	SDL_SetRenderDrawColorFloat(app_state->renderer, 0.1f, 0.3f, 0.2f, SDL_ALPHA_OPAQUE_FLOAT);
	SDL_RenderClear(app_state->renderer);

	// points
	SDL_SetRenderDrawColorFloat(app_state->renderer, 1.0, 1.0, 1.0, SDL_ALPHA_OPAQUE_FLOAT);
	for (uint32_t y = 0; y < app_state->grid.height(); y++) {
		for (uint32_t x = 0; x < app_state->grid.width(); x++) {
			if (app_state->grid.is_set(x, y)) {
				SDL_RenderPoint(app_state->renderer, static_cast<float>(x), static_cast<float>(y));
			}
		}
	}

	// text
	std::ostringstream oss;
	oss << "Iteration duration: " << last_iter_us.count() << "us";
	std::string last_iter_us_str = oss.str();
	render_text(app_state, last_iter_us_str, 0, 0);

	SDL_RenderPresent(app_state->renderer);

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void *raw_appstate, SDL_Event *event)
{
	AppState *app_state = (AppState *)raw_appstate;

	switch (event->type) {
	case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
	case SDL_EVENT_QUIT:
		return SDL_APP_SUCCESS;
		break;
	case SDL_EVENT_KEY_DOWN: {
		switch (event->key.scancode) {
		case SDL_SCANCODE_ESCAPE:
		case SDL_SCANCODE_Q:
			return SDL_APP_SUCCESS;
		case SDL_SCANCODE_X:
			app_state->grid.clear();
                        break;
		default:
			break;
		}

		break;
	}

	default:
		break;
	}

	return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void *raw_appstate, SDL_AppResult result)
{
	AppState *app_state = (AppState *)raw_appstate;

	SDL_DestroyRenderer(app_state->renderer);
	SDL_DestroyWindow(app_state->window);
}

void render_text(AppState *app_state, std::string &string, int pos_x, int pos_y)
{
	SDL_Surface *surface = TTF_RenderText_Blended(app_state->font, string.c_str(), string.length(),
						      SDL_Color{ 255, 255, 255, SDL_ALPHA_OPAQUE });
	if (!surface) {
		SDL_Log("Couldn't create text surface: %s\n", SDL_GetError());
	}

	SDL_Texture *texture = SDL_CreateTextureFromSurface(app_state->renderer, surface);

	if (!texture) {
		SDL_Log("Couldn't create text texture: %s\n", SDL_GetError());
	}

	SDL_FRect dst_rect;
	SDL_GetTextureSize(texture, &dst_rect.w, &dst_rect.h);
	dst_rect.x = pos_x;
	dst_rect.y = pos_y;

	SDL_RenderTexture(app_state->renderer, texture, NULL, &dst_rect);

	SDL_DestroyTexture(texture);
	SDL_DestroySurface(surface);
}
