#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#define SDL_MAIN_USE_CALLBACKS

#include "SDL3/SDL_error.h"
#include "SDL3/SDL_log.h"
#include "SDL3/SDL_pixels.h"
#include "SDL3/SDL_render.h"
#include "SDL3/SDL_scancode.h"
#include <stdio.h>
#include <stdlib.h>

#include "SDL3/SDL_events.h"
#include "SDL3/SDL_init.h"
#include "SDL3/SDL_video.h"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#define WIDTH 1280
#define HEIGHT 720

#define CUDA_CHECK(expr_to_check)                                                                          \
	do {                                                                                               \
		cudaError_t result = expr_to_check;                                                        \
		if (result != cudaSuccess) {                                                               \
			fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__, __LINE__, result, \
				cudaGetErrorString(result));                                               \
		}                                                                                          \
	} while (0)

struct AppState {
	SDL_Window *window;
	SDL_Renderer *renderer;
};

void enumerate_cuda_devices(void)
{
	int device_count;
	CUDA_CHECK(cudaGetDeviceCount(&device_count));

	printf("Found %d CUDA devices\n", device_count);

	for (int device_i = 0; device_i < device_count; device_i++) {
		cudaDeviceProp device_prop;
		CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_i));

		printf("Device %d properties:\n", device_i);
		printf(" name                     : %s\n", device_prop.name);
		printf(" total global mem         : %lu bytes\n", device_prop.totalGlobalMem);
		printf(" total const mem          : %lu bytes\n", device_prop.totalConstMem);
		printf(" shared mem per block     : %lu bytes\n", device_prop.sharedMemPerBlock);
		printf(" registers per block      : %d\n", device_prop.regsPerBlock);
		printf(" warp size                : %d\n", device_prop.warpSize);
		printf(" max threads per block    : %d\n", device_prop.maxThreadsPerBlock);
		printf(" max threads dim          : [%d, %d, %d]\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
		       device_prop.maxThreadsDim[2]);
		printf(" max grid size            : [%d, %d, %d]\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1],
		       device_prop.maxGridSize[2]);
		printf(" clock rate               : %dkHz\n", device_prop.clockRate);
		printf(" major/minor capabilities : %d/%d\n", device_prop.major, device_prop.minor);
		printf(" multiprocessor count     : %d\n", device_prop.multiProcessorCount);
		printf(" max threads per SM       : %d\n", device_prop.maxThreadsPerMultiProcessor);
		printf(" shared memory per SM     : %lu bytes\n", device_prop.sharedMemPerMultiprocessor);
		printf(" regs per SM              : %d\n", device_prop.regsPerMultiprocessor);
		printf(" memory clock rate        : %dkHz\n", device_prop.memoryClockRate);
		printf(" memory bus width         : %d bits\n", device_prop.memoryBusWidth);
		printf(" ecc enabled              : %d\n", device_prop.ECCEnabled);
		printf(" L2 cache size            : %d bytes\n", device_prop.l2CacheSize);
	}
}

SDL_AppResult SDL_AppInit(void **raw_appstate, int argc, char **argv)
{
	AppState *app_state = (AppState *)malloc(sizeof(AppState));
	*raw_appstate = (void *)app_state;

	SDL_SetAppMetadata("fluid-sim", "0.9.0", "xyz.bengel.fluidsim");

	if (!SDL_Init(SDL_INIT_VIDEO)) {
		SDL_Log("Couldn't initialize SDL: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	if (!SDL_CreateWindowAndRenderer("fluid-sim", WIDTH, HEIGHT, SDL_WINDOW_RESIZABLE, &app_state->window,
					 &app_state->renderer)) {
		SDL_Log("Couldn't create renderer or window: %s\n", SDL_GetError());
		return SDL_APP_FAILURE;
	}

	SDL_SetRenderLogicalPresentation(app_state->renderer, WIDTH, HEIGHT, SDL_LOGICAL_PRESENTATION_LETTERBOX);

	enumerate_cuda_devices();

	return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void *raw_appstate)
{
	AppState *app_state = (AppState *)raw_appstate;

	SDL_SetRenderDrawColorFloat(app_state->renderer, 0.0f, 0.0f, 0.0f, SDL_ALPHA_OPAQUE_FLOAT);
	SDL_RenderClear(app_state->renderer);

	/* put the newly-cleared rendering on the screen. */
	SDL_RenderPresent(app_state->renderer);

	return SDL_APP_CONTINUE; /* carry on with the program! */
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
