#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>
#include <pthread.h>
#include <stdint.h>
#include <stdatomic.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// PCG32 for high-quality random numbers
typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double pcg32_double(pcg32_random_t* rng) {
    return (double)pcg32_random_r(rng) / 4294967296.0;
}

typedef struct {
    uint64_t balls;
    char out[256];
    int workers;
    uint32_t width;
    uint32_t height;
    int levels;
    double peg_radius;
    double ball_radius;
    double peg_spacing_x;
    double peg_spacing_y;
    double gravity;
    double restitution;
    double fuzz;
    double dt;
} Config;

void config_init(Config* cfg) {
    cfg->balls = 1000000;
    strcpy(cfg->out, "galton_board.png");
    cfg->workers = 0; // 0 means auto
    cfg->width = 1920;
    cfg->height = 1080;
    cfg->levels = 100;
    cfg->peg_radius = 2.0;
    cfg->ball_radius = 1.5;
    cfg->peg_spacing_x = 18.0;
    cfg->peg_spacing_y = 15.0;
    cfg->gravity = 981.0;
    cfg->restitution = 0.6;
    cfg->fuzz = 10.0;
    cfg->dt = 0.005;
}

int config_validate(const Config* cfg) {
    if (cfg->width == 0 || cfg->height == 0) return 0;
    if (cfg->balls == 0) return 0;
    if (cfg->levels <= 0) return 0;
    if (cfg->peg_radius <= 0.0 || cfg->ball_radius <= 0.0) return 0;
    if (cfg->peg_spacing_x <= 0.0 || cfg->peg_spacing_y <= 0.0) return 0;
    if (cfg->gravity <= 0.0) return 0;
    if (cfg->dt <= 0.0) return 0;
    if (strlen(cfg->out) == 0) return 0;
    return 1;
}

typedef struct {
    const Config* cfg;
    atomic_uint_fast64_t* dist;
    uint64_t balls_to_simulate;
    uint64_t seed;
} ThreadData;

size_t simulate_ball(const Config* cfg, pcg32_random_t* rng) {
    double x = (double)cfg->width / 2.0;
    x += (pcg32_double(rng) - 0.5) * 0.1;
    double y = 0.0;
    double vx = 0.0;
    double vy = 0.0;

    double max_y = (double)cfg->levels * cfg->peg_spacing_y;
    double min_dist = cfg->peg_radius + cfg->ball_radius;
    double min_dist_sq = min_dist * min_dist;
    double half_dt_sq_gravity = 0.5 * cfg->gravity * cfg->dt * cfg->dt;

    while (y < max_y) {
        x += vx * cfg->dt;
        y += vy * cfg->dt + half_dt_sq_gravity;
        vy += cfg->gravity * cfg->dt;

        if (x < 0.0) {
            x = 0.0;
            vx = -vx * cfg->restitution;
        } else if (x > (double)(cfg->width - 1)) {
            x = (double)(cfg->width - 1);
            vx = -vx * cfg->restitution;
        }

        int row = (int)round(y / cfg->peg_spacing_y);
        if (row >= 0 && row < cfg->levels) {
            double peg_y = (double)row * cfg->peg_spacing_y;
            double offset = (row % 2 != 0) ? cfg->peg_spacing_x / 2.0 : 0.0;

            int col = (int)round((x - offset) / cfg->peg_spacing_x);
            double peg_x = (double)col * cfg->peg_spacing_x + offset;

            double dx = x - peg_x;
            double dy = y - peg_y;
            double dist_sq = dx * dx + dy * dy;

            if (dist_sq < min_dist_sq && dist_sq > 0.0001) {
                double d = sqrt(dist_sq);
                double nx = dx / d;
                double ny = dy / d;

                double overlap = min_dist - d;
                x += nx * overlap;
                y += ny * overlap;

                double dot = vx * nx + vy * ny;
                if (dot < 0.0) {
                    vx = (vx - 2.0 * dot * nx) * cfg->restitution;
                    vy = (vy - 2.0 * dot * ny) * cfg->restitution;
                    vx += (pcg32_double(rng) - 0.5) * cfg->fuzz;
                    vy += (pcg32_double(rng) - 0.5) * cfg->fuzz;
                }
            }
        }
    }

    int bin = (int)round(x);
    if (bin < 0) return 0;
    if (bin >= (int)cfg->width) return (size_t)cfg->width - 1;
    return (size_t)bin;
}

void* simulation_thread(void* arg) {
    ThreadData* td = (ThreadData*)arg;
    pcg32_random_t rng;
    rng.state = td->seed;
    rng.inc = (uint64_t)td; // Use pointer as unique stream id

    for (uint64_t i = 0; i < td->balls_to_simulate; ++i) {
        size_t bin = simulate_ball(td->cfg, &rng);
        if (bin < td->cfg->width) {
            atomic_fetch_add_explicit(&td->dist[bin], 1, memory_order_relaxed);
        }
    }
    return NULL;
}

unsigned char lerp(double a, double b, double t) {
    return (unsigned char)(a + (b - a) * t);
}

void render(const Config* cfg, const uint64_t* dist) {
    unsigned char* img = (unsigned char*)malloc(cfg->width * cfg->height * 4);
    memset(img, 0, cfg->width * cfg->height * 4);

    uint64_t max_count = 0;
    for (size_t i = 0; i < cfg->width; ++i) {
        if (dist[i] > max_count) max_count = dist[i];
    }

    if (max_count == 0) {
        stbi_write_png(cfg->out, cfg->width, cfg->height, 4, img, cfg->width * 4);
        free(img);
        return;
    }

    double inv_max_count = 1.0 / (double)max_count;
    uint32_t* bar_heights = (uint32_t*)malloc(cfg->width * sizeof(uint32_t));
    for (size_t x = 0; x < cfg->width; ++x) {
        bar_heights[x] = (uint32_t)((double)dist[x] * inv_max_count * (double)cfg->height);
        if (bar_heights[x] > cfg->height) bar_heights[x] = cfg->height;
    }

    for (uint32_t py = 0; py < cfg->height; ++py) {
        uint32_t y_from_bottom = cfg->height - 1 - py;
        
        // Color mapping
        double t = (double)y_from_bottom / (double)cfg->height;
        unsigned char r, g, b;
        if (t < 0.5) {
            double t2 = t * 2.0;
            r = lerp(10, 220, t2);
            g = lerp(10, 20, t2);
            b = lerp(40, 60, t2);
        } else {
            double t2 = t * 2.0 - 1.0;
            r = lerp(220, 255, t2);
            g = lerp(20, 215, t2);
            b = lerp(60, 0, t2);
        }

        for (uint32_t x = 0; x < cfg->width; ++x) {
            size_t idx = (py * cfg->width + x) * 4;
            if (y_from_bottom < bar_heights[x]) {
                img[idx + 0] = r;
                img[idx + 1] = g;
                img[idx + 2] = b;
                img[idx + 3] = 255;
            } else {
                img[idx + 0] = 10;
                img[idx + 1] = 10;
                img[idx + 2] = 15;
                img[idx + 3] = 255;
            }
        }
    }

    stbi_write_png(cfg->out, cfg->width, cfg->height, 4, img, cfg->width * 4);
    free(img);
    free(bar_heights);
}

int main(int argc, char** argv) {
    Config cfg;
    config_init(&cfg);

    static struct option long_options[] = {
        {"balls", required_argument, 0, 'b'},
        {"out", required_argument, 0, 'o'},
        {"workers", required_argument, 0, 'w'},
        {"width", required_argument, 0, 'W'},
        {"height", required_argument, 0, 'H'},
        {"levels", required_argument, 0, 'l'},
        {"peg-radius", required_argument, 0, 'p'},
        {"ball-radius", required_argument, 0, 'B'},
        {"peg-spacing-x", required_argument, 0, 'x'},
        {"peg-spacing-y", required_argument, 0, 'y'},
        {"gravity", required_argument, 0, 'g'},
        {"restitution", required_argument, 0, 'r'},
        {"fuzz", required_argument, 0, 'f'},
        {"dt", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "b:o:w:W:H:l:p:B:x:y:g:r:f:d:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'b': cfg.balls = strtoull(optarg, NULL, 10); break;
            case 'o': strncpy(cfg.out, optarg, 255); break;
            case 'w': cfg.workers = atoi(optarg); break;
            case 'W': cfg.width = atoi(optarg); break;
            case 'H': cfg.height = atoi(optarg); break;
            case 'l': cfg.levels = atoi(optarg); break;
            case 'p': cfg.peg_radius = atof(optarg); break;
            case 'B': cfg.ball_radius = atof(optarg); break;
            case 'x': cfg.peg_spacing_x = atof(optarg); break;
            case 'y': cfg.peg_spacing_y = atof(optarg); break;
            case 'g': cfg.gravity = atof(optarg); break;
            case 'r': cfg.restitution = atof(optarg); break;
            case 'f': cfg.fuzz = atof(optarg); break;
            case 'd': cfg.dt = atof(optarg); break;
        }
    }

    if (!config_validate(&cfg)) {
        fprintf(stderr, "Configuration validation failed\n");
        return 1;
    }

    if (cfg.workers <= 0) {
        cfg.workers = 4; // Default to 4 if not specified or auto
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    atomic_uint_fast64_t* dist_atomic = (atomic_uint_fast64_t*)calloc(cfg.width, sizeof(atomic_uint_fast64_t));
    
    pthread_t* threads = (pthread_t*)malloc(cfg.workers * sizeof(pthread_t));
    ThreadData* td = (ThreadData*)malloc(cfg.workers * sizeof(ThreadData));

    uint64_t balls_per_worker = cfg.balls / cfg.workers;
    for (int i = 0; i < cfg.workers; ++i) {
        td[i].cfg = &cfg;
        td[i].dist = dist_atomic;
        td[i].balls_to_simulate = (i == cfg.workers - 1) ? (cfg.balls - balls_per_worker * (cfg.workers - 1)) : balls_per_worker;
        td[i].seed = (uint64_t)time(NULL) + i;
        pthread_create(&threads[i], NULL, simulation_thread, &td[i]);
    }

    for (int i = 0; i < cfg.workers; ++i) {
        pthread_join(threads[i], NULL);
    }

    uint64_t* dist = (uint64_t*)malloc(cfg.width * sizeof(uint64_t));
    for (size_t i = 0; i < cfg.width; ++i) {
        dist[i] = atomic_load_explicit(&dist_atomic[i], memory_order_relaxed);
    }

    render(&cfg, dist);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Completed in %.3f seconds. Saved to %s\n", elapsed, cfg.out);

    free(dist_atomic);
    free(threads);
    free(td);
    free(dist);

    return 0;
}
