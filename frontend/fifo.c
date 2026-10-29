#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <sys/stat.h>

/* ODAS uses buffered FILE streams. Disable buffering only for named pipes. */
static FILE *open_fifo(const char *name, const char *mode, const char *symbol) {
    FILE *(*original)(const char *, const char *) = dlsym(RTLD_NEXT, symbol);
    FILE *stream = original(name, mode);
    struct stat info;
    if (stream && fstat(fileno(stream), &info) == 0 && S_ISFIFO(info.st_mode)) {
        setvbuf(stream, NULL, _IONBF, 0);
    }
    return stream;
}

FILE *fopen(const char *name, const char *mode) {
    return open_fifo(name, mode, "fopen");
}

FILE *fopen64(const char *name, const char *mode) {
    return open_fifo(name, mode, "fopen64");
}
