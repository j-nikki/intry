#include "util.h"

int main(int argc, char *argv[])
{
    int cols = atoi(argv[3]);
    cols     = cols ? cols : 80;
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "couldn't open %s: %s\n", argv[1], strerror(errno));
        return 1;
    }
    union {
        uint8_t arr[3];
        struct {
            uint8_t wpkey;
            uint8_t wlkey;
            uint8_t wtkey;
        };
    } u;
    if (fread(&u.arr[0], 1, 3, f) != 3) {
        fprintf(stderr, "failed to read %s - please rerun populate logic\n", argv[1]);
        fclose(f);
        return 1;
    }
    fclose(f);
    const char *itr = argv[2];
    char buf[1024];
    char *dit = buf;
#define APPN(N, Str)                                                                               \
    do {                                                                                           \
        for (uint32_t CAT(n, __LINE__) = (uint32_t)(N); CAT(n, __LINE__)--;)                       \
            append(Str, &dit);                                                                     \
    } while (0)

    const string tl = STRLIT("╔"), tr = STRLIT("╗"), h = STRLIT("═"), v = STRLIT("║"),
                 bl = STRLIT("╚"), br = STRLIT("╝"), sep = STRLIT("│"), tsep = STRLIT("╤"),
                 bsep = STRLIT("╧"), pad = STRLIT(" ");

    const int narrow = u.wpkey + u.wlkey + u.wtkey > cols;
    if (narrow) {
        appends("\x1b[2mPlatform|Cl|CPI\x1b[0m\n", &dit);
        while (*itr) {
            string platform = until('$', &itr);
            append(&platform, &dit);
            appends("\x1b[2m|\x1b[0m", &dit);
            string latency = until('$', &itr);
            append(&latency, &dit);
            appends("\x1b[2m|\x1b[0m", &dit);
            string throughput = until(',', &itr);
            append(&throughput, &dit);
            *dit++ = '\n';
        }
    } else {
        appends("\x1b[2m", &dit);
        append(&tl, &dit);
        string plabel = STRLIT("Platform"), llabel = STRLIT("Cl"), tlabel = STRLIT("CPI");
        center(u.wpkey, &h, &plabel, &dit);
        append(&tsep, &dit);
        center(u.wlkey, &h, &llabel, &dit);
        append(&tsep, &dit);
        center(u.wtkey, &h, &tlabel, &dit);
        append(&tr, &dit);
        *dit++ = '\n';
        while (*itr) {
            append(&v, &dit);
            string platform = until('$', &itr);
            center(u.wpkey, &pad, &platform, &dit);
            append(&sep, &dit);
            string latency = until('$', &itr);
            center(u.wlkey, &pad, &latency, &dit);
            append(&sep, &dit);
            string throughput = until(',', &itr);
            center(u.wtkey, &pad, &throughput, &dit);
            append(&v, &dit);
            *dit++ = '\n';
        }
        append(&bl, &dit);
        APPN(u.wpkey, &h);
        append(&bsep, &dit);
        APPN(u.wlkey, &h);
        append(&bsep, &dit);
        APPN(u.wtkey, &h);
        append(&br, &dit);
        appends("\x1b[0m", &dit);
        *dit++ = '\n';
    }
    fwrite(buf, 1, (size_t)(dit - buf), stdout);
}
