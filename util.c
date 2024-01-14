#include "util.h"

string until(char c, char const **itr)
{
    const char *f = *itr;
    for (char cur; (cur = **itr) != c; ++*itr) {
        if (!cur) {
            fprintf(stderr, "unexpected end of string\n");
            quick_exit(1);
        }
    }
    return (string){.f = f, .v = (*itr)++};
}

void append(const string *s, char **itr)
{
    uint32_t ns     = (uint32_t)(s->v - s->f);
    const char *sit = s->f;
    while (ns--)
        *(*itr)++ = *sit++;
}

void appends(const char *str, char **itr)
{
    while (*str)
        *(*itr)++ = *str++;
}

string strlit(const char *s, size_t n) { return (string){.f = s, .v = s + n}; }

#define STRLIT(S) (strlit(S, sizeof(S) - 1))

void center(uint32_t nfield, const string *pad, string *s, char **itr)
{
    uint32_t ns   = (uint32_t)(s->v - s->f);
    uint32_t lpad = (nfield - ns + 1) / 2;
    uint32_t rpad = (nfield - ns - lpad);
    while (lpad--)
        append(pad, itr);
    appends("\x1b[0m", itr);
    append(s, itr);
    appends("\x1b[2m", itr);
    while (rpad--)
        append(pad, itr);
}
