#pragma once

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAT(A, B)     CAT_EXP(A, B)
#define CAT_EXP(A, B) A##B

typedef struct string {
    const char *f, *v;
} string;

string until(char c, char const **itr);

void append(const string *s, char **itr);
void appends(const char *str, char **itr);

string strlit(const char *s, size_t n);
#define STRLIT(S) (strlit(S, sizeof(S) - 1))

void center(uint32_t nfield, const string *pad, string *s, char **itr);
