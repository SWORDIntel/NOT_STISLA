/**
 * KEYSTONE - tar.zst streaming search implementation
 *
 * Uses libarchive for tar + zstd streaming, minimal in-memory parsing,
 * arena allocation, and member offset indexing.
 */

#include "../include/keystone_tar_zst.h"
#include "../include/keystone.h"
#include <archive.h>
#include <archive_entry.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <ctype.h>
#include <stdarg.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================================
 * Arena Allocator
 * ============================================================================ */

typedef struct keystone_arena_slab {
    struct keystone_arena_slab* next;
    size_t size;
    size_t used;
    unsigned char data[];
} keystone_arena_slab_t;

typedef struct {
    keystone_arena_slab_t* slabs;
    size_t slab_size;
} keystone_arena_t;

static keystone_arena_t* arena_create(size_t slab_size) {
    keystone_arena_t* arena = calloc(1, sizeof(keystone_arena_t));
    if (!arena) return NULL;
    arena->slab_size = slab_size ? slab_size : (1u << 20); /* 1 MiB default */
    return arena;
}

static void arena_destroy(keystone_arena_t* arena) {
    if (!arena) return;
    keystone_arena_slab_t* s = arena->slabs;
    while (s) {
        keystone_arena_slab_t* next = s->next;
        free(s);
        s = next;
    }
    free(arena);
}

static void arena_reset(keystone_arena_t* arena) {
    if (!arena) return;
    keystone_arena_slab_t* s = arena->slabs;
    while (s) {
        s->used = 0;
        s = s->next;
    }
}

static void* arena_alloc(keystone_arena_t* arena, size_t n) {
    if (!arena || n == 0) return NULL;
    /* Try existing slabs */
    keystone_arena_slab_t* s = arena->slabs;
    while (s) {
        if (s->used + n <= s->size) {
            void* p = &s->data[s->used];
            s->used += n;
            return p;
        }
        s = s->next;
    }
    /* Need new slab */
    size_t alloc_size = arena->slab_size;
    if (n > alloc_size) alloc_size = n;
    keystone_arena_slab_t* slab = malloc(sizeof(keystone_arena_slab_t) + alloc_size);
    if (!slab) return NULL;
    slab->next = arena->slabs;
    slab->size = alloc_size;
    slab->used = n;
    arena->slabs = slab;
    return slab->data;
}

/* ============================================================================
 * Member Offset Index (simple hash table)
 * ============================================================================ */

#define TAR_ZST_INDEX_BUCKETS 256

/* Forward declaration: defined after hash functions */
typedef struct tar_zst_bloom tar_zst_bloom_t;

static void tar_zst_bloom_destroy(tar_zst_bloom_t *b);
static tar_zst_bloom_t* tar_zst_bloom_create(size_t expected_elements);
static void tar_zst_bloom_add(tar_zst_bloom_t *b, int64_t key);
static int tar_zst_bloom_may_contain(tar_zst_bloom_t *b, int64_t key);

typedef struct tar_zst_index_entry {
    char* name;
    size_t name_len;
    uint64_t compressed_offset;
    uint64_t uncompressed_offset;
    size_t key_count;
    int64_t first_key;
    int64_t last_key;
    tar_zst_bloom_t *bloom;  /* compact negative-lookup filter */
    int64_t* keys;           /* retained sorted keys (NULL if not retained) */
    size_t keys_capacity;    /* allocated capacity of keys[] */
} tar_zst_index_entry_t;

typedef struct {
    tar_zst_index_entry_t* buckets[TAR_ZST_INDEX_BUCKETS];
    size_t bucket_counts[TAR_ZST_INDEX_BUCKETS];
    size_t bucket_caps[TAR_ZST_INDEX_BUCKETS];
} tar_zst_index_t;

static uint32_t tar_zst_hash_name(const char* name, size_t len) {
    uint32_t h = 5381;
    for (size_t i = 0; i < len; i++) {
        h = ((h << 5) + h) + (unsigned char)name[i];
    }
    return h;
}

static tar_zst_index_t* tar_zst_index_create(void) {
    return calloc(1, sizeof(tar_zst_index_t));
}

static void tar_zst_index_destroy(tar_zst_index_t* idx) {
    if (!idx) return;
    for (size_t b = 0; b < TAR_ZST_INDEX_BUCKETS; b++) {
        tar_zst_index_entry_t* entries = idx->buckets[b];
        for (size_t i = 0; i < idx->bucket_counts[b]; i++) {
            free(entries[i].name);
            tar_zst_bloom_destroy(entries[i].bloom);
            free(entries[i].keys);
        }
        free(entries);
    }
    free(idx);
}

static tar_zst_index_entry_t* tar_zst_index_find(tar_zst_index_t* idx,
                                                  const char* name,
                                                  size_t name_len) {
    if (!idx || !name) return NULL;
    uint32_t h = tar_zst_hash_name(name, name_len);
    size_t b = h & (TAR_ZST_INDEX_BUCKETS - 1);
    tar_zst_index_entry_t* entries = idx->buckets[b];
    for (size_t i = 0; i < idx->bucket_counts[b]; i++) {
        if (entries[i].name_len == name_len &&
            memcmp(entries[i].name, name, name_len) == 0) {
            return &entries[i];
        }
    }
    return NULL;
}

static int tar_zst_index_add(tar_zst_index_t* idx,
                              const char* name,
                              size_t name_len,
                              uint64_t compressed_offset,
                              uint64_t uncompressed_offset,
                              size_t key_count,
                              int64_t first_key,
                              int64_t last_key,
                              tar_zst_bloom_t* bloom,
                              const int64_t* sorted_keys) {
    if (!idx || !name) return -1;
    uint32_t h = tar_zst_hash_name(name, name_len);
    size_t b = h & (TAR_ZST_INDEX_BUCKETS - 1);

    tar_zst_index_entry_t* entries = idx->buckets[b];
    size_t count = idx->bucket_counts[b];
    size_t cap = idx->bucket_caps[b];

    if (count >= cap) {
        size_t new_cap = cap ? cap * 2 : 4;
        tar_zst_index_entry_t* new_entries = realloc(entries,
                                                        new_cap * sizeof(*new_entries));
        if (!new_entries) return -1;
        idx->buckets[b] = new_entries;
        idx->bucket_caps[b] = new_cap;
        entries = new_entries;
    }

    char* name_copy = malloc(name_len + 1);
    if (!name_copy) return -1;
    memcpy(name_copy, name, name_len);
    name_copy[name_len] = '\0';

    /* Retain a private copy of the sorted keys so positive lookups
     * can search directly without re-decompressing the archive member. */
    int64_t* keys_copy = NULL;
    if (sorted_keys && key_count > 0) {
        keys_copy = malloc(key_count * sizeof(int64_t));
        if (!keys_copy) {
            free(name_copy);
            return -1;
        }
        memcpy(keys_copy, sorted_keys, key_count * sizeof(int64_t));
    }

    entries[count].name = name_copy;
    entries[count].name_len = name_len;
    entries[count].compressed_offset = compressed_offset;
    entries[count].uncompressed_offset = uncompressed_offset;
    entries[count].key_count = key_count;
    entries[count].first_key = first_key;
    entries[count].last_key = last_key;
    entries[count].bloom = bloom;
    entries[count].keys = keys_copy;
    entries[count].keys_capacity = key_count;
    idx->bucket_counts[b] = count + 1;
    return 0;
}

/* ============================================================================
 * Bloom Filter (compact negative-lookup for member keys)
 * ============================================================================ */

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
}

static inline uint64_t hash_int64_1(int64_t key) {
    return splitmix64((uint64_t)key);
}

static inline uint64_t hash_int64_2(int64_t key) {
    /* FNV-1a 64-bit */
    uint64_t h = 0xcbf29ce484222325ULL;
    uint8_t *p = (uint8_t*)&key;
    for (size_t i = 0; i < sizeof(key); i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

typedef struct tar_zst_bloom {
    uint64_t *bits;
    size_t num_bits;
    size_t num_hashes;
} tar_zst_bloom_t;

static tar_zst_bloom_t* tar_zst_bloom_create(size_t expected_elements) {
    tar_zst_bloom_t *b = calloc(1, sizeof(tar_zst_bloom_t));
    if (!b) return NULL;
    b->num_hashes = 2;
    size_t num_bits = expected_elements * 8;
    if (num_bits < 1024) num_bits = 1024;
    size_t words = (num_bits + 63) / 64;
    b->bits = calloc(words, sizeof(uint64_t));
    if (!b->bits) {
        free(b);
        return NULL;
    }
    b->num_bits = num_bits;
    return b;
}

static void tar_zst_bloom_destroy(tar_zst_bloom_t *b) {
    if (!b) return;
    free(b->bits);
    free(b);
}

static void tar_zst_bloom_add(tar_zst_bloom_t *b, int64_t key) {
    if (!b) return;
    uint64_t h1 = hash_int64_1(key);
    uint64_t h2 = hash_int64_2(key);
    for (size_t i = 0; i < b->num_hashes; i++) {
        uint64_t pos = (h1 + i * h2) % b->num_bits;
        b->bits[pos >> 6] |= (1ULL << (pos & 63));
    }
}

static int tar_zst_bloom_may_contain(tar_zst_bloom_t *b, int64_t key) {
    if (!b || !b->bits) return 0;
    uint64_t h1 = hash_int64_1(key);
    uint64_t h2 = hash_int64_2(key);
    for (size_t i = 0; i < b->num_hashes; i++) {
        uint64_t pos = (h1 + i * h2) % b->num_bits;
        if ((b->bits[pos >> 6] & (1ULL << (pos & 63))) == 0) {
            return 0; /* definitely not present */
        }
    }
    return 1; /* may be present */
}

/* ============================================================================
 * Parse Layer
 * ============================================================================ */

typedef enum {
    PARSE_CSV,
    PARSE_JSON,
    PARSE_TEXT
} parse_mode_t;

typedef struct parse_ctx {
    parse_mode_t mode;
    int skip_header;
    int header_skipped;
    int in_array;      /* JSON: inside [...] */
    /* Bounded streaming integer parser state.
     * Numbers are accumulated across chunk boundaries so we never
     * need to buffer an entire member, and never read past buf+len. */
    int in_number;     /* 1 if currently accumulating digits */
    int sign;          /* +1 or -1 */
    uint64_t accum;    /* unsigned accumulator (avoids signed UB) */
    unsigned digit_count;
    int overflow;      /* set if the number exceeds int64_t range */
    size_t count;
    int64_t first_key;
    int64_t last_key;
    keystone_arena_t* arena;
    int64_t* keys;
    size_t capacity;
} parse_ctx_t;

static inline int parse_ctx_grow(parse_ctx_t* ctx) {
    if (ctx->count < ctx->capacity) return 0;
    size_t new_cap = ctx->capacity ? ctx->capacity * 2 : 1024;
    int64_t* new_keys = arena_alloc(ctx->arena, new_cap * sizeof(int64_t));
    if (!new_keys) return -1;
    if (ctx->keys && ctx->count > 0) {
        memcpy(new_keys, ctx->keys, ctx->count * sizeof(int64_t));
    }
    ctx->keys = new_keys;
    ctx->capacity = new_cap;
    return 0;
}

static inline void parse_ctx_emit(parse_ctx_t* ctx, int64_t val) {
    if (parse_ctx_grow(ctx) != 0) return;
    ctx->keys[ctx->count] = val;
    if (ctx->count == 0) ctx->first_key = val;
    ctx->last_key = val;
    ctx->count++;
}

/* ============================================================================
 * Bounded Streaming Integer Parser
 *
 * Consumes exactly [buf, buf+len) — never reads past the buffer.  Numbers
 * that span chunk boundaries are accumulated across calls via ctx state.
 * This eliminates the OOB-read hazard of strtoll() and removes the need to
 * buffer an entire decompressed member before parsing.
 * ============================================================================ */

static inline int parse_is_digit(char c) { return c >= '0' && c <= '9'; }

static inline void parse_number_start(parse_ctx_t* ctx, int sign) {
    ctx->in_number = 1;
    ctx->sign = sign;
    ctx->accum = 0;
    ctx->digit_count = 0;
    ctx->overflow = 0;
}

static inline void parse_number_digit(parse_ctx_t* ctx, char c) {
    if (ctx->overflow) return;
    ctx->digit_count++;
    /* int64_t max is 9223372036854775807 (19 digits).  Any 20+ digit
     * sequence overflows.  We also detect uint64 overflow below. */
    if (ctx->digit_count > 19) {
        ctx->overflow = 1;
        return;
    }
    unsigned d = (unsigned)(c - '0');
    uint64_t next = ctx->accum * 10u + d;
    if (next < ctx->accum) {          /* unsigned wrap → overflow */
        ctx->overflow = 1;
        return;
    }
    ctx->accum = next;
}

static inline void parse_number_end(parse_ctx_t* ctx) {
    if (!ctx->in_number) return;
    ctx->in_number = 0;
    if (ctx->digit_count == 0 || ctx->overflow) return;  /* discard */

    if (ctx->sign > 0) {
        if (ctx->accum > (uint64_t)INT64_MAX) return;    /* out of range */
        parse_ctx_emit(ctx, (int64_t)ctx->accum);
    } else {
        /* INT64_MIN abs value is 9223372036854775808 = INT64_MAX + 1 */
        if (ctx->accum > (uint64_t)INT64_MAX + 1u) return;
        if (ctx->accum == (uint64_t)INT64_MAX + 1u)
            parse_ctx_emit(ctx, INT64_MIN);
        else
            parse_ctx_emit(ctx, -(int64_t)ctx->accum);
    }
}

/* Feed a chunk to the text parser. */
static void parse_feed_text(parse_ctx_t* ctx, const char* buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        char c = buf[i];
        if (ctx->in_number) {
            if (parse_is_digit(c)) {
                parse_number_digit(ctx, c);
            } else {
                parse_number_end(ctx);
                if (c == '-' || c == '+')
                    parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        } else {
            if (parse_is_digit(c)) {
                parse_number_start(ctx, 1);
                parse_number_digit(ctx, c);
            } else if (c == '-' || c == '+') {
                parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        }
    }
}

/* Feed a chunk to the CSV parser (same as text, plus header-line skipping). */
static void parse_feed_csv(parse_ctx_t* ctx, const char* buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        char c = buf[i];
        if (c == '\n' && ctx->skip_header && !ctx->header_skipped) {
            /* A number straddling the header newline is flushed first. */
            parse_number_end(ctx);
            ctx->header_skipped = 1;
            continue;
        }
        if (ctx->in_number) {
            if (parse_is_digit(c)) {
                parse_number_digit(ctx, c);
            } else {
                parse_number_end(ctx);
                if (c == '-' || c == '+')
                    parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        } else {
            if (parse_is_digit(c)) {
                parse_number_start(ctx, 1);
                parse_number_digit(ctx, c);
            } else if (c == '-' || c == '+') {
                parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        }
    }
}

/* Feed a chunk to the JSON parser (only inside [...] brackets). */
static void parse_feed_json(parse_ctx_t* ctx, const char* buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        char c = buf[i];
        if (c == '[') {
            ctx->in_array = 1;
            continue;
        }
        if (c == ']') {
            parse_number_end(ctx);
            ctx->in_array = 0;
            continue;
        }
        if (!ctx->in_array) continue;

        if (ctx->in_number) {
            if (parse_is_digit(c)) {
                parse_number_digit(ctx, c);
            } else {
                parse_number_end(ctx);
                if (c == '-' || c == '+')
                    parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        } else {
            if (parse_is_digit(c)) {
                parse_number_start(ctx, 1);
                parse_number_digit(ctx, c);
            } else if (c == '-' || c == '+') {
                parse_number_start(ctx, c == '-' ? -1 : 1);
            }
        }
    }
}

static void parse_feed(parse_ctx_t* ctx, const char* buf, size_t len) {
    switch (ctx->mode) {
        case PARSE_CSV:   parse_feed_csv(ctx, buf, len);   break;
        case PARSE_TEXT:  parse_feed_text(ctx, buf, len);  break;
        case PARSE_JSON:  parse_feed_json(ctx, buf, len);  break;
    }
}

/* Flush any pending number at end-of-member. */
static void parse_finish(parse_ctx_t* ctx) {
    parse_number_end(ctx);
}

/* Proper int64_t comparator for qsort */
static int int64_compare(const void* a, const void* b) {
    int64_t av = *(const int64_t*)a;
    int64_t bv = *(const int64_t*)b;
    return (av > bv) - (av < bv);
}

/* ============================================================================
 * Archive Handle
 * ============================================================================ */

#define KEYSTONE_TAR_ZST_MAX_DECOMPRESS_BYTES (500ULL * 1024 * 1024) /* 500 MB */

struct keystone_tar_zst {
    struct archive* archive;
    struct archive_entry* current_entry;
    keystone_arena_t* arena;
    tar_zst_index_t* index;
    parse_ctx_t parse_ctx;
    keystone_tar_zst_options_t options;
    keystone_tar_zst_stats_t stats;
    char* archive_path;
    char error_buf[512];
    char member_name[512];
    size_t member_name_len;
    int index_built;
    int at_eof;
};

static inline uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void set_error(keystone_tar_zst_t* tz, const char* fmt, ...) {
    if (!tz) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(tz->error_buf, sizeof(tz->error_buf), fmt, args);
    va_end(args);
}

/* ============================================================================
 * Public API
 * ============================================================================ */

keystone_tar_zst_t* keystone_tar_zst_open(const char* path,
                                              const keystone_tar_zst_options_t* opts) {
    if (!path) return NULL;

    keystone_tar_zst_t* tz = calloc(1, sizeof(keystone_tar_zst_t));
    if (!tz) return NULL;

    tz->options.format = KEYSTONE_TAR_ZST_FORMAT_AUTO;
    tz->options.chunk_size = 256 * 1024;
    tz->options.arena_slab_size = 1u << 20;
    tz->options.zstd_workers = 0;
    tz->options.enable_pipeline = 0;
    tz->options.skip_header = 0;

    if (opts) {
        tz->options = *opts;
    }

    tz->arena = arena_create(tz->options.arena_slab_size);
    if (!tz->arena) {
        free(tz);
        return NULL;
    }

    tz->archive = archive_read_new();
    if (!tz->archive) {
        arena_destroy(tz->arena);
        free(tz);
        return NULL;
    }

    archive_read_support_format_tar(tz->archive);
    archive_read_support_filter_zstd(tz->archive);
    archive_read_support_filter_all(tz->archive);

    int r = archive_read_open_filename(tz->archive, path, tz->options.chunk_size);
    if (r != ARCHIVE_OK) {
        set_error(tz, "archive_read_open_filename failed: %s",
                  archive_error_string(tz->archive));
        archive_read_free(tz->archive);
        arena_destroy(tz->arena);
        free(tz);
        return NULL;
    }

    tz->archive_path = strdup(path);
    if (!tz->archive_path) {
        archive_read_free(tz->archive);
        arena_destroy(tz->arena);
        free(tz);
        return NULL;
    }

    return tz;
}

void keystone_tar_zst_close(keystone_tar_zst_t* tz) {
    if (!tz) return;
    if (tz->archive) {
        archive_read_free(tz->archive);
        tz->archive = NULL;
    }
    if (tz->index) {
        tar_zst_index_destroy(tz->index);
        tz->index = NULL;
    }
    if (tz->arena) {
        arena_destroy(tz->arena);
        tz->arena = NULL;
    }
    free(tz->archive_path);
    free(tz);
}

int keystone_tar_zst_next_member(keystone_tar_zst_t* tz,
                                   char** out_name,
                                   size_t* out_name_len) {
    if (!tz || !tz->archive) {
        if (tz) set_error(tz, "Invalid handle");
        return -1;
    }

    if (tz->at_eof) return 0;

    int r = archive_read_next_header(tz->archive, &tz->current_entry);
    if (r == ARCHIVE_EOF) {
        tz->at_eof = 1;
        return 0;
    }
    if (r != ARCHIVE_OK) {
        set_error(tz, "archive_read_next_header failed: %s",
                  archive_error_string(tz->archive));
        return -1;
    }

    const char* name = archive_entry_pathname(tz->current_entry);
    if (!name) name = "";
    size_t n = strlen(name);
    if (n >= sizeof(tz->member_name)) n = sizeof(tz->member_name) - 1;
    memcpy(tz->member_name, name, n);
    tz->member_name[n] = '\0';
    tz->member_name_len = n;

    if (out_name) *out_name = tz->member_name;
    if (out_name_len) *out_name_len = n;

    return 1;
}

/* Stream the current entry chunk-by-chunk through the bounded parser.
 * This avoids buffering the entire decompressed member and eliminates
 * the strtoll() OOB-read hazard.  Numbers split across chunk boundaries
 * are accumulated in parse_ctx state. */
static int tar_zst_parse_current_entry(keystone_tar_zst_t* tz,
                                        int64_t** out_keys,
                                        size_t* out_count,
                                        int64_t* out_first_key,
                                        int64_t* out_last_key) {
    if (!tz || !tz->archive || !tz->current_entry) {
        if (tz) set_error(tz, "No current entry");
        return -1;
    }

    /* Reset arena for this member */
    arena_reset(tz->arena);

    /* Determine parse mode from options + filename heuristic */
    parse_mode_t mode;
    if (tz->options.format == KEYSTONE_TAR_ZST_FORMAT_AUTO) {
        const char* name = tz->member_name;
        size_t len = tz->member_name_len;
        if (len > 4 && strcasecmp(name + len - 4, ".csv") == 0) {
            mode = PARSE_CSV;
        } else if ((len > 5 && strcasecmp(name + len - 5, ".json") == 0) ||
                   (len > 3 && strcasecmp(name + len - 3, ".js") == 0)) {
            mode = PARSE_JSON;
        } else {
            mode = PARSE_TEXT;
        }
    } else {
        switch (tz->options.format) {
            case KEYSTONE_TAR_ZST_FORMAT_CSV:   mode = PARSE_CSV;  break;
            case KEYSTONE_TAR_ZST_FORMAT_JSON:  mode = PARSE_JSON; break;
            default:                              mode = PARSE_TEXT; break;
        }
    }

    memset(&tz->parse_ctx, 0, sizeof(tz->parse_ctx));
    tz->parse_ctx.mode = mode;
    tz->parse_ctx.skip_header = tz->options.skip_header;
    tz->parse_ctx.arena = tz->arena;

    size_t chunk_size = tz->options.chunk_size;
    if (chunk_size < 4096) chunk_size = 4096;

    uint64_t t0_decompress = ns_now();
    ssize_t bytes_total = 0;

    char* chunk = malloc(chunk_size);
    if (!chunk) {
        set_error(tz, "Out of memory allocating chunk buffer");
        return -1;
    }

    for (;;) {
        ssize_t n = archive_read_data(tz->archive, chunk, chunk_size);
        if (n < 0) {
            set_error(tz, "archive_read_data failed: %s",
                      archive_error_string(tz->archive));
            free(chunk);
            return -1;
        }
        if (n == 0) break;
        bytes_total += n;
        if ((uint64_t)bytes_total > KEYSTONE_TAR_ZST_MAX_DECOMPRESS_BYTES) {
            set_error(tz, "Member exceeds max decompression size (%llu bytes)",
                      (unsigned long long)KEYSTONE_TAR_ZST_MAX_DECOMPRESS_BYTES);
            free(chunk);
            return -1;
        }

        uint64_t t0_parse = ns_now();
        parse_feed(&tz->parse_ctx, chunk, (size_t)n);
        tz->stats.parse_time_ns += ns_now() - t0_parse;
    }

    free(chunk);
    tz->stats.decompress_time_ns += ns_now() - t0_decompress;
    tz->stats.bytes_read += (uint64_t)bytes_total;
    tz->stats.members_read++;

    /* Flush any number pending at end-of-member */
    parse_finish(&tz->parse_ctx);

    /* Sort keys (KEYSTONE requires sorted input) */
    if (tz->parse_ctx.count > 1 && tz->parse_ctx.keys) {
        qsort(tz->parse_ctx.keys, tz->parse_ctx.count,
              sizeof(int64_t), int64_compare);
        /* Recompute first/last from the sorted array — the pre-sort
         * first_key/last_key reflect insertion order, not min/max. */
        tz->parse_ctx.first_key = tz->parse_ctx.keys[0];
        tz->parse_ctx.last_key  = tz->parse_ctx.keys[tz->parse_ctx.count - 1];
    }

    if (out_keys)   *out_keys = tz->parse_ctx.keys;
    if (out_count)  *out_count = tz->parse_ctx.count;
    if (out_first_key) *out_first_key = tz->parse_ctx.first_key;
    if (out_last_key)  *out_last_key = tz->parse_ctx.last_key;

    return 0;
}

keystone_result_t keystone_tar_zst_search_member(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config) {
    if (!tz || !member_name) return KEYSTONE_NOT_FOUND;

    /* If indexed, jump directly */
    if (tz->index_built && tz->index) {
        return keystone_tar_zst_search_indexed(tz, member_name, key, table, config);
    }

    /* Otherwise linear scan from current position */
    if (!tz->current_entry ||
        tz->member_name_len != strlen(member_name) ||
        memcmp(tz->member_name, member_name, tz->member_name_len) != 0) {
        /* Need to find the member */
        int found = 0;
        for (;;) {
            int r = keystone_tar_zst_next_member(tz, NULL, NULL);
            if (r <= 0) break;
            if (tz->member_name_len == strlen(member_name) &&
                memcmp(tz->member_name, member_name, tz->member_name_len) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            set_error(tz, "Member not found: %s", member_name);
            return KEYSTONE_NOT_FOUND;
        }
    }

    int64_t* keys = NULL;
    size_t count = 0;
    if (tar_zst_parse_current_entry(tz, &keys, &count, NULL, NULL) != 0) {
        return KEYSTONE_NOT_FOUND;
    }
    if (count == 0 || !keys) {
        return KEYSTONE_NOT_FOUND;
    }

    keystone_config_t default_config;
    if (!config) {
        keystone_config_init(&default_config, KEYSTONE_WORKLOAD_IDS);
        config = &default_config;
    }

    keystone_result_t result = keystone_search_enhanced(keys, count, key,
                                                             table, config);
    return result;
}

size_t keystone_tar_zst_search_member_batch(
    keystone_tar_zst_t* tz,
    const char* member_name,
    keystone_batch_item_t* items,
    size_t num_items,
    keystone_anchor_table_t* table,
    size_t tol,
    const keystone_parallel_config_t* config) {
    if (!tz || !member_name || !items || num_items == 0) return 0;

    int64_t* keys = NULL;
    size_t count = 0;

    /* If we already have a current entry matching member_name, parse it */
    if (!tz->current_entry ||
        tz->member_name_len != strlen(member_name) ||
        memcmp(tz->member_name, member_name, tz->member_name_len) != 0) {
        int found = 0;
        for (;;) {
            int r = keystone_tar_zst_next_member(tz, NULL, NULL);
            if (r <= 0) break;
            if (tz->member_name_len == strlen(member_name) &&
                memcmp(tz->member_name, member_name, tz->member_name_len) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            set_error(tz, "Member not found: %s", member_name);
            return 0;
        }
    }

    if (tar_zst_parse_current_entry(tz, &keys, &count, NULL, NULL) != 0) {
        return 0;
    }
    if (count == 0 || !keys) {
        return 0;
    }

    return keystone_search_batch_auto(keys, count, items, num_items,
                                         table, tol, config);
}

/* ============================================================================
 * Extract Member Keys
 * ============================================================================ */

int keystone_tar_zst_extract_member(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t** out_keys,
    size_t* out_count) {
    if (!tz || !member_name) return -1;

    /* Navigate to member */
    if (!tz->current_entry ||
        tz->member_name_len != strlen(member_name) ||
        memcmp(tz->member_name, member_name, tz->member_name_len) != 0) {
        int found = 0;
        for (;;) {
            int r = keystone_tar_zst_next_member(tz, NULL, NULL);
            if (r <= 0) break;
            if (tz->member_name_len == strlen(member_name) &&
                memcmp(tz->member_name, member_name, tz->member_name_len) == 0) {
                found = 1;
                break;
            }
        }
        if (!found) {
            set_error(tz, "Member not found: %s", member_name);
            return -1;
        }
    }

    int64_t* keys = NULL;
    size_t count = 0;
    if (tar_zst_parse_current_entry(tz, &keys, &count, NULL, NULL) != 0) {
        return -1;
    }

    if (out_keys) *out_keys = keys;
    if (out_count) *out_count = count;
    return 0;
}

/* ============================================================================
 * Index API
 * ============================================================================ */

int keystone_tar_zst_build_index(keystone_tar_zst_t* tz) {
    if (!tz || !tz->archive) return -1;

    if (tz->index) {
        tar_zst_index_destroy(tz->index);
    }
    tz->index = tar_zst_index_create();
    if (!tz->index) return -1;

    /* Scan all members, recording offsets */
    for (;;) {
        int r = keystone_tar_zst_next_member(tz, NULL, NULL);
        if (r <= 0) break;

        int64_t* keys = NULL;
        size_t count = 0;
        int64_t first_key = 0, last_key = 0;
        if (tar_zst_parse_current_entry(tz, &keys, &count,
                                         &first_key, &last_key) != 0) {
            continue;
        }

        /* Build Bloom filter from parsed keys */
        tar_zst_bloom_t *bloom = tar_zst_bloom_create(count);
        if (bloom && keys) {
            for (size_t i = 0; i < count; i++) {
                tar_zst_bloom_add(bloom, keys[i]);
            }
        }

        tar_zst_index_add(tz->index, tz->member_name, tz->member_name_len,
                           0, 0, count, first_key, last_key, bloom,
                           (keys && count > 0) ? keys : NULL);
    }

    tz->index_built = 1;

    /* Store archive path for reopen in search_indexed */
    /* Already stored in tz->archive_path from keystone_tar_zst_open */

    return 0;
}

keystone_result_t keystone_tar_zst_search_indexed(
    keystone_tar_zst_t* tz,
    const char* member_name,
    int64_t key,
    keystone_anchor_table_t* table,
    const keystone_config_t* config) {
    if (!tz || !member_name || !tz->index) {
        if (tz) set_error(tz, "Index not built");
        return KEYSTONE_NOT_FOUND;
    }

    tar_zst_index_entry_t* entry = tar_zst_index_find(tz->index, member_name,
                                                       strlen(member_name));
    if (!entry) {
        set_error(tz, "Member not in index: %s", member_name);
        return KEYSTONE_NOT_FOUND;
    }

    /* Quick-reject via bounds check */
    if (key < entry->first_key || key > entry->last_key) {
        return KEYSTONE_NOT_FOUND;
    }

    /* Quick-reject via Bloom filter (avoids decompression on negative) */
    if (entry->bloom && !tar_zst_bloom_may_contain(entry->bloom, key)) {
        return KEYSTONE_NOT_FOUND;
    }

    /* Fast path: search the retained sorted keys directly — no
     * decompression, no reopen, no re-parse.  This turns the index
     * from a negative-only accelerator into a real positive index. */
    if (entry->keys && entry->key_count > 0) {
        keystone_config_t default_config;
        if (!config) {
            keystone_config_init(&default_config, KEYSTONE_WORKLOAD_IDS);
            config = &default_config;
        }
        return keystone_search_enhanced(entry->keys, entry->key_count,
                                          key, table, config);
    }

    /* Fallback: keys were not retained — reopen archive and verify by
     * streaming.  This path is only hit if key retention failed at
     * index-build time (e.g. memory pressure). */
    if (!tz->archive_path) {
        set_error(tz, "Archive path not available for reopen");
        return KEYSTONE_NOT_FOUND;
    }

    keystone_tar_zst_t* tz2 = keystone_tar_zst_open(tz->archive_path, &tz->options);
    if (!tz2) {
        set_error(tz, "Failed to reopen archive for indexed search");
        return KEYSTONE_NOT_FOUND;
    }

    keystone_result_t result = keystone_tar_zst_search_member(
        tz2, member_name, key, table, config);

    keystone_tar_zst_close(tz2);
    return result;
}

/* ============================================================================
 * Error & Stats Helpers
 * ============================================================================ */

const char* keystone_tar_zst_error_string(keystone_tar_zst_t* tz) {
    if (!tz) return "Invalid handle";
    return tz->error_buf[0] ? tz->error_buf : "No error";
}

int keystone_tar_zst_get_stats(keystone_tar_zst_t* tz,
                                 keystone_tar_zst_stats_t* stats) {
    if (!tz || !stats) return -1;
    *stats = tz->stats;
    return 0;
}
