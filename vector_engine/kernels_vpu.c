/*
 * kernels_vpu.c - KEYSTONE Vector Engine VPU (Intel Myriad X) kernel backend
 *
 * The VPU accelerates BATCH similarity computation by sending a query
 * vector plus a candidate matrix to the VPU over a Unix domain socket
 * and receiving the resulting distances back. Single-query (dist1) and
 * normalization are NOT offloaded: the socket round-trip overhead
 * dwarfs the compute for those operations, so they fall back to the
 * internal scalar implementations compiled into this file.
 *
 * Compile-time guard: the entire implementation is compiled only when
 * KEYSTONE_HAVE_VPU is defined. Otherwise this file is an empty comment
 * so it links cleanly on hosts without VPU support. Even when compiled
 * in, the VPU path is only used at runtime if the socket is reachable
 * (see keystone_vpu_kernel_connect / keystone_vpu_kernel_is_connected).
 *
 * Wire protocol (little-endian, all fields host-order on a Unix socket):
 *
 *   Request:
 *     uint32_t magic      = 0x4B564556  ("KVEV")
 *     uint32_t msg_type   = 1           (BATCH_DIST)
 *     uint32_t dim
 *     uint32_t n_cands
 *     float    query[dim]               (dim * 4 bytes)
 *     float    cands[n_cands * dim]     (n_cands * dim * 4 bytes)
 *
 *   Response:
 *     uint32_t magic      = 0x4B564556
 *     uint32_t status     = 0 on success, non-zero on error
 *     uint32_t n_cands
 *     float    distances[n_cands]
 */

#ifdef KEYSTONE_HAVE_VPU

#include "kernel_dispatch.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

/* ---------------------------------------------------------------
 * Protocol constants
 * --------------------------------------------------------------- */
#define VPU_MAGIC       0x4B564556u   /* "KVEV" */
#define VPU_MSG_BATCH   1u            /* BATCH_DIST */
#define VPU_STATUS_OK   0u

/* ---------------------------------------------------------------
 * Thread-local metric. Default cosine. The engine calls the setter
 * once after creating the backend so all subsequent dist1 calls use
 * the right formula without an ABI change.
 * --------------------------------------------------------------- */
static __thread keystone_metric_t g_metric = KEYSTONE_METRIC_COSINE;

void keystone_vpu_set_metric(keystone_metric_t m)
{
    g_metric = m;
}

/* ---------------------------------------------------------------
 * VPU connection state. A single Unix domain socket is shared across
 * all callers in this process. The engine connects once at startup
 * and disconnects at shutdown.
 * --------------------------------------------------------------- */
static struct {
    int  fd;
    int  connected;
} g_vpu = { .fd = -1, .connected = 0 };

int keystone_vpu_kernel_is_connected(void)
{
    return g_vpu.connected && g_vpu.fd >= 0;
}

int keystone_vpu_kernel_connect(const char *socket_path)
{
    if (!socket_path || socket_path[0] == '\0')
        return -1;

    /* Already connected: close and reconnect cleanly. */
    if (g_vpu.connected && g_vpu.fd >= 0) {
        close(g_vpu.fd);
        g_vpu.fd = -1;
        g_vpu.connected = 0;
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;

    /* Bound by sizeof(sun_path) - 1 to guarantee NUL termination. */
    size_t plen = strlen(socket_path);
    if (plen >= sizeof(addr.sun_path)) {
        close(fd);
        return -1;
    }
    memcpy(addr.sun_path, socket_path, plen);
    addr.sun_path[plen] = '\0';

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    g_vpu.fd = fd;
    g_vpu.connected = 1;
    return 0;
}

void keystone_vpu_kernel_disconnect(void)
{
    if (g_vpu.fd >= 0) {
        close(g_vpu.fd);
        g_vpu.fd = -1;
    }
    g_vpu.connected = 0;
}

/* Check if a VPU socket is available (non-invasive probe).
 * Used by keystone_vec_detect_features() at runtime. */
int keystone_vec_vpu_available(const char *socket_path)
{
    const char *path = socket_path;
    if (!path || path[0] == '\0')
        path = "/tmp/myriad_embed.sock";

    /* Quick check: does the socket file exist? */
    struct sockaddr_un addr;
    if (strlen(path) >= sizeof(addr.sun_path))
        return 0;

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
        return 0;

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    close(fd);
    return (rc == 0) ? 1 : 0;
}

/* ---------------------------------------------------------------
 * write_all / read_all - robust I/O helpers that handle partial
 * writes/reads on the socket. Return 0 on success, -1 on error.
 * --------------------------------------------------------------- */
static int write_all(int fd, const void *buf, size_t len)
{
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (w == 0)
            return -1;
        off += (size_t)w;
    }
    return 0;
}

static int read_all(int fd, void *buf, size_t len)
{
    char *p = (char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t r = read(fd, p + off, len - off);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (r == 0)
            return -1;
        off += (size_t)r;
    }
    return 0;
}

/* ---------------------------------------------------------------
 * Internal scalar fallbacks.
 *
 * dist1 and normalize_batch always use these: the VPU socket round-trip
 * is far more expensive than a single 384-dim dot product or a few
 * divisions, so offloading them would be a net loss. dist_batch uses
 * these too when the VPU socket is not connected.
 * --------------------------------------------------------------- */
static float vpu_scalar_dot(const float *a, const float *b, uint32_t dim)
{
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++)
        acc += a[i] * b[i];
    return acc;
}

static float vpu_scalar_dist1(const float *a, const float *b, uint32_t dim)
{
    if (!a || !b || dim == 0)
        return 0.0f;

    switch (g_metric) {
    case KEYSTONE_METRIC_L2: {
        float acc = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float d = a[i] - b[i];
            acc += d * d;
        }
        return acc;
    }
    case KEYSTONE_METRIC_DOT:
        return -vpu_scalar_dot(a, b, dim);
    case KEYSTONE_METRIC_COSINE:
    default:
        return 1.0f - vpu_scalar_dot(a, b, dim);
    }
}

static void vpu_scalar_dist_batch(const float *q,
                                  const float *cands,
                                  uint32_t n_cands,
                                  uint32_t dim,
                                  float *out)
{
    for (uint32_t i = 0; i < n_cands; i++)
        out[i] = vpu_scalar_dist1(q, cands + (size_t)i * dim, dim);
}

static void vpu_scalar_normalize_batch(float *vectors, uint32_t n, uint32_t dim)
{
    if (!vectors || dim == 0 || n == 0)
        return;

    for (uint32_t v = 0; v < n; v++) {
        float *vec = vectors + (size_t)v * dim;
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++)
            sum += vec[i] * vec[i];
        float norm = sqrtf(sum);
        if (norm < 1e-12f)
            continue;
        float inv = 1.0f / norm;
        for (uint32_t i = 0; i < dim; i++)
            vec[i] *= inv;
    }
}

/* ---------------------------------------------------------------
 * vpu_dist_batch - offload batch distance to the VPU over the socket.
 *
 * Sends the query + candidate matrix, receives n_cands float distances.
 * On ANY error (socket not connected, write/read failure, bad magic,
 * status != OK, count mismatch) this falls back to the scalar batch
 * implementation so the caller always gets correct results.
 * --------------------------------------------------------------- */
static void vpu_dist_batch(const float *q,
                           const float *cands,
                           uint32_t n_cands,
                           uint32_t dim,
                           float *out)
{
    if (!q || !cands || !out || dim == 0 || n_cands == 0)
        return;

    /* If the VPU is not connected, use the scalar path. */
    if (!keystone_vpu_kernel_is_connected()) {
        vpu_scalar_dist_batch(q, cands, n_cands, dim, out);
        return;
    }

    int fd = g_vpu.fd;

    /* Build the fixed-size header. */
    uint32_t hdr[4];
    hdr[0] = VPU_MAGIC;
    hdr[1] = VPU_MSG_BATCH;
    hdr[2] = dim;
    hdr[3] = n_cands;

    if (write_all(fd, hdr, sizeof(hdr)) != 0)
        goto fallback;

    /* Send the query vector. */
    if (write_all(fd, q, (size_t)dim * sizeof(float)) != 0)
        goto fallback;

    /* Send the candidate matrix (n_cands * dim floats, row-major). */
    if (write_all(fd, cands, (size_t)n_cands * dim * sizeof(float)) != 0)
        goto fallback;

    /* Read the response header. */
    uint32_t resp[3];
    if (read_all(fd, resp, sizeof(resp)) != 0)
        goto fallback;

    if (resp[0] != VPU_MAGIC)
        goto fallback;
    if (resp[1] != VPU_STATUS_OK)
        goto fallback;
    if (resp[2] != n_cands)
        goto fallback;

    /* Read the resulting distances. */
    if (read_all(fd, out, (size_t)n_cands * sizeof(float)) != 0)
        goto fallback;

    return;

fallback:
    /* Socket path failed: mark disconnected and finish on the CPU so
     * the caller still gets correct results. The engine may reconnect
     * later; until then subsequent batches go straight to scalar. */
    keystone_vpu_kernel_disconnect();
    vpu_scalar_dist_batch(q, cands, n_cands, dim, out);
}

/* ---------------------------------------------------------------
 * Exported kernel table.
 *
 * dist1 and normalize_batch point at the internal scalar fallbacks
 * (single-query / normalization are not worth a VPU round-trip).
 * dist_batch points at the VPU offload path, which itself falls back
 * to scalar when the socket is not connected.
 * --------------------------------------------------------------- */
const keystone_kernels_t keystone_kernels_vpu = {
    .dist1           = vpu_scalar_dist1,
    .dist_batch      = vpu_dist_batch,
    .normalize_batch = vpu_scalar_normalize_batch,
};

#else  /* !KEYSTONE_HAVE_VPU */

/*
 * VPU support not compiled in. This translation unit is intentionally
 * empty so the file links cleanly on hosts without VPU support. The
 * scalar backend (always compiled) provides the fallback.
 */

#endif /* KEYSTONE_HAVE_VPU */
