#include "../include/qihse_keystone_bridge.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

/* =====================================================================
 * 16,384-way CRC16 hash slot routing for distributed QIHSE cluster
 * ingestion.
 * ===================================================================== */

uint16_t keystone_qihse_crc16(const void* data, size_t len) {
    if (!data && len != 0u) return 0u;
    const unsigned char* p = (const unsigned char*)data;
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0u; i < len; i++) {
        crc ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000u) crc = (uint16_t)((crc << 1) ^ 0x1021u);
            else               crc = (uint16_t)(crc << 1);
        }
    }
    return crc;
}

uint32_t keystone_qihse_bridge_route_slot(const char* key, size_t key_len) {
    if (!key) return 0u;
    uint16_t crc = keystone_qihse_crc16(key, key_len);
    return (uint32_t)crc & (KEYSTONE_QIHSE_ROUTING_SLOTS - 1u);
}

uint32_t keystone_qihse_bridge_slot_to_node(uint32_t slot, uint32_t num_nodes) {
    if (num_nodes == 0u) return 0u;
    slot &= (KEYSTONE_QIHSE_ROUTING_SLOTS - 1u);
    return (uint32_t)(((uint64_t)slot * (uint64_t)num_nodes) /
                      KEYSTONE_QIHSE_ROUTING_SLOTS);
}

#ifdef KEYSTONE_ENABLE_QIHSE_BRIDGE
#include <qihse_kv_store.h>

static keystone_qihse_bridge_config_t g_bridge_cfg = {0};
static int g_bridge_active = 0;
static atomic_flag g_bridge_cfg_lock = ATOMIC_FLAG_INIT;

static void bridge_lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_bridge_cfg_lock, memory_order_acquire)) {
        /* Configuration critical sections are intentionally tiny. */
    }
}

static void bridge_unlock(void) {
    atomic_flag_clear_explicit(&g_bridge_cfg_lock, memory_order_release);
}

static int snapshot_bridge_config(keystone_qihse_bridge_config_t* out) {
    if (!out) return -1;
    bridge_lock();
    int active = g_bridge_active;
    if (active) *out = g_bridge_cfg;
    bridge_unlock();
    return active ? 0 : -1;
}

static void secure_zero(void* ptr, size_t len) {
    volatile unsigned char* p = (volatile unsigned char*)ptr;
    while (ptr && len-- > 0u) *p++ = 0u;
}

int keystone_qihse_bridge_init(const keystone_qihse_bridge_config_t* config) {
    if (!config) return -1;

    int has_single = (config->kv_target != NULL);
    int has_cluster = (config->num_cluster_nodes > 0u && config->cluster_targets != NULL);
    if (!has_single && !has_cluster) return -1;

    /* The public routing helper is intentionally fixed at 16,384 slots. Do
     * not accept a configuration value that advertises a different topology
     * while silently routing with the fixed function. */
    if (config->routing_slots != 0u &&
        config->routing_slots != KEYSTONE_QIHSE_ROUTING_SLOTS) {
        return -1;
    }

    if (has_cluster) {
        for (uint32_t i = 0u; i < config->num_cluster_nodes; i++) {
            if (!config->cluster_targets[i]) return -1;
        }
    }

    keystone_qihse_bridge_config_t next = *config;
    next.routing_slots = KEYSTONE_QIHSE_ROUTING_SLOTS;

    bridge_lock();
    g_bridge_cfg = next;
    g_bridge_active = 1;
    bridge_unlock();
    return 0;
}

void keystone_qihse_bridge_set_principal(void* principal) {
    bridge_lock();
    g_bridge_cfg.ingestion_principal = principal;
    bridge_unlock();
}

static qihse_kv_store_t* select_target(
    const keystone_qihse_bridge_config_t* cfg,
    const char* email
) {
    if (!cfg || !email) return NULL;

    if (cfg->num_cluster_nodes > 0u) {
        if (!cfg->cluster_targets) return NULL;
        uint32_t slot = keystone_qihse_bridge_route_slot(email, strlen(email));
        uint32_t node = keystone_qihse_bridge_slot_to_node(slot, cfg->num_cluster_nodes);
        if (node >= cfg->num_cluster_nodes) return NULL;
        return (qihse_kv_store_t*)cfg->cluster_targets[node];
    }

    return (qihse_kv_store_t*)cfg->kv_target;
}

int keystone_qihse_bridge_dispatch_credential(
    const char* email,
    const char* pass,
    int semantic_class
) {
    /* ABI-compatible legacy entry point, now fail-closed. Context-free writes
     * are no longer performed: all dispatches require the configured
     * authenticated ingestion principal. */
    return keystone_qihse_bridge_dispatch_credential_authenticated(
        email, pass, semantic_class);
}

int keystone_qihse_bridge_dispatch_credential_authenticated(
    const char* email,
    const char* pass,
    int semantic_class
) {
    if (!email || !pass || email[0] == '\0') return -1;

    keystone_qihse_bridge_config_t cfg;
    if (snapshot_bridge_config(&cfg) != 0) return -1;

    qihse_user_t* principal = (qihse_user_t*)cfg.ingestion_principal;
    if (!principal) return -1;

    qihse_kv_store_t* kv = select_target(&cfg, email);
    if (!kv) return -1;

    char enriched_value[512];
    int written = snprintf(
        enriched_value, sizeof(enriched_value), "class=%d|pass=%s", semantic_class, pass);
    if (written < 0 || (size_t)written >= sizeof(enriched_value)) {
        secure_zero(enriched_value, sizeof(enriched_value));
        return -1;
    }

    bool stored = qihse_kv_set_user(
        kv,
        email,
        enriched_value,
        cfg.default_clearance,
        cfg.default_compartment,
        principal);

    secure_zero(enriched_value, sizeof(enriched_value));

    /* Preserve the public bridge ABI: zero is success, negative is failure.
     * QIHSE's KV API is boolean, so never leak its 1/0 convention through the
     * bridge boundary. */
    return stored ? 0 : -1;
}

#else

int keystone_qihse_bridge_init(const keystone_qihse_bridge_config_t* config) {
    (void)config;
    return -1;
}

int keystone_qihse_bridge_dispatch_credential(
    const char* email,
    const char* pass,
    int semantic_class
) {
    (void)email;
    (void)pass;
    (void)semantic_class;
    return -1;
}

void keystone_qihse_bridge_set_principal(void* principal) {
    (void)principal;
}

int keystone_qihse_bridge_dispatch_credential_authenticated(
    const char* email,
    const char* pass,
    int semantic_class
) {
    (void)email;
    (void)pass;
    (void)semantic_class;
    return -1;
}

#endif
