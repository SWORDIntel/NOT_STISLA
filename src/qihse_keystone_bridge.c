#include "../include/qihse_keystone_bridge.h"
#include <stdio.h>
#include <string.h>

/* =====================================================================
 * 16,384-way CRC16 hash slot routing for distributed QIHSE cluster
 * ingestion.
 *
 * These routing helpers are pure functions with no dependency on libqihse,
 * so they are compiled in both standalone and bridge-enabled builds. This
 * lets the slot distribution be unit-tested without linking the QIHSE
 * engine, and lets the dirty-parser hot path resolve the destination node
 * for a credential without any conditional compilation.
 * ===================================================================== */

uint16_t keystone_qihse_crc16(const void* data, size_t len) {
    /* CRC16-CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection. */
    const unsigned char* p = (const unsigned char*)data;
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000u) crc = (uint16_t)((crc << 1) ^ 0x1021u);
            else               crc = (uint16_t)(crc << 1);
        }
    }
    return crc;
}

uint32_t keystone_qihse_bridge_route_slot(const char* key, size_t key_len) {
    if (!key) return 0;
    /* KEYSTONE_QIHSE_ROUTING_SLOTS is a power of two (2^14), so a bitmask
     * collapses the 16-bit CRC into [0, 16384) without an expensive modulo. */
    uint16_t crc = keystone_qihse_crc16(key, key_len);
    return (uint32_t)crc & (KEYSTONE_QIHSE_ROUTING_SLOTS - 1u);
}

uint32_t keystone_qihse_bridge_slot_to_node(uint32_t slot, uint32_t num_nodes) {
    if (num_nodes == 0) return 0;
    /* Even, deterministic slot->node mapping via fixed-point multiplication.
     * (slot * num_nodes) / 16384 spreads the 16,384 slots uniformly across
     * the cluster with no hot-spotting. */
    return (uint32_t)(((uint64_t)slot * (uint64_t)num_nodes) / KEYSTONE_QIHSE_ROUTING_SLOTS);
}

#ifdef KEYSTONE_ENABLE_QIHSE_BRIDGE
/*
 * If the bridge is enabled, we link against libqihse.so and use its headers.
 * We include the bare minimum needed for UWP dispatch.
 */
#include <qihse.h>

static keystone_qihse_bridge_config_t g_bridge_cfg = {0};
static int g_bridge_active = 0;

int keystone_qihse_bridge_init(const keystone_qihse_bridge_config_t* config) {
    if (!config) return -1;

    /* Accept either the legacy single-node target or a multi-node cluster. */
    int has_single  = (config->kv_target != NULL);
    int has_cluster = (config->num_cluster_nodes > 0 && config->cluster_targets != NULL);
    if (!has_single && !has_cluster) return -1;

    g_bridge_cfg = *config;
    if (g_bridge_cfg.routing_slots == 0) {
        g_bridge_cfg.routing_slots = KEYSTONE_QIHSE_ROUTING_SLOTS;
    }
    g_bridge_active = 1;
    return 0;
}

void keystone_qihse_bridge_set_principal(void* principal) {
    g_bridge_cfg.ingestion_principal = principal;
}

int keystone_qihse_bridge_dispatch_credential(
    const char* email,
    const char* pass,
    int semantic_class)
{
    if (!g_bridge_active) return -1;
    if (!email || !pass) return -1;

    qihse_kv_store_t* kv = (qihse_kv_store_t*)g_bridge_cfg.kv_target;
    uint16_t clearance   = g_bridge_cfg.default_clearance;
    uint16_t compartment = g_bridge_cfg.default_compartment;

    if (g_bridge_cfg.num_cluster_nodes > 0 && g_bridge_cfg.cluster_targets) {
        uint32_t slot = keystone_qihse_bridge_route_slot(email, strlen(email));
        uint32_t node = keystone_qihse_bridge_slot_to_node(slot, g_bridge_cfg.num_cluster_nodes);
        qihse_kv_store_t* node_kv = (qihse_kv_store_t*)g_bridge_cfg.cluster_targets[node];
        if (node_kv) {
            kv = node_kv;
        }
    } else if (!kv) {
        return -1;
    }

    char enriched_value[512];
    snprintf(enriched_value, sizeof(enriched_value), "class=%d|pass=%s", semantic_class, pass);

    /* Legacy context-free write path.  Retained for backward compatibility
     * but deprecated — new callers should use the authenticated variant. */
    int rc = qihse_kv_set(
        kv,
        email,
        enriched_value,
        clearance,
        compartment
    );

    return rc;
}

int keystone_qihse_bridge_dispatch_credential_authenticated(
    const char* email,
    const char* pass,
    int semantic_class)
{
    if (!g_bridge_active) return -1;
    if (!email || !pass) return -1;

    /* Per QIHSE's security model (AGENTS.md invariant #1), no classified
     * write primitive may be invoked without an explicit authenticated
     * security context.  Refuse the write if no principal is set. */
    qihse_user_t* principal = (qihse_user_t*)g_bridge_cfg.ingestion_principal;
    if (!principal) {
        return -1;
    }

    qihse_kv_store_t* kv = (qihse_kv_store_t*)g_bridge_cfg.kv_target;
    uint16_t clearance   = g_bridge_cfg.default_clearance;
    uint16_t compartment = g_bridge_cfg.default_compartment;

    if (g_bridge_cfg.num_cluster_nodes > 0 && g_bridge_cfg.cluster_targets) {
        uint32_t slot = keystone_qihse_bridge_route_slot(email, strlen(email));
        uint32_t node = keystone_qihse_bridge_slot_to_node(slot, g_bridge_cfg.num_cluster_nodes);
        qihse_kv_store_t* node_kv = (qihse_kv_store_t*)g_bridge_cfg.cluster_targets[node];
        if (node_kv) {
            kv = node_kv;
        }
    } else if (!kv) {
        return -1;
    }

    char enriched_value[512];
    snprintf(enriched_value, sizeof(enriched_value), "class=%d|pass=%s", semantic_class, pass);

    /* Authenticated write: propagates the ingestion principal to QIHSE's
     * authorization layer so the write inherits clearance + SCI compartment
     * enforcement rather than being a context-free write. */
    int rc = qihse_kv_set_user(
        kv,
        email,
        enriched_value,
        clearance,
        compartment,
        principal
    );

    return rc;
}

#else

/*
 * Stub implementation for standalone KEYSTONE builds.
 * The bridge does nothing and returns an error if not explicitly compiled in.
 * The CRC16 routing helpers above remain available so that slot distribution
 * can be validated without linking libqihse.
 */

int keystone_qihse_bridge_init(const keystone_qihse_bridge_config_t* config) {
    (void)config;
    return -1;
}

int keystone_qihse_bridge_dispatch_credential(
    const char* email,
    const char* pass,
    int semantic_class)
{
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
    int semantic_class)
{
    (void)email;
    (void)pass;
    (void)semantic_class;
    return -1;
}

#endif
