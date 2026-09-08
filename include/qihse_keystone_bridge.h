#ifndef QIHSE_KEYSTONE_BRIDGE_H
#define QIHSE_KEYSTONE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

typedef void qihse_kv_bridge_handle_t;

/** Number of CRC16 hash slots used for distributed QIHSE ingestion. */
#define KEYSTONE_QIHSE_ROUTING_SLOTS 16384u

/**
 * Configuration for the KEYSTONE -> QIHSE ingestion bridge.
 *
 * The bridge treats QIHSE as the authoritative authorization/storage layer.
 * Classified writes require an authenticated ingestion principal and are sent
 * through qihse_kv_set_user().
 */
typedef struct {
    qihse_kv_bridge_handle_t* kv_target;
    uint16_t default_clearance;
    uint16_t default_compartment;

    qihse_kv_bridge_handle_t** cluster_targets;
    uint32_t num_cluster_nodes;

    /** 0 selects the fixed 16,384-slot topology; any other accepted value must
     * equal KEYSTONE_QIHSE_ROUTING_SLOTS. */
    uint32_t routing_slots;

    /** Opaque authenticated qihse_user_t*. NULL means dispatch is denied. */
    void* ingestion_principal;
} keystone_qihse_bridge_config_t;

/**
 * Initialize the bridge. Either kv_target or a complete non-empty cluster
 * target array must be supplied. Cluster configurations containing NULL node
 * targets are rejected rather than falling back to another node.
 */
int keystone_qihse_bridge_init(const keystone_qihse_bridge_config_t* config);

/**
 * Backward-compatible dispatch symbol.
 *
 * This function no longer performs a context-free write. It is an alias for
 * keystone_qihse_bridge_dispatch_credential_authenticated() and therefore
 * fails if no authenticated ingestion principal is configured.
 */
int keystone_qihse_bridge_dispatch_credential(
    const char* email,
    const char* pass,
    int semantic_class);

/**
 * Set or clear the authenticated ingestion principal.
 * Pass NULL to make subsequent dispatches fail closed.
 */
void keystone_qihse_bridge_set_principal(void* principal);

/**
 * Dispatch a discovered credential to QIHSE with the configured authenticated
 * principal. The write is routed through qihse_kv_set_user().
 *
 * Returns 0/true-compatible success from QIHSE, or -1 for bridge validation,
 * routing, missing-principal, or formatting failures.
 */
int keystone_qihse_bridge_dispatch_credential_authenticated(
    const char* email,
    const char* pass,
    int semantic_class);

/** Compute CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF). */
uint16_t keystone_qihse_crc16(const void* data, size_t len);

/** Route a key to one of KEYSTONE_QIHSE_ROUTING_SLOTS fixed slots. */
uint32_t keystone_qihse_bridge_route_slot(const char* key, size_t key_len);

/** Map a fixed routing slot to a cluster node index. */
uint32_t keystone_qihse_bridge_slot_to_node(uint32_t slot, uint32_t num_nodes);

#ifdef __cplusplus
}
#endif
#endif
