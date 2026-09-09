#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qihse_keystone_bridge.h"
#include "qihse_auth.h"
#include "qihse_kv_store.h"

#define BRIDGE_GUEST_ID 62001u
#define BRIDGE_OPERATOR_PASSWORD "KeystoneBridgeOperatorPass1!"
#define BRIDGE_GUEST_PASSWORD "KeystoneBridgeGuestPass1!"

static void assert_missing(qihse_kv_store_t* store,
                           const char* key,
                           qihse_user_t* reader) {
    char* value = qihse_kv_get_user(store, key, reader);
    assert(value == NULL);
}

int main(void) {
    /* QIHSE principals are authoritative registry objects, not caller-created
     * structs. Initialize the registry first and make the system operator
     * usable before creating the low-clearance negative-test principal. */
    assert(qihse_auth_init());
    qihse_user_t* operator_user = qihse_auth_get_user(0);
    assert(operator_user != NULL);
    if (qihse_auth_is_operator_password_default()) {
        assert(qihse_auth_bootstrap_operator(BRIDGE_OPERATOR_PASSWORD));
    }

    qihse_kv_store_t* store = qihse_kv_store_create();
    assert(store != NULL);

    keystone_qihse_bridge_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.kv_target = (qihse_kv_bridge_handle_t*)store;
    cfg.default_clearance = 3;
    cfg.default_compartment = 1;
    cfg.routing_slots = KEYSTONE_QIHSE_ROUTING_SLOTS;

    /* Initialization is allowed without ambient authority, but every dispatch
     * must fail closed until an authenticated principal is explicitly bound. */
    assert(keystone_qihse_bridge_init(&cfg) == 0);
    assert(keystone_qihse_bridge_dispatch_credential_authenticated(
               "missing-principal@example.test", "DeniedSecret", 2) == -1);
    assert_missing(store, "missing-principal@example.test", operator_user);

    /* The privileged principal may write the configured classified label. The
     * bridge ABI is zero-on-success even though QIHSE's KV API is boolean. */
    keystone_qihse_bridge_set_principal(operator_user);
    assert(keystone_qihse_bridge_dispatch_credential_authenticated(
               "operator@example.test", "OperatorSecret", 5) == 0);

    char* stored = qihse_kv_get_user(store, "operator@example.test", operator_user);
    assert(stored != NULL);
    assert(strstr(stored, "class=5") != NULL);
    assert(strstr(stored, "pass=OperatorSecret") != NULL);
    free(stored);

    /* The legacy symbol must preserve the same authenticated, fail-closed
     * semantics rather than reintroducing a context-free write path. */
    assert(keystone_qihse_bridge_dispatch_credential(
               "legacy-authenticated@example.test", "LegacySecret", 7) == 0);
    stored = qihse_kv_get_user(store,
                               "legacy-authenticated@example.test",
                               operator_user);
    assert(stored != NULL);
    assert(strstr(stored, "class=7") != NULL);
    free(stored);

    qihse_user_t* guest = qihse_auth_create_user(
        operator_user,
        BRIDGE_GUEST_ID,
        QIHSE_ROLE_GUEST,
        0,
        0,
        BRIDGE_GUEST_PASSWORD,
        false);
    assert(guest != NULL);

    /* A low-clearance principal cannot write the configured classified label,
     * and the bridge must translate QIHSE's boolean denial to -1. */
    keystone_qihse_bridge_set_principal(guest);
    assert(keystone_qihse_bridge_dispatch_credential_authenticated(
               "guest-denied@example.test", "ShouldNeverPersist", 3) == -1);
    assert_missing(store, "guest-denied@example.test", operator_user);

    /* Clearing the principal also closes the legacy entry point. */
    keystone_qihse_bridge_set_principal(NULL);
    assert(keystone_qihse_bridge_dispatch_credential(
               "cleared-principal@example.test", "NoAmbientAuthority", 1) == -1);
    assert_missing(store, "cleared-principal@example.test", operator_user);

    /* Cluster configuration must reject holes instead of silently falling back
     * to a different target. */
    qihse_kv_bridge_handle_t* cluster[2] = {
        (qihse_kv_bridge_handle_t*)store,
        NULL
    };
    keystone_qihse_bridge_config_t invalid_cluster = cfg;
    invalid_cluster.kv_target = NULL;
    invalid_cluster.cluster_targets = cluster;
    invalid_cluster.num_cluster_nodes = 2;
    invalid_cluster.ingestion_principal = operator_user;
    assert(keystone_qihse_bridge_init(&invalid_cluster) == -1);

    assert(qihse_auth_destroy_user(operator_user, BRIDGE_GUEST_ID));
    qihse_kv_store_destroy(store);
    puts("KEYSTONE/QIHSE bridge authorization and ABI regression passed");
    return 0;
}
