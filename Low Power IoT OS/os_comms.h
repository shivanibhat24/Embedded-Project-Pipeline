/**
 * @file os_comms.h
 * @brief Communications framework for MicroOS
 */

#ifndef OS_COMMS_H
#define OS_COMMS_H

#include "os_core.h"

/**
 * @brief Communication interface handle
 */
typedef uint8_t CommsHandle;

/**
 * @brief Communication protocol types
 */
typedef enum {
    COMMS_PROTOCOL_MQTT,
    COMMS_PROTOCOL_COAP,
    COMMS_PROTOCOL_HTTP,
    COMMS_PROTOCOL_BLE,
    COMMS_PROTOCOL_LORA,
    COMMS_PROTOCOL_ZIGBEE,
    COMMS_PROTOCOL_CUSTOM
} CommsProtocol;

/**
 * @brief Communication security levels
 */
typedef enum {
    COMMS_SECURITY_NONE,           // No security
    COMMS_SECURITY_ENCRYPTION,     // Encryption only
    COMMS_SECURITY_AUTHENTICATION, // Authentication only
    COMMS_SECURITY_FULL            // Full security (encryption + authentication)
} CommsSecurity;

/**
 * @brief Communication power modes
 */
typedef enum {
    COMMS_POWER_OFF,               // Interface powered off
    COMMS_POWER_LOW,               // Low power (reduced range/rate)
    COMMS_POWER_NORMAL,            // Normal operation
    COMMS_POWER_HIGH               // High power (extended range/rate)
} CommsPowerMode;

/**
 * @brief Communication interface configuration
 */
typedef struct {
    CommsProtocol protocol;        // Protocol type
    CommsSecurity security;        // Security level
    CommsPowerMode power_mode;     // Power mode
    uint32_t tx_power_dbm;         // Transmit power in dBm
    uint32_t data_rate_bps;        // Data rate in bits per second
    uint16_t retry_count;          // Number of retries
    uint16_t retry_interval_ms;    // Retry interval
    bool auto_reconnect;           // Auto reconnect on disconnect
    bool low_power_listen;         // Low power listening mode
    char server_address[64];       // Server address
    uint16_t server_port;          // Server port
    void* protocol_config;         // Protocol-specific configuration
} CommsConfig;

/**
 * @brief Communication status
 */
typedef enum {
    COMMS_STATUS_DISCONNECTED,     // Not connected
    COMMS_STATUS_CONNECTING,       // Connection in progress
    COMMS_STATUS_CONNECTED,        // Connected
    COMMS_STATUS_DISCONNECTING,    // Disconnection in progress
    COMMS_STATUS_ERROR             // Error state
} CommsStatus;

/**
 * @brief Communication event types
 */
typedef enum {
    COMMS_EVENT_CONNECTED,         // Connected to server
    COMMS_EVENT_DISCONNECTED,      // Disconnected from server
    COMMS_EVENT_DATA_RECEIVED,     // Data received
    COMMS_EVENT_DATA_SENT,         // Data sent successfully
    COMMS_EVENT_ERROR              // Error occurred
} CommsEvent;

/**
 * @brief Communication event data
 */
typedef struct {
    CommsEvent event;              // Event type
    uint8_t* data;                 // Event data
    uint16_t data_length;          // Data length
    uint32_t error_code;           // Error code (for COMMS_EVENT_ERROR)
} CommsEventData;

/**
 * @brief Communication statistics
 */
typedef struct {
    uint32_t bytes_sent;           // Total bytes sent
    uint32_t bytes_received;       // Total bytes received
    uint32_t packets_sent;         // Total packets sent
    uint32_t packets_received;     // Total packets received
    uint32_t retry_count;          // Number of retries
    uint32_t error_count;          // Number of errors
    uint32_t connection_count;     // Number of connections
    uint32_t uptime_ms;            // Connection uptime
    float signal_strength;         // Signal strength (RSSI)
    float signal_quality;          // Signal quality (SNR)
    float power_consumption_mw;    // Power consumption
} CommsStats;

/**
 * @brief Communication callback function
 */
typedef void (*CommsCallback)(CommsHandle handle, const CommsEventData* event_data, void* param);

/**
 * @brief Initialize communications framework
 * 
 * @return OS_SUCCESS if initialization was successful
 */
OsError os_comms_init(void);

/**
 * @brief Create a communication interface
 * 
 * @param config Interface configuration
 * @param callback Callback function
 * @param callback_param Parameter to pass to callback
 * @param handle Pointer to store interface handle
 * @return OS_SUCCESS if interface was created successfully
 */
OsError os_comms_create(const CommsConfig* config, CommsCallback callback, 
                       void* callback_param, CommsHandle* handle);

/**
 * @brief Delete a communication interface
 * 
 * @param handle Interface handle
 * @return OS_SUCCESS if interface was deleted successfully
 */
OsError os_comms_delete(CommsHandle handle);

/**
 * @brief Connect a communication interface
 * 
 * @param handle Interface handle
 * @param timeout_ms Connection timeout
 * @return OS_SUCCESS if connection was initiated successfully
 */
OsError os_comms_connect(CommsHandle handle, uint32_t timeout_ms);

/**
 * @brief Disconnect a communication interface
 * 
 * @param handle Interface handle
 * @return OS_SUCCESS if disconnection was initiated successfully
 */
OsError os_comms_disconnect(CommsHandle handle);

/**
 * @brief Send data through a communication interface
 * 
 * @param handle Interface handle
 * @param data Data to send
 * @param length Data length
 * @param timeout_ms Send timeout
 * @return OS_SUCCESS if data was sent successfully
 */
OsError os_comms_send(CommsHandle handle, const uint8_t* data, uint16_t length, uint32_t timeout_ms);

/**
 * @brief Receive data through a communication interface
 * 
 * @param handle Interface handle
 * @param data Buffer to store received data
 * @param max_length Maximum data length
 * @param received_length Pointer to store actual received length
 * @param timeout_ms Receive timeout
 * @return OS_SUCCESS if data was received successfully
 */
OsError os_comms_receive(CommsHandle handle, uint8_t* data, uint16_t max_length, 
                        uint16_t* received_length, uint32_t timeout_ms);

/**
 * @brief Get communication interface status
 * 
 * @param handle Interface handle
 * @param status Pointer to store status
 * @return OS_SUCCESS if status was retrieved successfully
 */
OsError os_comms_get_status(CommsHandle handle, CommsStatus* status);

/**
 * @brief Get communication interface statistics
 * 
 * @param handle Interface handle
 * @param stats Pointer to store statistics
 * @return OS_SUCCESS if statistics were retrieved successfully
 */
OsError os_comms_get_stats(CommsHandle handle, CommsStats* stats);

/**
 * @brief Set communication interface power mode
 * 
 * @param handle Interface handle
 * @param power_mode Power mode
 * @return OS_SUCCESS if power mode was set successfully
 */
OsError os_comms_set_power_mode(CommsHandle handle, CommsPowerMode power_mode);

/**
 * @brief Configure communication interface wake-up
 * 
 * @param handle Interface handle
 * @param enable Enable wake-up
 * @param min_power_mode Minimum power mode to wake from
 * @return OS_SUCCESS if wake-up was configured successfully
 */
OsError os_comms_configure_wakeup(CommsHandle handle, bool enable, PowerMode min_power_mode);

#endif /* OS_COMMS_H */
