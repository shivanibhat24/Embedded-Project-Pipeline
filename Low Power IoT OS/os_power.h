/**
 * @file os_power.h
 * @brief Power management for MicroOS
 */

#ifndef OS_POWER_H
#define OS_POWER_H

#include "os_core.h"

/**
 * @brief Power consumption profile
 */
typedef struct {
    float active_power_mw;       // Active mode power in mW
    float idle_power_mw;         // Idle mode power in mW
    float sleep_power_mw;        // Sleep mode power in mW
    float deep_sleep_power_mw;   // Deep sleep mode power in mW
    float hibernation_power_mw;  // Hibernation mode power in mW
} PowerProfile;

/**
 * @brief Wake-up source enumeration
 */
typedef enum {
    WAKEUP_SOURCE_RTC,       // Real-time clock
    WAKEUP_SOURCE_GPIO,      // GPIO pin
    WAKEUP_SOURCE_UART,      // UART activity
    WAKEUP_SOURCE_I2C,       // I2C activity
    WAKEUP_SOURCE_SPI,       // SPI activity
    WAKEUP_SOURCE_SENSOR,    // Sensor threshold
    WAKEUP_SOURCE_TIMER,     // System timer
    WAKEUP_SOURCE_EXTERNAL   // External event
} WakeupSource;

/**
 * @brief Wake-up configuration
 */
typedef struct {
    WakeupSource source;     // Wake-up source
    uint32_t timeout_ms;     // Timeout in milliseconds (for RTC/timer)
    uint8_t gpio_pin;        // GPIO pin number (for GPIO)
    bool gpio_rising_edge;   // True for rising edge, false for falling (for GPIO)
    uint8_t uart_channel;    // UART channel (for UART)
    uint8_t sensor_id;       // Sensor ID (for sensor)
    float sensor_threshold;  // Sensor threshold value (for sensor)
} WakeupConfig;

/**
 * @brief Power state transition callback
 */
typedef void (*PowerStateCallback)(PowerMode old_mode, PowerMode new_mode);

/**
 * @brief Initialize power management
 * 
 * @return OS_SUCCESS if initialization was successful
 */
OsError os_power_init(void);

/**
 * @brief Configure a wake-up source
 * 
 * @param config Wake-up configuration
 * @return OS_SUCCESS if configuration was successful
 */
OsError os_power_configure_wakeup(const WakeupConfig* config);

/**
 * @brief Enable a wake-up source
 * 
 * @param source Wake-up source to enable
 * @return OS_SUCCESS if source was enabled successfully
 */
OsError os_power_enable_wakeup_source(WakeupSource source);

/**
 * @brief Disable a wake-up source
 * 
 * @param source Wake-up source to disable
 * @return OS_SUCCESS if source was disabled successfully
 */
OsError os_power_disable_wakeup_source(WakeupSource source);

/**
 * @brief Set dynamic RTC wake-up timing
 * 
 * @param ms Wake-up time in milliseconds
 * @return OS_SUCCESS if timing was set successfully
 */
OsError os_power_set_rtc_wakeup(uint32_t ms);

/**
 * @brief Register a callback for power state transitions
 * 
 * @param callback Callback function
 * @return OS_SUCCESS if callback was registered successfully
 */
OsError os_power_register_state_callback(PowerStateCallback callback);

/**
 * @brief Get current power consumption estimate
 * 
 * @param power_mw Pointer to store power in milliwatts
 * @return OS_SUCCESS if power was estimated successfully
 */
OsError os_power_get_current_consumption(float* power_mw);

/**
 * @brief Get power profile for the system
 * 
 * @param profile Pointer to store power profile
 * @return OS_SUCCESS if profile was retrieved successfully
 */
OsError os_power_get_profile(PowerProfile* profile);

/**
 * @brief Set power budget for the system
 * 
 * When the system approaches the power budget, it will automatically
 * transition to lower power modes.
 * 
 * @param budget_mw Power budget in milliwatts
 * @return OS_SUCCESS if budget was set successfully
 */
OsError os_power_set_budget(float budget_mw);

/**
 * @brief Configure adaptive power management
 * 
 * @param enable Enable/disable adaptive power management
 * @param sensitivity Sensitivity to workload changes (0-100)
 * @return OS_SUCCESS if configuration was successful
 */
OsError os_power_configure_adaptive_management(bool enable, uint8_t sensitivity);

/**
 * @brief Force immediate power mode transition
 * 
 * @param mode Power mode to enter
 * @param timeout_ms Time to stay in this mode (0 for indefinite)
 * @return OS_SUCCESS if transition was successful
 */
OsError os_power_force_mode(PowerMode mode, uint32_t timeout_ms);

#endif /* OS_POWER_H */
