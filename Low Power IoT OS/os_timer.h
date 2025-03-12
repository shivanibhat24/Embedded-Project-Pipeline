/**
 * @file os_timer.h
 * @brief Timer management for MicroOS
 */

#ifndef OS_TIMER_H
#define OS_TIMER_H

#include "os_core.h"

/**
 * @brief Timer handle
 */
typedef uint8_t TimerHandle;

/**
 * @brief Timer callback function
 */
typedef void (*TimerCallback)(TimerHandle timer_id, void* param);

/**
 * @brief Timer mode
 */
typedef enum {
    TIMER_MODE_ONE_SHOT,  // Timer runs once
    TIMER_MODE_PERIODIC   // Timer runs periodically
} TimerMode;

/**
 * @brief Create a timer
 * 
 * @param name Timer name
 * @param period_ms Period in milliseconds
 * @param mode Timer mode
 * @param callback Callback function
 * @param callback_param Parameter to pass to callback
 * @param handle Pointer to store timer handle
 * @return OS_SUCCESS if timer was created successfully
 */
OsError os_timer_create(const char* name, uint32_t period_ms, TimerMode mode, 
                       TimerCallback callback, void* callback_param, TimerHandle* handle);

/**
 * @brief Delete a timer
 * 
 * @param handle Timer handle
 * @return OS_SUCCESS if timer was deleted successfully
 */
OsError os_timer_delete(TimerHandle handle);

/**
 * @brief Start a timer
 * 
 * @param handle Timer handle
 * @return OS_SUCCESS if timer was started successfully
 */
OsError os_timer_start(TimerHandle handle);

/**
 * @brief Stop a timer
 * 
 * @param handle Timer handle
 * @return OS_SUCCESS if timer was stopped successfully
 */
OsError os_timer_stop(TimerHandle handle);

/**
 * @brief Reset a timer
 * 
 * @param handle Timer handle
 * @return OS_SUCCESS if timer was reset successfully
 */
OsError os_timer_reset(TimerHandle handle);

/**
 * @brief Change a timer's period
 * 
 * @param handle Timer handle
 * @param period_ms New period in milliseconds
 * @return OS_SUCCESS if period was changed successfully
 */
OsError os_timer_change_period(TimerHandle handle, uint32_t period_ms);

/**
 * @brief Get time remaining until timer expiry
 * 
 * @param handle Timer handle
 * @param remaining_ms Pointer to store remaining time in milliseconds
 * @return OS_SUCCESS if time was retrieved successfully
 */
OsError os_timer_get_remaining(TimerHandle handle, uint32_t* remaining_ms);

/**
 * @brief Check if a timer is active
 * 
 * @param handle Timer handle
 * @param is_active Pointer to store result
 * @return OS_SUCCESS if check was successful
 */
OsError os_timer_is_active(TimerHandle handle, bool* is_active);

/**
 * @brief Get timer period
 * 
 * @param handle Timer handle
 * @param period_ms Pointer to store period in milliseconds
 * @return OS_SUCCESS if period was retrieved successfully
 */
OsError os_timer_get_period(TimerHandle handle, uint32_t* period_ms);

/**
 * @brief Register a low-power timer for system wake-up
 * 
 * Low-power timers can wake the system from sleep modes.
 * 
 * @param handle Timer handle
 * @param wakeup_mode Minimum power mode to wake from
 * @return OS_SUCCESS if registration was successful
 */
OsError os_timer_register_wakeup(TimerHandle handle, PowerMode wakeup_mode);

/**
 * @brief Configure a timer's power impact
 * 
 * @param handle Timer handle
 * @param allow_sleep Allow the system to enter sleep while timer is active
 * @return OS_SUCCESS if configuration was successful
 */
OsError os_timer_configure_power_impact(TimerHandle handle, bool allow_sleep);

#endif /* OS_TIMER_H */
