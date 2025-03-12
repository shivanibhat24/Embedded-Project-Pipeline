/**
 * @file os_core.h
 * @brief Core definitions for MicroOS - A lightweight IoT operating system
 * 
 * MicroOS is designed for low-power IoT applications with focus on:
 * - Energy efficiency with intelligent power management
 * - Lightweight task scheduling
 * - Hardware abstraction for portability
 * - Dynamic wake-up timing based on application needs
 */

#ifndef OS_CORE_H
#define OS_CORE_H

#include <stdint.h>
#include <stdbool.h>

/**
 * @brief MicroOS version information
 */
#define MICROOS_VERSION_MAJOR 1
#define MICROOS_VERSION_MINOR 0
#define MICROOS_VERSION_PATCH 0

/**
 * @brief System configuration constants
 */
#define MAX_TASKS 16
#define MAX_TASK_NAME_LENGTH 16
#define MAX_TIMERS 8
#define MAX_MUTEX 8
#define MAX_SEMAPHORES 8
#define MAX_QUEUES 4
#define MAX_QUEUE_SIZE 32

/**
 * @brief Power modes enumeration
 */
typedef enum {
    POWER_MODE_ACTIVE,       // Full power operation
    POWER_MODE_IDLE,         // CPU idle, peripherals active
    POWER_MODE_SLEEP,        // CPU + some peripherals off
    POWER_MODE_DEEP_SLEEP,   // Only wake-up sources active
    POWER_MODE_HIBERNATION   // Everything off except RTC
} PowerMode;

/**
 * @brief Error codes
 */
typedef enum {
    OS_SUCCESS = 0,
    OS_ERROR_GENERIC,
    OS_ERROR_TIMEOUT,
    OS_ERROR_RESOURCE_UNAVAILABLE,
    OS_ERROR_INVALID_PARAMETER,
    OS_ERROR_OUT_OF_MEMORY,
    OS_ERROR_BUFFER_OVERFLOW,
    OS_ERROR_NOT_INITIALIZED
} OsError;

/**
 * @brief MicroOS initialization
 * 
 * Must be called before any other MicroOS function.
 * Initializes the scheduler, HAL, power management, and system timers.
 * 
 * @return OS_SUCCESS on successful initialization
 */
OsError os_init(void);

/**
 * @brief MicroOS main execution loop
 * 
 * This function never returns under normal operation. It runs the 
 * scheduler and power management systems.
 */
void os_run(void);

/**
 * @brief Get current system uptime
 * 
 * @return System uptime in milliseconds
 */
uint32_t os_get_uptime_ms(void);

/**
 * @brief Set system to a specific power mode
 * 
 * @param mode Power mode to enter
 * @return OS_SUCCESS if mode was entered successfully
 */
OsError os_set_power_mode(PowerMode mode);

/**
 * @brief Get current system power mode
 * 
 * @return Current power mode
 */
PowerMode os_get_power_mode(void);

/**
 * @brief Register a callback for power mode changes
 * 
 * @param callback Function to call when power mode changes
 * @return OS_SUCCESS if callback was registered successfully
 */
OsError os_register_power_callback(void (*callback)(PowerMode old_mode, PowerMode new_mode));

/**
 * @brief Configure system idle behavior
 * 
 * @param idle_threshold_ms Time in ms before entering low-power mode
 * @param default_sleep_mode Default sleep mode to enter
 * @return OS_SUCCESS if configuration was successful
 */
OsError os_configure_idle_behavior(uint32_t idle_threshold_ms, PowerMode default_sleep_mode);

#endif /* OS_CORE_H */
