/**
 * @file os_power.c
 * @brief Power management implementation for MicroOS
 */

#include "os_power.h"
#include "os_hal.h"
#include <stdio.h>

// Power management state
static struct {
    bool initialized;
    WakeupConfig wakeup_sources[MAX_WAKEUP_SOURCES];
    uint8_t wakeup_source_count;
    uint32_t last_sleep_time;
    uint32_t total_sleep_time;
} power_state = {0};

OsError os_power_init(void) {
    if (power_state.initialized) {
        return OS_SUCCESS;  // Already initialized
    }
    
    printf("Initializing power management subsystem\n");
    
    // Initialize power state
    power_state.wakeup_source_count = 0;
    power_state.last_sleep_time = 0;
    power_state.total_sleep_time = 0;
    
    // Set up default power configurations in hardware
    HalError hal_result = hal_configure_power_system();
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to configure power hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    power_state.initialized = true;
    printf("Power management initialized successfully\n");
    
    return OS_SUCCESS;
}

OsError os_power_configure_wakeup(const WakeupConfig* config) {
    if (!config) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    if (power_state.wakeup_source_count >= MAX_WAKEUP_SOURCES) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Check if this source is already configured
    for (uint8_t i = 0; i < power_state.wakeup_source_count; i++) {
        if (power_state.wakeup_sources[i].source == config->source) {
            // Update existing configuration
            power_state.wakeup_sources[i] = *config;
            
            // Apply to hardware
            HalError hal_result = hal_configure_wakeup_source(config);
            if (hal_result != HAL_SUCCESS) {
                printf("Failed to configure wakeup source in hardware: %d\n", hal_result);
                return OS_ERROR_HARDWARE;
            }
            
            return OS_SUCCESS;
        }
    }
    
    // Add new configuration
    power_state.wakeup_sources[power_state.wakeup_source_count] = *config;
    power_state.wakeup_source_count++;
    
    // Apply to hardware
    HalError hal_result = hal_configure_wakeup_source(config);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to configure wakeup source in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    return OS_SUCCESS;
}

OsError os_power_disable_wakeup(WakeupSource source) {
    bool found = false;
    
    for (uint8_t i = 0; i < power_state.wakeup_source_count; i++) {
        if (power_state.wakeup_sources[i].source == source) {
            // Remove by shifting remaining elements
            for (uint8_t j = i; j < power_state.wakeup_source_count - 1; j++) {
                power_state.wakeup_sources[j] = power_state.wakeup_sources[j + 1];
            }
            power_state.wakeup_source_count--;
            found = true;
            break;
        }
    }
    
    if (!found) {
        return OS_ERROR_NOT_FOUND;
    }
    
    // Disable in hardware
    HalError hal_result = hal_disable_wakeup_source(source);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to disable wakeup source in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    return OS_SUCCESS;
}

WakeupSource os_power_get_last_wakeup_source(void) {
    return hal_get_last_wakeup_source();
}

uint32_t os_power_get_sleep_time(void) {
    return power_state.total_sleep_time;
}

void os_power_before_sleep(PowerMode mode) {
    power_state.last_sleep_time = hal_get_system_time();
    
    // Configure all enabled wake-up sources before sleep
    for (uint8_t i = 0; i < power_state.wakeup_source_count; i++) {
        hal_configure_wakeup_source(&power_state.wakeup_sources[i]);
    }
    
    printf("Preparing to enter power mode: %d\n", mode);
}

void os_power_after_wakeup(PowerMode mode) {
    uint32_t current_time = hal_get_system_time();
    uint32_t sleep_duration = current_time - power_state.last_sleep_time;
    power_state.total_sleep_time += sleep_duration;
    
    WakeupSource wake_source = hal_get_last_wakeup_source();
    printf("Woke up from power mode %d after %lu ms by source %d\n", 
           mode, sleep_duration, wake_source);
}

OsError os_power_get_consumption_estimate(PowerConsumptionStats* stats) {
    if (!stats) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Get hardware-specific power consumption data
    HalError hal_result = hal_get_power_stats(stats);
    if (hal_result != HAL_SUCCESS) {
        return OS_ERROR_HARDWARE;
    }
    
    return OS_SUCCESS;
}
