/**
 * @file os_core.c
 * @brief Core implementation for MicroOS
 */

#include "os_core.h"
#include "os_task.h"
#include "os_power.h"
#include "os_hal.h"
#include "os_timer.h"
#include <stdio.h>

// System state
static struct {
    bool initialized;
    uint32_t uptime_ms;
    PowerMode current_power_mode;
    PowerMode target_power_mode;
    uint32_t idle_threshold_ms;
    uint32_t last_activity_time;
    PowerStateCallback power_callbacks[5];
    uint8_t power_callback_count;
} os_state = {0};

OsError os_init(void) {
    if (os_state.initialized) {
        return OS_SUCCESS;  // Already initialized
    }
    
    printf("MicroOS v%d.%d.%d initializing...\n", 
           MICROOS_VERSION_MAJOR, MICROOS_VERSION_MINOR, MICROOS_VERSION_PATCH);
    
    // Initialize hardware abstraction layer
    HalError hal_result = hal_init();
    if (hal_result != HAL_SUCCESS) {
        printf("HAL initialization failed: %d\n", hal_result);
        return OS_ERROR_GENERIC;
    }
    
    // Initialize power management
    OsError power_result = os_power_init();
    if (power_result != OS_SUCCESS) {
        printf("Power management initialization failed: %d\n", power_result);
        return power_result;
    }
    
    // Initialize default values
    os_state.uptime_ms = 0;
    os_state.current_power_mode = POWER_MODE_ACTIVE;
    os_state.target_power_mode = POWER_MODE_ACTIVE;
    os_state.idle_threshold_ms = 5000;  // Default 5 seconds before idle
    os_state.last_activity_time = 0;
    os_state.power_callback_count = 0;
    
    // Configure wake-up sources
    WakeupConfig rtc_wakeup = {
        .source = WAKEUP_SOURCE_RTC,
        .timeout_ms = 1000  // 1 second default RTC wake-up
    };
    os_power_configure_wakeup(&rtc_wakeup);
    
    WakeupConfig gpio_wakeup = {
        .source = WAKEUP_SOURCE_GPIO,
        .gpio_pin = 0,
        .gpio_rising_edge = true
    };
    os_power_configure_wakeup(&gpio_wakeup);
    
    os_state.initialized = true;
    printf("MicroOS initialized successfully\n");
    
    return OS_SUCCESS;
}

void os_run(void) {
    if (!os_state.initialized) {
        printf("Error: MicroOS not initialized\n");
        return;
    }
    
    printf("MicroOS starting main execution loop\n");
    
    while (1) {
        uint32_t current_time = hal_get_system_time();
        os_state.uptime_ms = current_time;
        
        // Run the scheduler to execute tasks
        bool tasks_executed = os_scheduler_run_once();
        
        // Check for idle condition
        if (tasks_executed) {
            os_state.last_activity_time = current_time;
        } else if ((current_time - os_state.last_activity_time) > os_state.idle_threshold_ms) {
            // System is idle, enter power saving mode
            os_set_power_mode(POWER_MODE_SLEEP);
        }
        
        // Process pending power mode transitions
        if (os_state.current_power_mode != os_state.target_power_mode) {
            // Notify power callbacks
            for (uint8_t i = 0; i < os_state.power_callback_count; i++) {
                if (os_state.power_callbacks[i]) {
                    os_state.power_callbacks[i](os_state.current_power_mode, os_state.target_power_mode);
                }
            }
            
            // Enter the new power mode
            hal_enter_power_mode(os_state.target_power_mode);
            
            // Update current mode
            PowerMode previous_mode = os_state.current_power_mode;
            os_state.current_power_mode = os_state.target_power_mode;
            
            printf("Power mode changed: %d -> %d\n", previous_mode, os_state.current_power_mode);
        }
    }
}

uint32_t os_get_uptime_ms(void) {
    return os_state.uptime_ms;
}

OsError os_set_power_mode(PowerMode mode) {
    if (mode < POWER_MODE_ACTIVE || mode > POWER_MODE_HIBERNATION) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    os_state.target_power_mode = mode;
    return OS_SUCCESS;
}

PowerMode os_get_power_mode(void) {
    return os_state.current_power_mode;
}

OsError os_register_power_callback(void (*callback)(PowerMode old_mode, PowerMode new_mode)) {
    if (!callback) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    if (os_state.power_callback_count >= 5) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    os_state.power_callbacks[os_state.power_callback_count++] = callback;
    return OS_SUCCESS;
}

OsError os_configure_idle_behavior(uint32_t idle_threshold_ms, PowerMode default_sleep_mode) {
    if (default_sleep_mode < POWER_MODE_IDLE || default_sleep_mode > POWER_MODE_HIBERNATION) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    os_state.idle_threshold_ms = idle_threshold_ms;
    os_state.target_power_mode = default_sleep_mode;
    return OS_SUCCESS;
}
