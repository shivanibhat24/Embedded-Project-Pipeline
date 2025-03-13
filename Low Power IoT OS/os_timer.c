/**
 * @file os_timer.c
 * @brief Timer services implementation for MicroOS
 */

#include "os_timer.h"
#include "os_hal.h"
#include "os_core.h"
#include <stdio.h>

// Maximum number of software timers
#define MAX_TIMERS 16

// Timer state structure
typedef struct {
    bool active;
    uint32_t interval_ms;
    uint32_t expiry_time;
    bool periodic;
    TimerCallback callback;
    void* user_data;
} Timer;

// Timer management state
static struct {
    bool initialized;
    Timer timers[MAX_TIMERS];
    uint8_t timer_count;
    uint32_t next_expiry;
} timer_state = {0};

OsError os_timer_init(void) {
    if (timer_state.initialized) {
        return OS_SUCCESS;  // Already initialized
    }
    
    printf("Initializing timer subsystem\n");
    
    // Initialize timer hardware
    HalError hal_result = hal_timer_init();
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to initialize timer hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    // Initialize timer state
    timer_state.timer_count = 0;
    timer_state.next_expiry = UINT32_MAX;
    
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        timer_state.timers[i].active = false;
    }
    
    timer_state.initialized = true;
    printf("Timer subsystem initialized successfully\n");
    
    return OS_SUCCESS;
}

TimerHandle os_timer_create(uint32_t interval_ms, bool periodic, TimerCallback callback, void* user_data) {
    if (!callback || interval_ms == 0) {
        return INVALID_TIMER_HANDLE;
    }
    
    // Find a free timer slot
    TimerHandle handle = INVALID_TIMER_HANDLE;
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (!timer_state.timers[i].active) {
            handle = i;
            break;
        }
    }
    
    if (handle == INVALID_TIMER_HANDLE) {
        printf("Failed to create timer: no free timer slots\n");
        return INVALID_TIMER_HANDLE;
    }
    
    // Configure the timer
    timer_state.timers[handle].interval_ms = interval_ms;
    timer_state.timers[handle].periodic = periodic;
    timer_state.timers[handle].callback = callback;
    timer_state.timers[handle].user_data = user_data;
    timer_state.timers[handle].active = false;  // Not started yet
    
    return handle;
}

OsError os_timer_start(TimerHandle timer) {
    if (timer >= MAX_TIMERS || !timer_state.timers[timer].active) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    uint32_t current_time = os_get_uptime_ms();
    timer_state.timers[timer].expiry_time = current_time + timer_state.timers[timer].interval_ms;
    timer_state.timers[timer].active = true;
    
    // Update next expiry time if needed
    if (timer_state.timers[timer].expiry_time < timer_state.next_expiry) {
        timer_state.next_expiry = timer_state.timers[timer].expiry_time;
        hal_set_timer_alarm(timer_state.next_expiry);
    }
    
    return OS_SUCCESS;
}

OsError os_timer_stop(TimerHandle timer) {
    if (timer >= MAX_TIMERS || !timer_state.timers[timer].active) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    timer_state.timers[timer].active = false;
    
    // Recalculate next expiry time
    timer_state.next_expiry = UINT32_MAX;
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (timer_state.timers[i].active && timer_state.timers[i].expiry_time < timer_state.next_expiry) {
            timer_state.next_expiry = timer_state.timers[i].expiry_time;
        }
    }
    
    if (timer_state.next_expiry != UINT32_MAX) {
        hal_set_timer_alarm(timer_state.next_expiry);
    }
    
    return OS_SUCCESS;
}

OsError os_timer_reset(TimerHandle timer) {
    if (timer >= MAX_TIMERS) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    // Stop and restart the timer
    timer_state.timers[timer].active = false;
    return os_timer_start(timer);
}

OsError os_timer_delete(TimerHandle timer) {
    if (timer >= MAX_TIMERS) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    // Mark as inactive
    timer_state.timers[timer].active = false;
    
    return OS_SUCCESS;
}

void os_timer_process(void) {
    uint32_t current_time = os_get_uptime_ms();
    bool recalculate_next = false;
    
    // Process expired timers
    for (uint8_t i = 0; i < MAX_TIMERS; i++) {
        if (timer_state.timers[i].active && current_time >= timer_state.timers[i].expiry_time) {
            // Timer expired
            TimerCallback callback = timer_state.timers[i].callback;
            void* user_data = timer_state.timers[i].user_data;
            
            if (timer_state.timers[i].periodic) {
                // Reschedule periodic timer
                timer_state.timers[i].expiry_time = current_time + timer_state.timers[i].interval_ms;
            } else {
                // One-shot timer - mark as inactive
                timer_state.timers[i].active = false;
            }
            
            recalculate_next = true;
            
            // Call the callback after updating timer state
            if (callback) {
                callback(i, user_data);
            }
        }
    }
    
    // Recalculate next expiry time if needed
    if (recalculate_next) {
        timer_state.next_expiry = UINT32_MAX;
        for (uint8_t i = 0; i < MAX_TIMERS; i++) {
            if (timer_state.timers[i].active && timer_state.timers[i].expiry_time < timer_state.next_expiry) {
                timer_state.next_expiry = timer_state.timers[i].expiry_time;
            }
        }
        
        if (timer_state.next_expiry != UINT32_MAX) {
            hal_set_timer_alarm(timer_state.next_expiry);
        }
    }
}

uint32_t os_timer_get_next_expiry(void) {
    return timer_state.next_expiry;
}

uint32_t os_timer_get_remaining(TimerHandle timer) {
    if (timer >= MAX_TIMERS || !timer_state.timers[timer].active) {
        return 0;
    }
    
    uint32_t current_time = os_get_uptime_ms();
    if (current_time >= timer_state.timers[timer].expiry_time) {
        return 0;
    }
    
    return timer_state.timers[timer].expiry_time - current_time;
}
