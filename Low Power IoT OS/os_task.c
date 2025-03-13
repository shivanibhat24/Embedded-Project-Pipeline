/**
 * @file os_task.c
 * @brief Task management implementation for MicroOS
 */

#include "os_task.h"
#include "os_core.h"
#include "os_hal.h"
#include <string.h>
#include <stdio.h>

// Task Control Block (TCB)
typedef struct {
    TaskConfig config;
    TaskState state;
    TaskStats stats;
    uint32_t last_run_time;
    PowerMode power_restriction;
    uint8_t* stack;
    uint16_t stack_size;
    bool valid;
} TaskControlBlock;

// Task list
static TaskControlBlock task_list[MAX_TASKS];
static uint8_t task_count = 0;
static TaskHandle current_task = 0xFF;
static TaskSchedulingPolicy scheduling_policy = TASK_SCHED_PRIORITY;

OsError os_task_create(const TaskConfig* config, TaskHandle* handle) {
    if (!config || !handle) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    if (task_count >= MAX_TASKS) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Find an empty slot
    uint8_t task_index = 0xFF;
    for (uint8_t i = 0; i < MAX_TASKS; i++) {
        if (!task_list[i].valid) {
            task_index = i;
            break;
        }
    }
    
    if (task_index == 0xFF) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Initialize the task
    memset(&task_list[task_index], 0, sizeof(TaskControlBlock));
    memcpy(&task_list[task_index].config, config, sizeof(TaskConfig));
    
    // Allocate stack if not already allocated
    if (!task_list[task_index].stack && config->stack_size > 0) {
        task_list[task_index].stack = (uint8_t*)malloc(config->stack_size);
        if (!task_list[task_index].stack) {
            return OS_ERROR_OUT_OF_MEMORY;
        }
        task_list[task_index].stack_size = config->stack_size;
    }
    
    // Initialize task state
    task_list[task_index].state = config->auto_start ? TASK_STATE_READY : TASK_STATE_DORMANT;
    task_list[task_index].power_restriction = POWER_MODE_ACTIVE;
    task_list[task_index].valid = true;
    task_list[task_index].last_run_time = os_get_uptime_ms();
    
    // Initialize statistics
    task_list[task_index].stats.total_runtime_ms = 0;
    task_list[task_index].stats.wakeups = 0;
    task_list[task_index].stats.sleeps = 0;
    
    // Set the task handle
    *handle = task_index;
    task_count++;
    
    return OS_ERROR_NONE;
}

OsError os_task_delete(TaskHandle handle) {
    if (handle >= MAX_TASKS || !task_list[handle].valid) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Free stack if allocated
    if (task_list[handle].stack) {
        free(task_list[handle].stack);
        task_list[handle].stack = NULL;
    }
    
    // Mark task as invalid
    task_list[handle].valid = false;
    task_count--;
    
    // If currently running task is deleted, force a context switch
    if (current_task == handle) {
        current_task = 0xFF;
        os_schedule();
    }
    
    return OS_ERROR_NONE;
}

OsError os_task_set_state(TaskHandle handle, TaskState state) {
    if (handle >= MAX_TASKS || !task_list[handle].valid) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Update state
    TaskState old_state = task_list[handle].state;
    task_list[handle].state = state;
    
    // Update statistics
    if (old_state == TASK_STATE_SLEEPING && state == TASK_STATE_READY) {
        task_list[handle].stats.wakeups++;
    } else if (old_state == TASK_STATE_READY && state == TASK_STATE_SLEEPING) {
        task_list[handle].stats.sleeps++;
    }
    
    // If task is current and going to sleep, force a context switch
    if (current_task == handle && state == TASK_STATE_SLEEPING) {
        current_task = 0xFF;
        os_schedule();
    }
    
    return OS_ERROR_NONE;
}

OsError os_task_set_power_restriction(TaskHandle handle, PowerMode mode) {
    if (handle >= MAX_TASKS || !task_list[handle].valid) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    task_list[handle].power_restriction = mode;
    
    // Update system power mode if necessary
    os_update_power_mode();
    
    return OS_ERROR_NONE;
}

TaskHandle os_task_get_current(void) {
    return current_task;
}

OsError os_task_yield(void) {
    if (current_task == 0xFF) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Force a context switch
    os_schedule();
    
    return OS_ERROR_NONE;
}

OsError os_task_sleep(uint32_t ms) {
    if (current_task == 0xFF) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Set task to sleeping state
    task_list[current_task].state = TASK_STATE_SLEEPING;
    task_list[current_task].stats.sleeps++;
    
    // Set wake time
    uint32_t wake_time = os_get_uptime_ms() + ms;
    task_list[current_task].config.wake_time = wake_time;
    
    // Force a context switch
    current_task = 0xFF;
    os_schedule();
    
    return OS_ERROR_NONE;
}

void os_task_set_scheduling_policy(TaskSchedulingPolicy policy) {
    scheduling_policy = policy;
}

TaskSchedulingPolicy os_task_get_scheduling_policy(void) {
    return scheduling_policy;
}

OsError os_task_get_stats(TaskHandle handle, TaskStats* stats) {
    if (handle >= MAX_TASKS || !task_list[handle].valid || !stats) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    memcpy(stats, &task_list[handle].stats, sizeof(TaskStats));
    return OS_ERROR_NONE;
}

// Internal function to schedule next task
void os_schedule(void) {
    // Update stats for current task if applicable
    if (current_task != 0xFF && task_list[current_task].valid) {
        uint32_t current_time = os_get_uptime_ms();
        uint32_t runtime = current_time - task_list[current_task].last_run_time;
        task_list[current_task].stats.total_runtime_ms += runtime;
    }
    
    // Check if any sleeping tasks need to wake up
    uint32_t current_time = os_get_uptime_ms();
    for (uint8_t i = 0; i < MAX_TASKS; i++) {
        if (task_list[i].valid && task_list[i].state == TASK_STATE_SLEEPING) {
            if (task_list[i].config.wake_time <= current_time) {
                task_list[i].state = TASK_STATE_READY;
                task_list[i].stats.wakeups++;
            }
        }
    }
    
    // Find next task to run based on scheduling policy
    TaskHandle next_task = 0xFF;
    
    if (scheduling_policy == TASK_SCHED_PRIORITY) {
        // Priority-based scheduling
        uint8_t highest_priority = 0;
        
        for (uint8_t i = 0; i < MAX_TASKS; i++) {
            if (task_list[i].valid && task_list[i].state == TASK_STATE_READY) {
                if (task_list[i].config.priority > highest_priority) {
                    highest_priority = task_list[i].config.priority;
                    next_task = i;
                }
            }
        }
    } else if (scheduling_policy == TASK_SCHED_ROUND_ROBIN) {
        // Round-robin scheduling
        if (current_task == 0xFF) {
            // Start from the beginning if no task is running
            for (uint8_t i = 0; i < MAX_TASKS; i++) {
                if (task_list[i].valid && task_list[i].state == TASK_STATE_READY) {
                    next_task = i;
                    break;
                }
            }
        } else {
            // Start from the next task after the current one
            for (uint8_t i = 1; i <= MAX_TASKS; i++) {
                uint8_t idx = (current_task + i) % MAX_TASKS;
                if (task_list[idx].valid && task_list[idx].state == TASK_STATE_READY) {
                    next_task = idx;
                    break;
                }
            }
        }
    }
    
    // Update current task
    if (next_task != 0xFF) {
        current_task = next_task;
        task_list[current_task].last_run_time = os_get_uptime_ms();
        
        // Execute the task
        if (task_list[current_task].config.task_func) {
            task_list[current_task].config.task_func(task_list[current_task].config.param);
        }
    } else {
        // No tasks ready to run, enter idle state
        os_enter_idle();
    }
}

// Function to update system power mode based on task restrictions
void os_update_power_mode(void) {
    PowerMode lowest_mode = POWER_MODE_DEEP_SLEEP;
    
    for (uint8_t i = 0; i < MAX_TASKS; i++) {
        if (task_list[i].valid && task_list[i].state != TASK_STATE_DORMANT) {
            if (task_list[i].power_restriction < lowest_mode) {
                lowest_mode = task_list[i].power_restriction;
            }
        }
    }
    
    // Set system power mode
    os_set_power_mode(lowest_mode);
}

// Function to initialize the task system
void os_task_init(void) {
    // Initialize task list
    memset(task_list, 0, sizeof(task_list));
    task_count = 0;
    current_task = 0xFF;
    scheduling_policy = TASK_SCHED_PRIORITY;
}
