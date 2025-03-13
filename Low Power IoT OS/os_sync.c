/**
 * @file os_sync.c
 * @brief Synchronization primitives implementation for MicroOS
 */

#include "os_sync.h"
#include "os_task.h"
#include "os_core.h"
#include <stdio.h>

// Maximum number of semaphores and mutexes
#define MAX_SEMAPHORES 8
#define MAX_MUTEXES 8

// Semaphore structure
typedef struct {
    bool initialized;
    uint8_t count;
    uint8_t max_count;
    TaskId waiting_tasks[MAX_TASKS];
    uint8_t waiting_task_count;
} Semaphore;

// Mutex structure
typedef struct {
    bool initialized;
    bool locked;
    TaskId owner;
    TaskId waiting_tasks[MAX_TASKS];
    uint8_t waiting_task_count;
} Mutex;

// Synchronization state
static struct {
    Semaphore semaphores[MAX_SEMAPHORES];
    Mutex mutexes[MAX_MUTEXES];
} sync_state = {0};

OsError os_semaphore_create(SemaphoreHandle* handle, uint8_t initial_count, uint8_t max_count) {
    if (!handle || max_count == 0 || initial_count > max_count) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Find a free semaphore slot
    SemaphoreHandle sem_handle = INVALID_SEMAPHORE_HANDLE;
    for (uint8_t i = 0; i < MAX_SEMAPHORES; i++) {
        if (!sync_state.semaphores[i].initialized) {
            sem_handle = i;
            break;
        }
    }
    
    if (sem_handle == INVALID_SEMAPHORE_HANDLE) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Initialize the semaphore
    sync_state.semaphores[sem_handle].initialized = true;
    sync_state.semaphores[sem_handle].count = initial_count;
    sync_state.semaphores[sem_handle].max_count = max_count;
    sync_state.semaphores[sem_handle].waiting_task_count = 0;
    
    *handle = sem_handle;
    return OS_SUCCESS;
}

OsError os_semaphore_take(SemaphoreHandle handle, uint32_t timeout_ms) {
    if (handle >= MAX_SEMAPHORES || !sync_state.semaphores[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    TaskId current_task = os_task_get_current();
    uint32_t start_time = os_get_uptime_ms();
    
    // Try to take the semaphore
    while (sync_state.semaphores[handle].count == 0) {
        // No available count, need to wait
        if (timeout_ms == 0) {
            // No waiting, return immediately
            return OS_ERROR_TIMEOUT;
        }
        
        // Add task to waiting list
        if (sync_state.semaphores[handle].waiting_task_count < MAX_TASKS) {
            sync_state.semaphores[handle].waiting_tasks[sync_state.semaphores[handle].waiting_task_count++] = current_task;
        } else {
            return OS_ERROR_RESOURCE_UNAVAILABLE;
        }
        
        // Block the task
        os_task_block(current_task, timeout_ms);
        
        // When task resumes, check if timeout occurred
        uint32_t elapsed = os_get_uptime_ms() - start_time;
        if (elapsed >= timeout_ms) {
            return OS_ERROR_TIMEOUT;
        }
        
        // Adjust remaining timeout
        timeout_ms -= elapsed;
    }
    
    // Take the semaphore
    sync_state.semaphores[handle].count--;
    return OS_SUCCESS;
}

OsError os_semaphore_give(SemaphoreHandle handle) {
    if (handle >= MAX_SEMAPHORES || !sync_state.semaphores[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sync_state.semaphores[handle].count >= sync_state.semaphores[handle].max_count) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Increment the count
    sync_state.semaphores[handle].count++;
    
    // Check if any tasks are waiting
    if (sync_state.semaphores[handle].waiting_task_count > 0) {
        // Unblock the first waiting task
        TaskId task_to_unblock = sync_state.semaphores[handle].waiting_tasks[0];
        
        // Remove task from waiting list
        for (uint8_t i = 0; i < sync_state.semaphores[handle].waiting_task_count - 1; i++) {
            sync_state.semaphores[handle].waiting_tasks[i] = sync_state.semaphores[handle].waiting_tasks[i + 1];
        }
        sync_state.semaphores[handle].waiting_task_count--;
        
        // Unblock the task
        os_task_unblock(task_to_unblock);
    }
    
    return OS_SUCCESS;
}

OsError os_semaphore_delete(SemaphoreHandle handle) {
    if (handle >= MAX_SEMAPHORES || !sync_state.semaphores[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    // Unblock all waiting tasks
    for (uint8_t i = 0; i < sync_state.semaphores[handle].waiting_task_count; i++) {
        os_task_unblock(sync_state.semaphores[handle].waiting_tasks[i]);
    }
    
    // Mark as uninitialized
    sync_state.semaphores[handle].initialized = false;
    
    return OS_SUCCESS;
}

OsError os_mutex_create(MutexHandle* handle) {
    if (!handle) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Find a free mutex slot
    MutexHandle mtx_handle = INVALID_MUTEX_HANDLE;
    for (uint8_t i = 0; i < MAX_MUTEXES; i++) {
        if (!sync_state.mutexes[i].initialized) {
            mtx_handle = i;
            break;
        }
    }
    
    if (mtx_handle == INVALID_MUTEX_HANDLE) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Initialize the mutex
    sync_state.mutexes[mtx_handle].initialized = true;
    sync_state.mutexes[mtx_handle].locked = false;
    sync_state.mutexes[mtx_handle].owner = INVALID_TASK_ID;
    sync_state.mutexes[mtx_handle].waiting_task_count = 0;
    
    *handle = mtx_handle;
    return OS_SUCCESS;
}

OsError os_mutex_lock(MutexHandle handle, uint32_t timeout_ms) {
    if (handle >= MAX_MUTEXES || !sync_state.mutexes[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    TaskId current_task = os_task_get_current();
    uint32_t start_time = os_get_uptime_ms();
    
    // Check for mutex recursion (same task already owns it)
    if (sync_state.mutexes[handle].locked && sync_state.mutexes[handle].owner == current_task) {
        return OS_ERROR_WOULD_DEADLOCK;
    }
    
    // Try to lock the mutex
    while (sync_state.mutexes[handle].locked) {
        // Mutex is locked, need to wait
        if (timeout_ms == 0) {
            // No waiting, return immediately
            return OS_ERROR_TIMEOUT;
        }
        
        // Add task to waiting list
        if (sync_state.mutexes[handle].waiting_task_count < MAX_TASKS) {
            sync_state.mutexes[handle].waiting_tasks[sync_state.mutexes[handle].waiting_task_count++] = current_task;
        } else {
            return OS_ERROR_RESOURCE_UNAVAILABLE;
        }
        
        // Block the task
        os_task_block(current_task, timeout_ms);
        
        // When task resumes, check if timeout occurred
        uint32_t elapsed = os_get_uptime_ms() - start_time;
        if (elapsed >= timeout_ms) {
            return OS_ERROR_TIMEOUT;
        }
        
        // Adjust remaining timeout
        timeout_ms -= elapsed;
    }
    
    // Lock the mutex
    sync_state.mutexes[handle].locked = true;
    sync_state.mutexes[handle].owner = current_task;
    
    return OS_SUCCESS;
}

OsError os_mutex_unlock(MutexHandle handle) {
    if (handle >= MAX_MUTEXES || !sync_state.mutexes[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    TaskId current_task = os_task_get_current();
    
    // Check if the caller owns the mutex
    if (!sync_state.mutexes[handle].locked || sync_state.mutexes[handle].owner != current_task) {
        return OS_ERROR_NOT_OWNER;
    }
    
    // Unlock the mutex
    sync_state.mutexes[handle].locked = false;
    sync_state.mutexes[handle].owner = INVALID_TASK_ID;
    
    // Check if any tasks are waiting
    if (sync_state.mutexes[handle].waiting_task_count > 0) {
        // Unblock the first waiting task
        TaskId task_to_unblock = sync_state.mutexes[handle].waiting_tasks[0];
        
        // Remove task from waiting list
        for (uint8_t i = 0; i < sync_state.mutexes[handle].waiting_task_count - 1; i++) {
            sync_state.mutexes[handle].waiting_tasks[i] = sync_state.mutexes[handle].waiting_tasks[i + 1];
        }
        sync_state.mutexes[handle].waiting_task_count--;
        
        // Unblock the task
        os_task_unblock(task_to_unblock);
    }
    
    return OS_SUCCESS;
}

OsError os_mutex_delete(MutexHandle handle) {
    if (handle >= MAX_MUTEXES || !sync_state.mutexes[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    // Check if the mutex is currently locked
    if (sync_state.mutexes[handle].locked) {
        return OS_ERROR_RESOURCE_BUSY;
    }
    
    // Unblock all waiting tasks
    for (uint8_t i = 0; i < sync_state.mutexes[handle].waiting_task_count; i++) {
        os_task_unblock(sync_state.mutexes[handle].waiting_tasks[i]);
    }
    
    // Mark as uninitialized
    sync_state.mutexes[handle].initialized = false;
    
    return OS_SUCCESS;
}
