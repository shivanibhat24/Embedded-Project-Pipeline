/**
 * @file os_task.h
 * @brief Task management for MicroOS
 */

#ifndef OS_TASK_H
#define OS_TASK_H

#include "os_core.h"

/**
 * @brief Task state enumeration
 */
typedef enum {
    TASK_STATE_READY,       // Task is ready to run
    TASK_STATE_RUNNING,     // Task is currently running
    TASK_STATE_BLOCKED,     // Task is blocked (mutex, semaphore, etc.)
    TASK_STATE_SUSPENDED,   // Task is suspended
    TASK_STATE_DORMANT      // Task exists but is not scheduled
} TaskState;

/**
 * @brief Task scheduling policy
 */
typedef enum {
    TASK_SCHED_PRIORITY,    // Priority-based scheduling
    TASK_SCHED_ROUND_ROBIN, // Round-robin scheduling
    TASK_SCHED_COOPERATIVE  // Cooperative scheduling
} TaskSchedulingPolicy;

/**
 * @brief Task handle
 */
typedef uint8_t TaskHandle;

/**
 * @brief Task configuration structure
 */
typedef struct {
    void (*task_func)(void*);   // Task function
    void* param;                // Task parameter
    uint8_t priority;           // Task priority (0 = highest)
    uint16_t stack_size;        // Stack size in bytes
    char name[MAX_TASK_NAME_LENGTH]; // Task name
    bool auto_start;            // Start task automatically
} TaskConfig;

/**
 * @brief Task statistics structure
 */
typedef struct {
    uint32_t execution_time;    // Total execution time in ms
    uint32_t execution_count;   // Number of times scheduled
    uint32_t stack_usage_max;   // Maximum stack usage in bytes
    float cpu_usage_percent;    // CPU usage percentage
    uint32_t last_runtime_ms;   // Last run duration in ms
    PowerMode lowest_power_mode; // Lowest power mode during execution
} TaskStats;

/**
 * @brief Create a new task
 * 
 * @param config Task configuration
 * @param handle Pointer to store the task handle
 * @return OS_SUCCESS if task was created successfully
 */
OsError os_task_create(const TaskConfig* config, TaskHandle* handle);

/**
 * @brief Delete a task
 * 
 * @param handle Task handle
 * @return OS_SUCCESS if task was deleted successfully
 */
OsError os_task_delete(TaskHandle handle);

/**
 * @brief Suspend a task
 * 
 * @param handle Task handle
 * @return OS_SUCCESS if task was suspended successfully
 */
OsError os_task_suspend(TaskHandle handle);

/**
 * @brief Resume a task
 * 
 * @param handle Task handle
 * @return OS_SUCCESS if task was resumed successfully
 */
OsError os_task_resume(TaskHandle handle);

/**
 * @brief Set task priority
 * 
 * @param handle Task handle
 * @param priority New priority
 * @return OS_SUCCESS if priority was set successfully
 */
OsError os_task_set_priority(TaskHandle handle, uint8_t priority);

/**
 * @brief Get task priority
 * 
 * @param handle Task handle
 * @param priority Pointer to store priority
 * @return OS_SUCCESS if priority was retrieved successfully
 */
OsError os_task_get_priority(TaskHandle handle, uint8_t* priority);

/**
 * @brief Get task statistics
 * 
 * @param handle Task handle
 * @param stats Pointer to store statistics
 * @return OS_SUCCESS if statistics were retrieved successfully
 */
OsError os_task_get_stats(TaskHandle handle, TaskStats* stats);

/**
 * @brief Yield execution to next task
 * 
 * @return OS_SUCCESS if yield was successful
 */
OsError os_task_yield(void);

/**
 * @brief Delay current task for specified time
 * 
 * @param ms Delay time in milliseconds
 * @return OS_SUCCESS if delay was successful
 */
OsError os_task_delay(uint32_t ms);

/**
 * @brief Get current task handle
 * 
 * @param handle Pointer to store current task handle
 * @return OS_SUCCESS if handle was retrieved successfully
 */
OsError os_task_get_current(TaskHandle* handle);

/**
 * @brief Set task scheduling policy
 * 
 * @param policy Scheduling policy
 * @return OS_SUCCESS if policy was set successfully
 */
OsError os_task_set_scheduling_policy(TaskSchedulingPolicy policy);

/**
 * @brief Set task power restrictions
 * 
 * @param handle Task handle
 * @param min_power_mode Minimum allowed power mode when task is active
 * @return OS_SUCCESS if restrictions were set successfully
 */
OsError os_task_set_power_restrictions(TaskHandle handle, PowerMode min_power_mode);

#endif /* OS_TASK_H */
