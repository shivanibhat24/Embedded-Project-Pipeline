/**
 * @file os_sync.h
 * @brief Synchronization primitives for MicroOS
 */

#ifndef OS_SYNC_H
#define OS_SYNC_H

#include "os_core.h"

/**
 * @brief Mutex handle
 */
typedef uint8_t MutexHandle;

/**
 * @brief Semaphore handle
 */
typedef uint8_t SemaphoreHandle;

/**
 * @brief Event group handle
 */
typedef uint8_t EventGroupHandle;

/**
 * @brief Queue handle
 */
typedef uint8_t QueueHandle;

/**
 * @brief Create a mutex
 * 
 * @param name Mutex name
 * @param handle Pointer to store mutex handle
 * @return OS_SUCCESS if mutex was created successfully
 */
OsError os_mutex_create(const char* name, MutexHandle* handle);

/**
 * @brief Delete a mutex
 * 
 * @param handle Mutex handle
 * @return OS_SUCCESS if mutex was deleted successfully
 */
OsError os_mutex_delete(MutexHandle handle);

/**
 * @brief Take a mutex
 * 
 * @param handle Mutex handle
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if mutex was taken successfully
 */
OsError os_mutex_take(MutexHandle handle, uint32_t timeout_ms);

/**
 * @brief Give a mutex
 * 
 * @param handle Mutex handle
 * @return OS_SUCCESS if mutex was given successfully
 */
OsError os_mutex_give(MutexHandle handle);

/**
 * @brief Create a semaphore
 * 
 * @param name Semaphore name
 * @param initial_count Initial count
 * @param max_count Maximum count
 * @param handle Pointer to store semaphore handle
 * @return OS_SUCCESS if semaphore was created successfully
 */
OsError os_semaphore_create(const char* name, uint32_t initial_count, uint32_t max_count, SemaphoreHandle* handle);

/**
 * @brief Delete a semaphore
 * 
 * @param handle Semaphore handle
 * @return OS_SUCCESS if semaphore was deleted successfully
 */
OsError os_semaphore_delete(SemaphoreHandle handle);

/**
 * @brief Take a semaphore
 * 
 * @param handle Semaphore handle
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if semaphore was taken successfully
 */
OsError os_semaphore_take(SemaphoreHandle handle, uint32_t timeout_ms);

/**
 * @brief Give a semaphore
 * 
 * @param handle Semaphore handle
 * @return OS_SUCCESS if semaphore was given successfully
 */
OsError os_semaphore_give(SemaphoreHandle handle);

/**
 * @brief Create an event group
 * 
 * @param name Event group name
 * @param handle Pointer to store event group handle
 * @return OS_SUCCESS if event group was created successfully
 */
OsError os_event_group_create(const char* name, EventGroupHandle* handle);

/**
 * @brief Delete an event group
 * 
 * @param handle Event group handle
 * @return OS_SUCCESS if event group was deleted successfully
 */
OsError os_event_group_delete(EventGroupHandle handle);

/**
 * @brief Set bits in an event group
 * 
 * @param handle Event group handle
 * @param bits_to_set Bits to set
 * @param bits_before_set Pointer to store bits before set
 * @return OS_SUCCESS if bits were set successfully
 */
OsError os_event_group_set_bits(EventGroupHandle handle, uint32_t bits_to_set, uint32_t* bits_before_set);

/**
 * @brief Clear bits in an event group
 * 
 * @param handle Event group handle
 * @param bits_to_clear Bits to clear
 * @param bits_before_clear Pointer to store bits before clear
 * @return OS_SUCCESS if bits were cleared successfully
 */
OsError os_event_group_clear_bits(EventGroupHandle handle, uint32_t bits_to_clear, uint32_t* bits_before_clear);

/**
 * @brief Wait for bits in an event group
 * 
 * @param handle Event group handle
 * @param bits_to_wait_for Bits to wait for
 * @param clear_on_exit Clear bits on exit
 * @param wait_for_all Wait for all bits
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @param bits_on_entry Pointer to store bits on entry
 * @return OS_SUCCESS if bits were waited for successfully
 */
OsError os_event_group_wait_bits(EventGroupHandle handle, uint32_t bits_to_wait_for, bool clear_on_exit, bool wait_for_all, uint32_t timeout_ms, uint32_t* bits_on_entry);

/**
 * @brief Create a queue
 * 
 * @param name Queue name
 * @param item_size Item size in bytes
 * @param max_items Maximum number of items
 * @param handle Pointer to store queue handle
 * @return OS_SUCCESS if queue was created successfully
 */
OsError os_queue_create(const char* name, uint32_t item_size, uint32_t max_items, QueueHandle* handle);

/**
 * @brief Delete a queue
 * 
 * @param handle Queue handle
 * @return OS_SUCCESS if queue was deleted successfully
 */
OsError os_queue_delete(QueueHandle handle);

/**
 * @brief Send an item to a queue
 * 
 * @param handle Queue handle
 * @param item Pointer to item to send
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if item was sent successfully
 */
OsError os_queue_send(QueueHandle handle, const void* item, uint32_t timeout_ms);

/**
 * @brief Send an item to the front of a queue
 * 
 * @param handle Queue handle
 * @param item Pointer to item to send
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if item was sent successfully
 */
OsError os_queue_send_front(QueueHandle handle, const void* item, uint32_t timeout_ms);

/**
 * @brief Receive an item from a queue
 * 
 * @param handle Queue handle
 * @param item Buffer to store received item
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if item was received successfully
 */
OsError os_queue_receive(QueueHandle handle, void* item, uint32_t timeout_ms);

/**
 * @brief Peek an item from a queue without removing it
 * 
 * @param handle Queue handle
 * @param item Buffer to store peeked item
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return OS_SUCCESS if item was peeked successfully
 */
OsError os_queue_peek(QueueHandle handle, void* item, uint32_t timeout_ms);

/**
 * @brief Get the number of items in a queue
 * 
 * @param handle Queue handle
 * @param count Pointer to store item count
 * @return OS_SUCCESS if count was retrieved successfully
 */
OsError os_queue_get_count(QueueHandle handle, uint32_t* count);

/**
 * @brief Reset a queue
 * 
 * @param handle Queue handle
 * @return OS_SUCCESS if queue was reset successfully
 */
OsError os_queue_reset(QueueHandle handle);

/**
 * @brief Check if a queue is full
 * 
 * @param handle Queue handle
 * @param is_full Pointer to store result
 * @return OS_SUCCESS if check was successful
 */
OsError os_queue_is_full(QueueHandle handle, bool* is_full);

/**
 * @brief Check if a queue is empty
 * 
 * @param handle Queue handle
 * @param is_empty Pointer to store result
 * @return OS_SUCCESS if check was successful
 */
OsError os_queue_is_empty(QueueHandle handle, bool* is_empty);

#endif /* OS_SYNC_H */
