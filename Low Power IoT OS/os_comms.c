/**
 * @file os_comms.c
 * @brief Communication services implementation for MicroOS
 */

#include "os_comms.h"
#include "os_hal.h"
#include "os_sync.h"
#include "os_power.h"
#include <stdio.h>
#include <string.h>

// Maximum number of communication channels
#define MAX_COMM_CHANNELS 4
#define MAX_MESSAGE_SIZE 256
#define MAX_PENDING_MESSAGES 8

// Message structure
typedef struct {
    uint8_t data[MAX_MESSAGE_SIZE];
    uint16_t length;
    CommPort source_port;
} Message;

// Channel structure
typedef struct {
    bool initialized;
    CommType type;
    CommState state;
    SemaphoreHandle msg_sem;
    Message message_queue[MAX_PENDING_MESSAGES];
    uint8_t queue_head;
    uint8_t queue_tail;
    uint8_t message_count;
    CommEventCallback event_callback;
    void* user_data;
    CommConfig config;
} CommChannel;

// Communication state
static struct {
    CommChannel channels[MAX_COMM_CHANNELS];
    bool power_save_enabled;
    PowerMode active_power_mode;
} comms_state = {0};

// Forward declarations
static void comms_power_callback(PowerMode old_mode, PowerMode new_mode);

OsError os_comms_init(void) {
    printf("Initializing communication subsystem\n");
    
    // Initialize communication hardware
    HalError hal_result = hal_comms_init();
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to initialize communication hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    // Initialize channel state
    for (uint8_t i = 0; i < MAX_COMM_CHANNELS; i++) {
        comms_state.channels[i].initialized = false;
        comms_state.channels[i].state = COMM_STATE_CLOSED;
    }
    
    // Register power callback
    os_register_power_callback(comms_power_callback);
    comms_state.power_save_enabled = true;
    comms_state.active_power_mode = POWER_MODE_ACTIVE;
    
    printf("Communication subsystem initialized successfully\n");
    return OS_SUCCESS;
}

OsError os_comms_channel_create(CommChannelHandle* handle, CommType type, const CommConfig* config) {
    if (!handle || !config) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Find a free channel slot
    CommChannelHandle ch_handle = INVALID_COMM_HANDLE;
    for (uint8_t i = 0; i < MAX_COMM_CHANNELS; i++) {
        if (!comms_state.channels[i].initialized) {
            ch_handle = i;
            break;
        }
    }
    
    if (ch_handle == INVALID_COMM_HANDLE) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Create message semaphore
    SemaphoreHandle sem;
    OsError sem_result = os_semaphore_create(&sem, 0, MAX_PENDING_MESSAGES);
    if (sem_result != OS_SUCCESS) {
        return sem_result;
    }
    
    // Initialize the channel
    comms_state.channels[ch_handle].initialized = true;
    comms_state.channels[ch_handle].type = type;
    comms_state.channels[ch_handle].state = COMM_STATE_CLOSED;
    comms_state.channels[ch_handle].msg_sem = sem;
    comms_state.channels[ch_handle].queue_head = 0;
    comms_state.channels[ch_handle].queue_tail = 0;
    comms_state.channels[ch_handle].message_count = 0;
    comms_state.channels[ch_handle].event_callback = NULL;
    comms_state.channels[ch_handle].user_data = NULL;
    memcpy(&comms_state.channels[ch_handle].config, config, sizeof(CommConfig));
    
    *handle = ch_handle;
    return OS_SUCCESS;
}

OsError os_comms_channel_open(CommChannelHandle handle) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (comms_state.channels[handle].state != COMM_STATE_CLOSED) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Configure and open the channel in hardware
    HalError hal_result = hal_comms_open(handle, comms_state.channels[handle].type, 
                                         &comms_state.channels[handle].config);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to open communication channel in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    comms_state.channels[handle].state = COMM_STATE_OPEN;
    
    // If we're in a low power mode, we need to switch to active mode
    if (comms_state.active_power_mode != POWER_MODE_ACTIVE && comms_state.power_save_enabled) {
        os_set_power_mode(POWER_MODE_ACTIVE);
    }
    
    return OS_SUCCESS;
}

OsError os_comms_channel_close(CommChannelHandle handle) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (comms_state.channels[handle].state == COMM_STATE_CLOSED) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Close the channel in hardware
    HalError hal_result = hal_comms_close(handle);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to close communication channel in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    comms_state.channels[handle].state = COMM_STATE_CLOSED;
    
    // Clear any pending messages
    while (comms_state.channels[handle].message_count > 0) {
        os_semaphore_take(comms_state.channels[handle].msg_sem, 0);
        comms_state.channels[handle].queue_head = (comms_state.channels[handle].queue_head + 1) % MAX_PENDING_MESSAGES;
        comms_state.channels[handle].message_count--;
    }
    
    return OS_SUCCESS;
}

OsError os_comms_channel_send(CommChannelHandle handle, const uint8_t* data, uint16_t length, CommPort dest_port) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (comms_state.channels[handle].state != COMM_STATE_OPEN) {
        return OS_ERROR_INVALID_STATE;
    }
    
    if (!data || length == 0 || length > MAX_MESSAGE_SIZE) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Send the message through hardware
    HalError hal_result = hal_comms_send(handle, data, length, dest_port);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to send message: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    return OS_SUCCESS;
}

OsError os_comms_channel_receive(CommChannelHandle handle, uint8_t* buffer, uint16_t* length, CommPort* source_port, uint32_t timeout_ms) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (comms_state.channels[handle].state != COMM_STATE_OPEN) {
        return OS_ERROR_INVALID_STATE;
    }
    
    if (!buffer || !length || !source_port) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Wait for a message to be available
    OsError sem_result = os_semaphore_take(comms_state.channels[handle].msg_sem, timeout_ms);
    if (sem_result != OS_SUCCESS) {
        return sem_result;  // Timeout or other error
    }
    
    // Get the message from the queue
    Message* msg = &comms_state.channels[handle].message_queue[comms_state.channels[handle].queue_head];
    
    // Copy the message data
    if (*length < msg->length) {
        // Buffer too small
        *length = msg->length;
        return OS_ERROR_BUFFER_TOO_SMALL;
    }
    
    memcpy(buffer, msg->data, msg->length);
    *length = msg->length;
    *source_port = msg->source_port;
    
    // Remove the message from the queue
    comms_state.channels[handle].queue_head = (comms_state.channels[handle].queue_head + 1) % MAX_PENDING_MESSAGES;
    comms_state.channels[handle].message_count--;
    
    return OS_SUCCESS;
}

OsError os_comms_channel_set_callback(CommChannelHandle handle, CommEventCallback callback, void* user_data) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    comms_state.channels[handle].event_callback = callback;
    comms_state.channels[handle].user_data = user_data;
    
    return OS_SUCCESS;
}

OsError os_comms_enable_power_save(bool enable) {
    comms_state.power_save_enabled = enable;
    return OS_SUCCESS;
}

// Internal function to handle received messages from hardware
void os_comms_handle_message(CommChannelHandle handle, const uint8_t* data, uint16_t length, CommPort source_port) {
    if (handle >= MAX_COMM_CHANNELS || !comms_state.channels[handle].initialized) {
        return;
    }
    
    // Check if there's space in the queue
    if (comms_state.channels[handle].message_count >= MAX_PENDING_MESSAGES) {
        printf("Message queue full, dropping message\n");
        return;
    }
    
    // Add the message to the queue
    Message* msg = &comms_state.channels[handle].message_queue[comms_state.channels[handle].queue_tail];
    
    if (length > MAX_MESSAGE_SIZE) {
        length = MAX_MESSAGE_SIZE;  // Truncate if too large
    }
    
    memcpy(msg->data, data, length);
    msg->length = length;
    msg->source_port = source_port;
    
    comms_state.channels[handle].queue_tail = (comms_state.channels[handle].queue_tail + 1) % MAX_PENDING_MESSAGES;
    comms_state.channels[handle].message_count++;
    
    // Signal that a message is available
    os_semaphore_give(comms_state.channels[handle].msg_sem);
    
    // Call the event callback if registered
    if (comms_state.channels[handle].event_callback) {
        comms_state.channels[handle].event_callback(handle, COMM_EVENT_DATA_RECEIVED, 
                                                 comms_state.channels[handle].user_data);
    }
}

// Power management callback
static void comms_power_callback(PowerMode old_mode, PowerMode new_mode) {
    comms_state.active_power_mode = new_mode;
    
    // If entering a low power mode, close all channels
    if (new_mode != POWER_MODE_ACTIVE && comms_state.power_save_enabled) {
        for (uint8_t i = 0; i < MAX_COMM_CHANNELS; i++) {
            if (comms_state.channels[i].initialized && comms_state.channels[i].state == COMM_STATE_OPEN) {
                // Notify the event callback
                if (comms_state.channels[i].event_callback) {
                    comms_state.channels[i].event_callback(i, COMM_EVENT_SLEEP, 
                                                       comms_state.channels[i].user_data);
                }
                
                // Close the channel in hardware
                hal_comms_close(i);
                comms_state.channels[i].state = COMM_STATE_SUSPENDED;
            }
        }
    }
    // If exiting a low power mode, reopen suspended channels
    else if (new_mode == POWER_MODE_ACTIVE && old_mode != POWER_MODE_ACTIVE && comms_state.power_save_enabled) {
        for (uint8_t i = 0; i < MAX_COMM_CHANNELS; i++) {
            if (comms_state.channels[i].initialized && comms_state.channels[i].state == COMM_STATE_SUSPENDED) {
                // Reopen the channel in hardware
                HalError hal_result = hal_comms_open(i, comms_state.channels[i].type, 
                                                     &comms_state.channels[i].config);
                if (hal_result == HAL_SUCCESS) {
                    comms_state.channels[i].state = COMM_STATE_OPEN;
                    
                    // Notify the event callback
                    if (comms_state.channels[i].event_callback) {
                        comms_state.channels[i].event_callback(i, COMM_EVENT_WAKE, 
                                                           comms_state.channels[i].user_data);
                    }
                }
            }
        }
    }
}
