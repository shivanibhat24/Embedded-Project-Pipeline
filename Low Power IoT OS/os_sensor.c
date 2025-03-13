/**
 * @file os_sensor.c
 * @brief Sensor management implementation for MicroOS
 */

#include "os_sensor.h"
#include "os_hal.h"
#include "os_timer.h"
#include "os_power.h"
#include <stdio.h>
#include <string.h>

// Maximum number of managed sensors
#define MAX_SENSORS 8

// Sensor structure
typedef struct {
    bool initialized;
    SensorType type;
    SensorConfig config;
    SensorState state;
    SensorDataCallback data_callback;
    void* user_data;
    TimerHandle sample_timer;
    uint32_t last_sample_time;
    uint32_t sample_interval_ms;
    uint32_t power_consumption_uA;
} Sensor;

// Sensor management state
static struct {
    Sensor sensors[MAX_SENSORS];
    bool power_save_enabled;
    PowerMode active_power_mode;
} sensor_state = {0};

// Forward declarations
static void sensor_power_callback(PowerMode old_mode, PowerMode new_mode);
static void sensor_timer_callback(TimerHandle timer, void* user_data);

OsError os_sensor_init(void) {
    printf("Initializing sensor subsystem\n");
    
    // Initialize sensor hardware
    HalError hal_result = hal_sensor_init();
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to initialize sensor hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    // Initialize sensor state
    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        sensor_state.sensors[i].initialized = false;
        sensor_state.sensors[i].state = SENSOR_STATE_IDLE;
    }
    
    // Register power callback
    os_register_power_callback(sensor_power_callback);
    sensor_state.power_save_enabled = true;
    sensor_state.active_power_mode = POWER_MODE_ACTIVE;
    
    printf("Sensor subsystem initialized successfully\n");
    return OS_SUCCESS;
}

OsError os_sensor_register(SensorHandle* handle, SensorType type, const SensorConfig* config) {
    if (!handle || !config) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Find a free sensor slot
    SensorHandle sensor_handle = INVALID_SENSOR_HANDLE;
    for (uint8_t i = 0; i < MAX_SENSORS; i++) {
        if (!sensor_state.sensors[i].initialized) {
            sensor_handle = i;
            break;
        }
    }
    
    if (sensor_handle == INVALID_SENSOR_HANDLE) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    // Register the sensor with hardware
    HalError hal_result = hal_sensor_register(sensor_handle, type, config);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to register sensor in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    // Initialize the sensor
    sensor_state.sensors[sensor_handle].initialized = true;
    sensor_state.sensors[sensor_handle].type = type;
    memcpy(&sensor_state.sensors[sensor_handle].config, config, sizeof(SensorConfig));
    sensor_state.sensors[sensor_handle].state = SENSOR_STATE_IDLE;
    sensor_state.sensors[sensor_handle].data_callback = NULL;
    sensor_state.sensors[sensor_handle].user_data = NULL;
    sensor_state.sensors[sensor_handle].sample_timer = INVALID_TIMER_HANDLE;
    sensor_state.sensors[sensor_handle].last_sample_time = 0;
    sensor_state.sensors[sensor_handle].sample_interval_ms = 0;
    
    // Get power consumption information
    hal_sensor_get_power_info(sensor_handle, &sensor_state.sensors[sensor_handle].power_consumption_uA);
    
    *handle = sensor_handle;
    return OS_SUCCESS;
}

OsError os_sensor_enable(SensorHandle handle) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sensor_state.sensors[handle].state != SENSOR_STATE_IDLE) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Enable the sensor in hardware
    HalError hal_result = hal_sensor_enable(handle);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to enable sensor in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    sensor_state.sensors[handle].state = SENSOR_STATE_ENABLED;
    
    return OS_SUCCESS;
}

OsError os_sensor_disable(SensorHandle handle) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sensor_state.sensors[handle].state == SENSOR_STATE_IDLE) {
        return OS_ERROR_INVALID_STATE;
    }
    
    // Stop any ongoing sampling
    if (sensor_state.sensors[handle].sample_timer != INVALID_TIMER_HANDLE) {
        os_timer_stop(sensor_state.sensors[handle].sample_timer);
        os_timer_delete(sensor_state.sensors[handle].sample_timer);
        sensor_state.sensors[handle].sample_timer = INVALID_TIMER_HANDLE;
    }
    
    // Disable the sensor in hardware
    HalError hal_result = hal_sensor_disable(handle);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to disable sensor in hardware: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    sensor_state.sensors[handle].state = SENSOR_STATE_IDLE;
    
    return OS_SUCCESS;
}

OsError os_sensor_read(SensorHandle handle, SensorData* data) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sensor_state.sensors[handle].state != SENSOR_STATE_ENABLED) {
        return OS_ERROR_INVALID_STATE;
    }
    
    if (!data) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Read sensor data from hardware
    HalError hal_result = hal_sensor_read(handle, data);
    if (hal_result != HAL_SUCCESS) {
        printf("Failed to read sensor data: %d\n", hal_result);
        return OS_ERROR_HARDWARE;
    }
    
    // Update last sample time
    sensor_state.sensors[handle].last_sample_time = os_get_uptime_ms();
    
    return OS_SUCCESS;
}

OsError os_sensor_start_sampling(SensorHandle handle, uint32_t interval_ms, SensorDataCallback callback, void* user_data) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sensor_state.sensors[handle].state != SENSOR_STATE_ENABLED) {
        return OS_ERROR_INVALID_STATE;
    }
    
    if (!callback || interval_ms == 0) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Stop previous sampling if active
    if (sensor_state.sensors[handle].sample_timer != INVALID_TIMER_HANDLE) {
        os_timer_stop(sensor_state.sensors[handle].sample_timer);
        os_timer_delete(sensor_state.sensors[handle].sample_timer);
    }
    
    // Create and start sampling timer
    TimerHandle timer = os_timer_create(interval_ms, true, sensor_timer_callback, (void*)(uintptr_t)handle);
    if (timer == INVALID_TIMER_HANDLE) {
        return OS_ERROR_RESOURCE_UNAVAILABLE;
    }
    
    sensor_state.sensors[handle].sample_timer = timer;
    sensor_state.sensors[handle].data_callback = callback;
    sensor_state.sensors[handle].user_data = user_data;
    sensor_state.sensors[handle].sample_interval_ms = interval_ms;
    
    OsError timer_result = os_timer_start(timer);
    if (timer_result != OS_SUCCESS) {
        os_timer_delete(timer);
        sensor_state.sensors[handle].sample_timer = INVALID_TIMER_HANDLE;
        return timer_result;
    }
    
    // Take first sample immediately
    SensorData data;
    OsError read_result = os_sensor_read(handle, &data);
    if (read_result == OS_SUCCESS && callback) {
        callback(handle, &data, user_data);
    }
    
    return OS_SUCCESS;
}

OsError os_sensor_stop_sampling(SensorHandle handle) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (sensor_state.sensors[handle].sample_timer == INVALID_TIMER_HANDLE) {
        return OS_ERROR_INVALID_STATE;  // Not sampling
    }
    
    // Stop and delete the timer
    os_timer_stop(sensor_state.sensors[handle].sample_timer);
    os_timer_delete(sensor_state.sensors[handle].sample_timer);
    
    sensor_state.sensors[handle].sample_timer = INVALID_TIMER_HANDLE;
    sensor_state.sensors[handle].data_callback = NULL;
    sensor_state.sensors[handle].sample_interval_ms = 0;
    
    return OS_SUCCESS;
}

OsError os_sensor_get_stats(SensorHandle handle, SensorStats* stats) {
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return OS_ERROR_INVALID_HANDLE;
    }
    
    if (!stats) {
        return OS_ERROR_INVALID_PARAMETER;
    }
    
    // Get sensor statistics from hardware
    HalError hal_result = hal_sensor_get_stats(handle, stats);
    if (hal_result != HAL_SUCCESS) {
        return OS_ERROR_HARDWARE;
    }
    
    // Add software statistics
    stats->last_sample_time = sensor_state.sensors[handle].last_sample_time;
    stats->sampling_interval_ms = sensor_state.sensors[handle].sample_interval_ms;
    stats->power_consumption_uA = sensor_state.sensors[handle].power_consumption_uA;
    
    return OS_SUCCESS;
}

OsError os_sensor_enable_power_save(bool enable) {
    sensor_state.power_save_enabled = enable;
    return OS_SUCCESS;
}

// Timer callback for sensor sampling
static void sensor_timer_callback(TimerHandle timer, void* user_data) {
    SensorHandle handle = (SensorHandle)(uintptr_t)user_data;
    
    if (handle >= MAX_SENSORS || !sensor_state.sensors[handle].initialized) {
        return;
    }
    
    // Read sensor data
    SensorData data;
    OsError read_result = os_sensor_read(handle, &data);
    
    // Call the user callback if read was successful
    if (read_result == OS_SUCCESS && sensor_state.sensors[handle].data_callback) {
        sensor_state.sensors[handle].data_callback(handle, &data, sensor_state.sensors[handle].user_data);
    }
}

// Power management callback
static void sensor_power_callback(PowerMode old_mode, PowerMode new_mode) {
    sensor_state.active_power_mode = new_mode;
    
    // If entering a low power mode, disable sensors
    if (new_mode != POWER_MODE_ACTIVE && sensor_state.power_save_enabled) {
        for (uint8_t i = 0; i < MAX_SENSORS; i++) {
            if (sensor_state.sensors[i].initialized && sensor_state.sensors[i].state == SENSOR_STATE_ENABLED) {
                // Stop any ongoing sampling
                if (sensor_state.sensors[i].sample_timer != INVALID_TIMER_HANDLE) {
                    os_timer_stop(sensor_state.sensors[i].sample_timer);
                }
                
                // Disable sensor hardware
                hal_sensor_disable(i);
                sensor_state.sensors[i].state = SENSOR_STATE_SUSPENDED;
            }
        }
    }
    // If exiting a low power mode, re-enable suspended sensors
    else if (new_mode == POWER_MODE_ACTIVE && old_mode != POWER_MODE_ACTIVE && sensor_state.power_save_enabled) {
        for (uint8_t i = 0; i < MAX_SENSORS; i++) {
            if (sensor_state.sensors[i].initialized && sensor_state.sensors[i].state == SENSOR_STATE_SUSPENDED) {
                // Re-enable sensor hardware
                HalError hal_result = hal_sensor_enable(i);
                if (hal_result == HAL_SUCCESS) {
                    sensor_state.sensors[i].state = SENSOR_STATE_ENABLED;
                    
                    // Restart sampling if it was active
                    if (sensor_state.sensors[i].sample_timer != INVALID_TIMER_HANDLE) {
                        os_timer_start(sensor_state.sensors[i].sample_timer);
                    }
                }
            }
        }
    }
}
