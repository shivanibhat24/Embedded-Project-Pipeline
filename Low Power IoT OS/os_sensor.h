/**
 * @file os_sensor.h
 * @brief Sensor framework for MicroOS
 */

#ifndef OS_SENSOR_H
#define OS_SENSOR_H

#include "os_core.h"

/**
 * @brief Sensor handle
 */
typedef uint8_t SensorHandle;

/**
 * @brief Sensor types
 */
typedef enum {
    SENSOR_TYPE_TEMPERATURE,
    SENSOR_TYPE_HUMIDITY,
    SENSOR_TYPE_PRESSURE,
    SENSOR_TYPE_LIGHT,
    SENSOR_TYPE_ACCELEROMETER,
    SENSOR_TYPE_GYROSCOPE,
    SENSOR_TYPE_MAGNETOMETER,
    SENSOR_TYPE_PROXIMITY,
    SENSOR_TYPE_CO2,
    SENSOR_TYPE_VOC,
    SENSOR_TYPE_MOTION,
    SENSOR_TYPE_SOUND,
    SENSOR_TYPE_CUSTOM
} SensorType;

/**
 * @brief Sensor power modes
 */
typedef enum {
    SENSOR_POWER_OFF,         // Sensor powered off
    SENSOR_POWER_LOW,         // Low power/low resolution
    SENSOR_POWER_NORMAL,      // Normal operation
    SENSOR_POWER_HIGH         // High performance/high resolution
} SensorPowerMode;

/**
 * @brief Sensor sampling modes
 */
typedef enum {
    SENSOR_SAMPLING_ONE_SHOT, // Sample once on demand
    SENSOR_SAMPLING_PERIODIC, // Sample periodically
    SENSOR_SAMPLING_TRIGGERED // Sample on trigger event
} SensorSamplingMode;

/**
 * @brief Sensor threshold direction
 */
typedef enum {
    SENSOR_THRESHOLD_ABOVE,   // Trigger when value is above threshold
    SENSOR_THRESHOLD_BELOW,   // Trigger when value is below threshold
    SENSOR_THRESHOLD_CHANGE   // Trigger when value changes by threshold amount
} SensorThresholdDir;

/**
 * @brief Sensor configuration
 */
typedef struct {
    SensorType type;                 // Sensor type
    SensorPowerMode power_mode;      // Power mode
    SensorSamplingMode sampling_mode; // Sampling mode
    uint32_t sampling_period_ms;     // Sampling period
    float threshold_value;           // Threshold value
    SensorThresholdDir threshold_dir; // Threshold direction
    bool enable_wakeup;              // Enable wake-up on threshold
    char name[16];                   // Sensor name
    void* driver_config;             // Driver-specific configuration
} SensorConfig;

/**
 * @brief Sensor reading
 */
typedef struct {
    float value;                     // Primary value
    float values[3];                 // Extended values (for multi-axis sensors)
    uint32_t timestamp;              // Timestamp of reading
    uint8_t accuracy;                // Accuracy (0-100)
    bool valid;                      // Validity flag
} SensorReading;

/**
 * @brief Sensor callback function
 */
typedef void (*SensorCallback)(SensorHandle handle, const SensorReading* reading, void* param);

/**
 * @brief Register a sensor
 * 
 * @param config Sensor configuration
 * @param callback Callback function
 * @param callback_param Parameter to pass to callback
 * @param handle Pointer to store sensor handle
 * @return OS_SUCCESS if sensor was registered successfully
 */
OsError os_sensor_register(const SensorConfig* config, SensorCallback callback, 
                          void* callback_param, SensorHandle* handle);

/**
 * @brief Unregister a sensor
 * 
 * @param handle Sensor handle
 * @return OS_SUCCESS if sensor was unregistered successfully
 */
OsError os_sensor_unregister(SensorHandle handle);

/**
 * @brief Start a sensor
 * 
 * @param handle Sensor handle
 * @return OS_SUCCESS if sensor was started successfully
 */
OsError os_sensor_start(SensorHandle handle);

/**
 * @brief Stop a sensor
 * 
 * @param handle Sensor handle
 * @return OS_SUCCESS if sensor was stopped successfully
 */
OsError os_sensor_stop(SensorHandle handle);

/**
 * @brief Read from a sensor
 * 
 * @param handle Sensor handle
 * @param reading Pointer to store reading
 * @return OS_SUCCESS if reading was successful
 */
OsError os_sensor_read(SensorHandle handle, SensorReading* reading);

/**
 * @brief Set sensor threshold
 * 
 * @param handle Sensor handle
 * @param threshold Threshold value
 * @param direction Threshold direction
 * @return OS_SUCCESS if threshold was set successfully
 */
OsError os_sensor_set_threshold(SensorHandle handle, float threshold, SensorThresholdDir direction);

/**
 * @brief Set sensor power mode
 * 
 * @param handle Sensor handle
 * @param power_mode Power mode
 * @return OS_SUCCESS if power mode was set successfully
 */
OsError os_sensor_set_power_mode(SensorHandle handle, SensorPowerMode power_mode);

/**
 * @brief Set sensor sampling parameters
 * 
 * @param handle Sensor handle
 * @param mode Sampling mode
 * @param period_ms Sampling period
 * @return OS_SUCCESS if parameters were set successfully
 */
OsError os_sensor_set_sampling(SensorHandle handle, SensorSamplingMode mode, uint32_t period_ms);

/**
 * @brief Configure sensor wake-up
 * 
 * @param handle Sensor handle
 * @param enable Enable wake-up
 * @param min_power_mode Minimum power mode to wake from
 * @return OS_SUCCESS if wake-up was configured successfully
 */
OsError os_sensor_configure_wakeup(SensorHandle handle, bool enable, PowerMode min_power_mode);

/**
 * @brief Calibrate a sensor
 * 
 * @param handle Sensor handle
 * @return OS_SUCCESS if calibration was successful
 */
OsError os_sensor_calibrate(SensorHandle handle);

/**
 * @brief Get sensor power consumption
 * 
 * @param handle Sensor handle
 * @param power_mw Pointer to store power in milliwatts
 * @return OS_SUCCESS if power was retrieved successfully
 */
OsError os_sensor_get_power_consumption(SensorHandle handle, float* power_mw);

#endif /* OS_SENSOR_H */
