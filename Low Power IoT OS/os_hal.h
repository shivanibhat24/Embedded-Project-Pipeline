/**
 * @file os_hal.h
 * @brief Hardware Abstraction Layer for MicroOS
 */

#ifndef OS_HAL_H
#define OS_HAL_H

#include "os_core.h"

/**
 * @brief HAL error codes
 */
typedef enum {
    HAL_SUCCESS = 0,
    HAL_ERROR_GENERIC,
    HAL_ERROR_TIMEOUT,
    HAL_ERROR_BUSY,
    HAL_ERROR_NOT_SUPPORTED,
    HAL_ERROR_INVALID_PARAMETER
} HalError;

/**
 * @brief GPIO pin configuration
 */
typedef enum {
    GPIO_MODE_INPUT,         // Input pin
    GPIO_MODE_OUTPUT,        // Output pin
    GPIO_MODE_INPUT_PULLUP,  // Input with pull-up
    GPIO_MODE_INPUT_PULLDOWN // Input with pull-down
} GpioMode;

/**
 * @brief GPIO interrupt trigger
 */
typedef enum {
    GPIO_TRIGGER_NONE,       // No interrupt
    GPIO_TRIGGER_RISING,     // Rising edge trigger
    GPIO_TRIGGER_FALLING,    // Falling edge trigger
    GPIO_TRIGGER_BOTH        // Both edges trigger
} GpioTrigger;

/**
 * @brief UART configuration
 */
typedef struct {
    uint32_t baud_rate;      // Baud rate (e.g., 9600, 115200)
    uint8_t data_bits;       // Data bits (7, 8, 9)
    uint8_t stop_bits;       // Stop bits (1, 2)
    bool parity_enable;      // Enable parity
    bool parity_odd;         // Odd parity (if enabled)
    bool flow_control;       // Hardware flow control
    uint8_t rx_buffer_size;  // RX buffer size in bytes
    uint8_t tx_buffer_size;  // TX buffer size in bytes
    bool low_power_mode;     // Enable low-power operation
} UartConfig;

/**
 * @brief SPI configuration
 */
typedef struct {
    uint32_t clock_hz;       // Clock frequency in Hz
    bool cpol;               // Clock polarity
    bool cpha;               // Clock phase
    uint8_t data_bits;       // Data bits (8, 16)
    bool msb_first;          // MSB first (true) or LSB first (false)
    uint8_t cs_pin;          // Chip select pin
    bool low_power_mode;     // Enable low-power operation
} SpiConfig;

/**
 * @brief I2C configuration
 */
typedef struct {
    uint32_t clock_hz;       // Clock frequency in Hz
    bool addressing_10bit;   // 10-bit addressing mode
    uint8_t retry_count;     // Number of retries
    bool low_power_mode;     // Enable low-power operation
} I2cConfig;

/**
 * @brief ADC configuration
 */
typedef struct {
    uint8_t resolution;      // Resolution in bits (8, 10, 12, 16)
    uint32_t sample_rate;    // Sample rate in Hz
    bool continuous;         // Continuous sampling
    uint8_t reference;       // Reference voltage source
    bool low_power_mode;     // Enable low-power operation
} AdcConfig;

/**
 * @brief GPIO callback function
 */
typedef void (*GpioCallback)(uint8_t pin);

/**
 * @brief UART callback function
 */
typedef void (*UartCallback)(uint8_t uart_id, uint8_t* data, uint16_t length);

/**
 * @brief System initialization
 * 
 * @return HAL_SUCCESS if initialization was successful
 */
HalError hal_init(void);

/**
 * @brief Enter low-power mode
 * 
 * @param mode Power mode to enter
 * @return HAL_SUCCESS if mode was entered successfully
 */
HalError hal_enter_power_mode(PowerMode mode);

/**
 * @brief Configure system clock
 * 
 * @param freq_hz Frequency in Hz
 * @return HAL_SUCCESS if clock was configured successfully
 */
HalError hal_configure_system_clock(uint32_t freq_hz);

/**
 * @brief Get system time
 * 
 * @return System time in milliseconds
 */
uint32_t hal_get_system_time(void);

/**
 * @brief Configure GPIO pin
 * 
 * @param pin Pin number
 * @param mode GPIO mode
 * @param trigger Interrupt trigger
 * @param callback Callback function (NULL for no callback)
 * @return HAL_SUCCESS if pin was configured successfully
 */
HalError hal_gpio_configure(uint8_t pin, GpioMode mode, GpioTrigger trigger, GpioCallback callback);

/**
 * @brief Set GPIO pin value
 * 
 * @param pin Pin number
 * @param value Pin value (0 or 1)
 * @return HAL_SUCCESS if value was set successfully
 */
HalError hal_gpio_write(uint8_t pin, uint8_t value);

/**
 * @brief Read GPIO pin value
 * 
 * @param pin Pin number
 * @param value Pointer to store pin value
 * @return HAL_SUCCESS if value was read successfully
 */
HalError hal_gpio_read(uint8_t pin, uint8_t* value);

/**
 * @brief Configure UART
 * 
 * @param uart_id UART ID
 * @param config UART configuration
 * @param rx_callback Receive callback function (NULL for no callback)
 * @return HAL_SUCCESS if UART was configured successfully
 */
HalError hal_uart_configure(uint8_t uart_id, const UartConfig* config, UartCallback rx_callback);

/**
 * @brief Send data over UART
 * 
 * @param uart_id UART ID
 * @param data Data to send
 * @param length Data length
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return HAL_SUCCESS if data was sent successfully
 */
HalError hal_uart_send(uint8_t uart_id, const uint8_t* data, uint16_t length, uint32_t timeout_ms);

/**
 * @brief Receive data from UART
 * 
 * @param uart_id UART ID
 * @param data Buffer to store received data
 * @param length Maximum data length
 * @param received Pointer to store actual received length
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return HAL_SUCCESS if data was received successfully
 */
HalError hal_uart_receive(uint8_t uart_id, uint8_t* data, uint16_t length, uint16_t* received, uint32_t timeout_ms);

/**
 * @brief Configure SPI
 * 
 * @param spi_id SPI ID
 * @param config SPI configuration
 * @return HAL_SUCCESS if SPI was configured successfully
 */
HalError hal_spi_configure(uint8_t spi_id, const SpiConfig* config);

/**
 * @brief Transfer data over SPI
 * 
 * @param spi_id SPI ID
 * @param tx_data Data to send (NULL for receive-only)
 * @param rx_data Buffer to store received data (NULL for send-only)
 * @param length Data length
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return HAL_SUCCESS if data was transferred successfully
 */
HalError hal_spi_transfer(uint8_t spi_id, const uint8_t* tx_data, uint8_t* rx_data, uint16_t length, uint32_t timeout_ms);

/**
 * @brief Configure I2C
 * 
 * @param i2c_id I2C ID
 * @param config I2C configuration
 * @return HAL_SUCCESS if I2C was configured successfully
 */
HalError hal_i2c_configure(uint8_t i2c_id, const I2cConfig* config);

/**
 * @brief Write data to I2C device
 * 
 * @param i2c_id I2C ID
 * @param device_addr Device address
 * @param data Data to write
 * @param length Data length
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return HAL_SUCCESS if data was written successfully
 */
HalError hal_i2c_write(uint8_t i2c_id, uint16_t device_addr, const uint8_t* data, uint16_t length, uint32_t timeout_ms);

/**
 * @brief Read data from I2C device
 * 
 * @param i2c_id I2C ID
 * @param device_addr Device address
 * @param data Buffer to store read data
 * @param length Data length
 * @param timeout_ms Timeout in milliseconds (0 for no timeout)
 * @return HAL_SUCCESS if data was read successfully
 */
HalError hal_i2c_read(uint8_t i2c_id, uint16_t device_addr, uint8_t* data, uint16_t length, uint32_t timeout_ms);

/**
 * @brief Configure ADC
 * 
 * @param adc_id ADC ID
 * @param config ADC configuration
 * @return HAL_SUCCESS if ADC was configured successfully
 */
HalError hal_adc_configure(uint8_t adc_id, const AdcConfig* config);

/**
 * @brief Read ADC value
 * 
 * @param adc_id ADC ID
 * @param channel ADC channel
 * @param value Pointer to store ADC value
 * @return HAL_SUCCESS if value was read successfully
 */
HalError hal_adc_read(uint8_t adc_id, uint8_t channel, uint16_t* value);

/**
 * @brief Configure RTC
 * 
 * @param wakeup_ms Wake-up time in milliseconds (0 to disable)
 * @return HAL_SUCCESS if RTC was configured successfully
 */
HalError hal_rtc_configure(uint32_t wakeup_ms);

/**
 * @brief Set RTC alarm
 * 
 * @param seconds Seconds from now
 * @return HAL_SUCCESS if alarm was set successfully
 */
HalError hal_rtc_set_alarm(uint32_t seconds);

/**
 * @brief Get RTC time
 * 
 * @param timestamp Pointer to store timestamp
 * @return HAL_SUCCESS if time was retrieved successfully
 */
HalError hal_rtc_get_time(uint32_t* timestamp);

/**
 * @brief Power profiling
 * 
 * @param power_mw Pointer to store current power consumption in mW
 * @return HAL_SUCCESS if power was measured successfully
 */
HalError hal_power_measure(float* power_mw);

#endif /* OS_HAL_H */
