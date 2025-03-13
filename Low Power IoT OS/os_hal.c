/**
 * @file os_hal.c
 * @brief Hardware Abstraction Layer implementation for MicroOS
 */
#include "os_hal.h"
#include "os_config.h"
#include <stdint.h>
#include <stdbool.h>

// Current system power mode
static PowerMode current_power_mode = POWER_MODE_ACTIVE;

// System uptime in milliseconds
static volatile uint32_t system_uptime_ms = 0;

// Flag indicating if an interrupt occurred during sleep
static volatile bool wakeup_interrupt_occurred = false;

/**
 * @brief Initialize the hardware abstraction layer
 */
void os_hal_init(void) {
    // Initialize hardware clock
    hal_clock_init();
    
    // Initialize timer for system ticks
    hal_timer_init();
    
    // Initialize GPIO
    hal_gpio_init();
    
    // Initialize power management hardware
    hal_power_init();
    
    // Set default power mode
    hal_set_power_mode(POWER_MODE_ACTIVE);
    
    // Enable interrupts
    hal_enable_interrupts();
}

/**
 * @brief Get system uptime in milliseconds
 * @return Current system uptime
 */
uint32_t os_get_uptime_ms(void) {
    return system_uptime_ms;
}

/**
 * @brief Delay for specified number of milliseconds
 * @param ms Milliseconds to delay
 */
void os_hal_delay_ms(uint32_t ms) {
    uint32_t start_time = system_uptime_ms;
    while ((system_uptime_ms - start_time) < ms) {
        // Insert CPU-specific idle instruction if available
        #if defined(PLATFORM_ARM_CORTEX_M)
        __WFI();  // Wait For Interrupt
        #elif defined(PLATFORM_RISC_V)
        __asm volatile("wfi");
        #else
        // Generic delay - less efficient
        hal_nop();
        #endif
    }
}

/**
 * @brief Set system power mode
 * @param mode Power mode to set
 */
void os_set_power_mode(PowerMode mode) {
    if (mode == current_power_mode) {
        return;
    }
    
    // Prepare for power mode transition
    hal_prepare_power_transition(current_power_mode, mode);
    
    // Update current mode
    current_power_mode = mode;
    
    // Apply new power mode
    hal_set_power_mode(mode);
}

/**
 * @brief Get current system power mode
 * @return Current power mode
 */
PowerMode os_get_power_mode(void) {
    return current_power_mode;
}

/**
 * @brief Enter idle state - called when no tasks are ready to run
 */
void os_enter_idle(void) {
    PowerMode idle_mode;
    
    // Select appropriate idle mode based on current power restrictions
    switch (current_power_mode) {
        case POWER_MODE_ACTIVE:
            idle_mode = POWER_MODE_IDLE;
            break;
        case POWER_MODE_IDLE:
        case POWER_MODE_SLEEP:
        case POWER_MODE_DEEP_SLEEP:
            idle_mode = current_power_mode;
            break;
        default:
            idle_mode = POWER_MODE_IDLE;
            break;
    }
    
    // Save current state if needed
    if (idle_mode > POWER_MODE_IDLE) {
        hal_save_context();
    }
    
    // Enter idle mode
    hal_enter_low_power_mode(idle_mode);
    
    // Restore state if needed
    if (idle_mode > POWER_MODE_IDLE) {
        hal_restore_context();
    }
}

/**
 * @brief System tick handler - called on each system timer tick
 */
void os_hal_system_tick_handler(void) {
    system_uptime_ms++;
    
    // Check if we need to wake up from sleep
    if (wakeup_interrupt_occurred) {
        wakeup_interrupt_occurred = false;
        
        // Wake up the system if needed
        if (current_power_mode > POWER_MODE_IDLE) {
            os_set_power_mode(POWER_MODE_ACTIVE);
        }
    }
}

/**
 * @brief Register interrupt handler for a specific interrupt
 * @param irq_num Interrupt number
 * @param handler Handler function
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_register_interrupt(uint8_t irq_num, InterruptHandler handler) {
    return hal_register_interrupt(irq_num, handler);
}

/**
 * @brief Enable specific interrupt
 * @param irq_num Interrupt number
 */
void os_hal_enable_interrupt(uint8_t irq_num) {
    hal_enable_interrupt(irq_num);
}

/**
 * @brief Disable specific interrupt
 * @param irq_num Interrupt number
 */
void os_hal_disable_interrupt(uint8_t irq_num) {
    hal_disable_interrupt(irq_num);
}

/**
 * @brief Called when a wakeup event occurs during sleep
 * 
 * This function is called by platform-specific interrupt handlers
 * that should wake the system from sleep modes.
 */
void os_hal_notify_wakeup(void) {
    wakeup_interrupt_occurred = true;
}

/**
 * @brief Configure GPIO pin
 * @param pin Pin number
 * @param direction Pin direction (input/output)
 * @param pull Pull-up/down configuration
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_config(uint8_t pin, GpioDirection direction, GpioPull pull) {
    return hal_gpio_config(pin, direction, pull);
}

/**
 * @brief Set GPIO pin output value
 * @param pin Pin number
 * @param value Pin value (0/1)
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_write(uint8_t pin, uint8_t value) {
    return hal_gpio_write(pin, value);
}

/**
 * @brief Read GPIO pin input value
 * @param pin Pin number
 * @param value Pointer to store read value
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_read(uint8_t pin, uint8_t* value) {
    return hal_gpio_read(pin, value);
}

/**
 * @brief Configure GPIO interrupt
 * @param pin Pin number
 * @param trigger Interrupt trigger condition
 * @param handler Interrupt handler function
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_interrupt_config(uint8_t pin, GpioInterruptTrigger trigger, GpioInterruptHandler handler) {
    return hal_gpio_interrupt_config(pin, trigger, handler);
}

/**
 * @brief Enable GPIO interrupt
 * @param pin Pin number
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_interrupt_enable(uint8_t pin) {
    return hal_gpio_interrupt_enable(pin);
}

/**
 * @brief Disable GPIO interrupt
 * @param pin Pin number
 * @return OS_ERROR_NONE on success, error code otherwise
 */
OsError os_hal_gpio_interrupt_disable(uint8_t pin) {
    return hal_gpio_interrupt_disable(pin);
}

/**
 * @brief Get battery voltage in millivolts
 * @return Battery voltage in mV
 */
uint16_t os_hal_get_battery_voltage(void) {
    return hal_get_battery_voltage();
}

/**
 * @brief Get CPU temperature in degrees Celsius
 * @return CPU temperature in °C
 */
int8_t os_hal_get_temperature(void) {
    return hal_get_temperature();
}

/* Platform-specific implementations */

#if defined(PLATFORM_ARM_CORTEX_M)
/* ARM Cortex-M specific implementations */

void hal_clock_init(void) {
    // Initialize system clock based on configuration
    #if defined(CLOCK_SPEED_HIGH)
    // Setup PLL for high-speed operation
    // Configure flash wait states
    #elif defined(CLOCK_SPEED_LOW)
    // Setup clock for low-power operation
    #else
    // Default clock configuration
    #endif
}

void hal_timer_init(void) {
    // Configure SysTick for 1ms intervals
    SysTick_Config(SystemCoreClock / 1000);
}

void hal_enable_interrupts(void) {
    __enable_irq();
}

void hal_disable_interrupts(void) {
    __disable_irq();
}

void hal_nop(void) {
    __NOP();
}

void hal_enter_low_power_mode(PowerMode mode) {
    switch (mode) {
        case POWER_MODE_IDLE:
            __WFI();
            break;
        case POWER_MODE_SLEEP:
            // Configure sleep mode
            SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk;
            __WFI();
            break;
        case POWER_MODE_DEEP_SLEEP:
            // Configure deep sleep mode
            SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
            __WFI();
            break;
        default:
            break;
    }
}

void SysTick_Handler(void) {
    os_hal_system_tick_handler();
}

#elif defined(PLATFORM_RISC_V)
/* RISC-V specific implementations */

void hal_clock_init(void) {
    // Initialize RISC-V system clock
}

void hal_timer_init(void) {
    // Configure RISC-V timer for 1ms intervals
}

void hal_enable_interrupts(void) {
    // Enable global interrupts
    __asm volatile("csrs mstatus, 8");
}

void hal_disable_interrupts(void) {
    // Disable global interrupts
    __asm volatile("csrc mstatus, 8");
}

void hal_nop(void) {
    __asm volatile("nop");
}

void hal_enter_low_power_mode(PowerMode mode) {
    switch (mode) {
        case POWER_MODE_IDLE:
            __asm volatile("wfi");
            break;
        case POWER_MODE_SLEEP:
            // Configure sleep mode registers
            __asm volatile("wfi");
            break;
        case POWER_MODE_DEEP_SLEEP:
            // Configure deep sleep mode registers
            __asm volatile("wfi");
            break;
        default:
            break;
    }
}

void timer_irq_handler(void) {
    os_hal_system_tick_handler();
}

#else
/* Generic implementations - should be overridden */

void hal_clock_init(void) {
    // Generic clock initialization
}

void hal_timer_init(void) {
    // Generic timer initialization
}

void hal_enable_interrupts(void) {
    // Generic enable interrupts
}

void hal_disable_interrupts(void) {
    // Generic disable interrupts
}

void hal_nop(void) {
    // Generic no-operation
}

void hal_enter_low_power_mode(PowerMode mode) {
    // Generic low power mode entry
}

#endif
