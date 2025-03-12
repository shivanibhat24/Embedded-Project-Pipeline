#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "hal.h"  // Hardware Abstraction Layer
#include "scheduler.h"
#include "power_mgmt.h"

#define MAX_TASKS 5

// Task Control Block (TCB)
typedef struct {
    void (*task_func)(void);
    uint8_t priority;
    bool is_running;
    uint32_t execution_time;  // Track execution time
} Task;

Task task_list[MAX_TASKS];
uint8_t task_count = 0;
uint32_t dynamic_wakeup_time = 10; // Default wake-up time

// Function to create a task
void create_task(void (*task_func)(void), uint8_t priority) {
    if (task_count < MAX_TASKS) {
        task_list[task_count].task_func = task_func;
        task_list[task_count].priority = priority;
        task_list[task_count].is_running = true;
        task_list[task_count].execution_time = 0;
        task_count++;
    }
}

// Function to remove a task
void remove_task(uint8_t index) {
    if (index < task_count) {
        for (uint8_t i = index; i < task_count - 1; i++) {
            task_list[i] = task_list[i + 1];
        }
        task_count--;
    }
}

// Function to pause a task
void pause_task(uint8_t index) {
    if (index < task_count) {
        task_list[index].is_running = false;
    }
}

// Function to resume a task
void resume_task(uint8_t index) {
    if (index < task_count) {
        task_list[index].is_running = true;
    }
}

// Priority-Based Task Scheduler with Cooperative Execution
void scheduler_run(void) {
    while (1) {
        bool all_tasks_idle = true;
        uint8_t highest_priority = 255;
        int8_t selected_task = -1;
        
        // Select the highest priority runnable task
        for (uint8_t i = 0; i < task_count; i++) {
            if (task_list[i].is_running && task_list[i].priority < highest_priority) {
                highest_priority = task_list[i].priority;
                selected_task = i;
            }
        }
        
        if (selected_task != -1) {
            uint32_t start_time = hal_get_system_time();
            task_list[selected_task].task_func();
            uint32_t end_time = hal_get_system_time();
            task_list[selected_task].execution_time += (end_time - start_time);
            all_tasks_idle = false;
        }
        
        if (all_tasks_idle) {
            enter_low_power_mode();  // Put CPU in low-power sleep
            return; // Exit loop to allow external wake-up handling
        }
    }
}

// Low-power sleep mode
void enter_low_power_mode(void) {
    printf("Entering low-power mode...\n");
    hal_enter_low_power();  // Use HAL-defined low-power mode
}

// Power Profiling
void power_profiling(void) {
    printf("Measuring power consumption...\n");
    printf("Current power consumption: 5mW\n");
}

// Hardware Abstraction Layer (HAL) Implementation
void hal_init(void) {
    printf("HAL initialized\n");
}

void hal_enter_low_power(void) {
    printf("HAL: Entering deep sleep mode...\n");
    __asm__("WFI");  // Wait for interrupt (ARM-based sleep mode)
}

void hal_gpio_init(void) {
    printf("GPIO initialized for low power\n");
}

void hal_uart_init(void) {
    printf("UART initialized in low-power mode\n");
}

void hal_i2c_init(void) {
    printf("I2C initialized in low-power mode\n");
}

void hal_spi_init(void) {
    printf("SPI initialized in low-power mode\n");
}

uint32_t hal_get_system_time(void) {
    static uint32_t fake_time = 0;
    return fake_time += 10;
}

// RTC Wake-up Implementation
void hal_rtc_init(void) {
    printf("RTC initialized for periodic wake-up\n");
}

void hal_rtc_set_wakeup(uint32_t seconds) {
    printf("RTC wake-up set for %u seconds\n", seconds);
    dynamic_wakeup_time = seconds;
}

void hal_rtc_set_dynamic_wakeup(uint32_t seconds) {
    printf("Setting dynamic RTC wake-up time to %u seconds\n", seconds);
    hal_rtc_set_wakeup(seconds);
}

// Wake-up Sources Implementation
void hal_configure_wakeup_sources(void) {
    printf("Configuring wake-up sources...\n");
    hal_rtc_init();
    hal_rtc_set_wakeup(dynamic_wakeup_time);
    printf("RTC wake-up enabled with dynamic timing\n");
    printf("GPIO interrupt wake-up enabled\n");
    printf("UART wake-up enabled\n");
    printf("Sensor-based wake-up enabled\n");
}

// Example Tasks
void task1(void) {
    printf("Task 1 running\n");
    hal_rtc_set_dynamic_wakeup(5);
}

void task2(void) {
    printf("Task 2 running\n");
}

void task3(void) {
    printf("Task 3: Power profiling\n");
    power_profiling();
}

int main(void) {
    hal_init();
    hal_gpio_init();
    hal_uart_init();
    hal_i2c_init();
    hal_spi_init();
    hal_configure_wakeup_sources();
    
    create_task(task1, 1);
    create_task(task2, 2);
    create_task(task3, 3);
    scheduler_run();
    return 0;
}
