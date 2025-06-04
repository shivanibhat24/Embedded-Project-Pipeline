#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>

// Hardware abstraction layer headers (platform specific)
// #include "gpio.h"
// #include "adc.h"
// #include "gsm_module.h"

// Rice types and their cooking parameters
typedef enum {
    BASMATI_RICE = 0,
    KOLAM_RICE,
    JASMINE_RICE,
    SONA_MASOORI,
    PONNI_RICE,
    BROWN_RICE,
    TOTAL_RICE_TYPES
} rice_type_t;

typedef struct {
    char name[20];
    int cooking_time_minutes;
    float water_ratio;          // Water to rice ratio
    int temp_threshold;         // Temperature threshold for done detection
    int steam_time_seconds;     // Additional steaming time after cooking
} rice_profile_t;

typedef struct {
    char phone_number[15];
    bool sms_enabled;
    int temperature_sensor_pin;
    int moisture_sensor_pin;
    int buzzer_pin;
    int led_pin;
} device_config_t;

typedef enum {
    STATE_IDLE = 0,
    STATE_COOKING,
    STATE_STEAMING,
    STATE_DONE,
    STATE_ERROR
} cooking_state_t;

// Global variables
rice_profile_t rice_profiles[TOTAL_RICE_TYPES] = {
    {"Basmati", 18, 1.5, 85, 300},        // Basmati: 18 min, 1.5:1 ratio, 85°C, 5 min steam
    {"Kolam", 25, 2.0, 88, 600},          // Kolam: 25 min, 2:1 ratio, 88°C, 10 min steam
    {"Jasmine", 15, 1.3, 82, 240},        // Jasmine: 15 min, 1.3:1 ratio, 82°C, 4 min steam
    {"Sona Masoori", 20, 1.8, 86, 360},   // Sona Masoori: 20 min, 1.8:1 ratio, 86°C, 6 min steam
    {"Ponni", 22, 1.9, 87, 480},          // Ponni: 22 min, 1.9:1 ratio, 87°C, 8 min steam
    {"Brown Rice", 35, 2.2, 90, 900}      // Brown: 35 min, 2.2:1 ratio, 90°C, 15 min steam
};

device_config_t device_config = {
    .phone_number = "+919876543210",
    .sms_enabled = true,
    .temperature_sensor_pin = 2,
    .moisture_sensor_pin = 3,
    .buzzer_pin = 4,
    .led_pin = 5
};

cooking_state_t current_state = STATE_IDLE;
rice_type_t selected_rice_type = BASMATI_RICE;
time_t cooking_start_time;
time_t steaming_start_time;

// Function prototypes
void init_hardware(void);
int read_temperature(void);
int read_moisture_level(void);
void send_sms(const char* message);
void sound_buzzer(int duration_ms);
void set_led_status(bool on);
void display_menu(void);
rice_type_t select_rice_type(void);
void start_cooking(rice_type_t rice_type);
void monitor_cooking_process(void);
void handle_cooking_complete(void);
void log_event(const char* event);

// Hardware abstraction functions (to be implemented based on platform)
void init_hardware(void) {
    printf("Initializing hardware...\n");
    // Initialize GPIO pins
    // gpio_init();
    // gpio_set_mode(device_config.buzzer_pin, GPIO_OUTPUT);
    // gpio_set_mode(device_config.led_pin, GPIO_OUTPUT);
    // gpio_set_mode(device_config.temperature_sensor_pin, GPIO_INPUT);
    // gpio_set_mode(device_config.moisture_sensor_pin, GPIO_INPUT);
    
    // Initialize GSM module
    // gsm_init();
    
    printf("Hardware initialized successfully.\n");
}

int read_temperature(void) {
    // Simulate temperature reading from ADC
    // In real implementation, this would read from temperature sensor
    // int adc_value = adc_read(device_config.temperature_sensor_pin);
    // return convert_adc_to_temperature(adc_value);
    
    // Simulation: return temperature based on cooking state
    switch(current_state) {
        case STATE_COOKING:
            return 75 + (rand() % 15); // 75-90°C during cooking
        case STATE_STEAMING:
            return 85 + (rand() % 5);  // 85-90°C during steaming
        case STATE_DONE:
            return 60 + (rand() % 10); // 60-70°C when done
        default:
            return 25 + (rand() % 5);  // Room temperature
    }
}

int read_moisture_level(void) {
    // Simulate moisture reading
    // In real implementation, this would read from moisture sensor
    // int adc_value = adc_read(device_config.moisture_sensor_pin);
    // return convert_adc_to_moisture(adc_value);
    
    switch(current_state) {
        case STATE_COOKING:
            return 80 + (rand() % 20); // High moisture during cooking
        case STATE_STEAMING:
            return 60 + (rand() % 20); // Medium moisture during steaming
        case STATE_DONE:
            return 30 + (rand() % 20); // Lower moisture when done
        default:
            return 40 + (rand() % 10); // Ambient moisture
    }
}

void send_sms(const char* message) {
    if (!device_config.sms_enabled) {
        printf("SMS disabled. Message: %s\n", message);
        return;
    }
    
    printf("Sending SMS to %s: %s\n", device_config.phone_number, message);
    
    // Real implementation would use GSM module
    // gsm_send_sms(device_config.phone_number, message);
    
    log_event("SMS sent");
}

void sound_buzzer(int duration_ms) {
    printf("Buzzer: BEEP for %d ms\n", duration_ms);
    // gpio_write(device_config.buzzer_pin, HIGH);
    // delay_ms(duration_ms);
    // gpio_write(device_config.buzzer_pin, LOW);
}

void set_led_status(bool on) {
    printf("LED: %s\n", on ? "ON" : "OFF");
    // gpio_write(device_config.led_pin, on ? HIGH : LOW);
}

void display_menu(void) {
    printf("\n=== Smart Rice Cooker Controller ===\n");
    printf("Select Rice Type:\n");
    for (int i = 0; i < TOTAL_RICE_TYPES; i++) {
        printf("%d. %s (Cook: %d min, Water ratio: %.1f:1)\n", 
               i + 1, 
               rice_profiles[i].name, 
               rice_profiles[i].cooking_time_minutes,
               rice_profiles[i].water_ratio);
    }
    printf("0. Exit\n");
    printf("Choice: ");
}

rice_type_t select_rice_type(void) {
    int choice;
    display_menu();
    
    if (scanf("%d", &choice) != 1) {
        printf("Invalid input!\n");
        return BASMATI_RICE; // Default
    }
    
    if (choice >= 1 && choice <= TOTAL_RICE_TYPES) {
        return (rice_type_t)(choice - 1);
    } else if (choice == 0) {
        exit(0);
    } else {
        printf("Invalid choice! Using Basmati as default.\n");
        return BASMATI_RICE;
    }
}

void start_cooking(rice_type_t rice_type) {
    selected_rice_type = rice_type;
    current_state = STATE_COOKING;
    cooking_start_time = time(NULL);
    
    printf("\n=== Cooking Started ===\n");
    printf("Rice Type: %s\n", rice_profiles[rice_type].name);
    printf("Expected cooking time: %d minutes\n", rice_profiles[rice_type].cooking_time_minutes);
    printf("Water ratio: %.1f:1\n", rice_profiles[rice_type].water_ratio);
    
    set_led_status(true);
    sound_buzzer(1000); // 1 second beep to indicate start
    
    char sms_message[200];
    snprintf(sms_message, sizeof(sms_message), 
             "Rice cooking started! Type: %s, Expected time: %d minutes", 
             rice_profiles[rice_type].name, 
             rice_profiles[rice_type].cooking_time_minutes);
    send_sms(sms_message);
    
    log_event("Cooking started");
}

void monitor_cooking_process(void) {
    time_t current_time = time(NULL);
    int elapsed_minutes = (int)((current_time - cooking_start_time) / 60);
    int temperature = read_temperature();
    int moisture = read_moisture_level();
    
    rice_profile_t* profile = &rice_profiles[selected_rice_type];
    
    printf("Time: %02d:%02d | Temp: %d°C | Moisture: %d%% | State: ", 
           elapsed_minutes, (int)((current_time - cooking_start_time) % 60), 
           temperature, moisture);
    
    switch(current_state) {
        case STATE_COOKING:
            printf("COOKING\n");
            
            // Check if cooking time is reached and temperature is stable
            if (elapsed_minutes >= profile->cooking_time_minutes && 
                temperature >= profile->temp_threshold) {
                
                current_state = STATE_STEAMING;
                steaming_start_time = time(NULL);
                printf("Switching to steaming phase...\n");
                log_event("Steaming phase started");
            }
            break;
            
        case STATE_STEAMING:
            printf("STEAMING\n");
            
            int steaming_elapsed = (int)(current_time - steaming_start_time);
            if (steaming_elapsed >= profile->steam_time_seconds) {
                current_state = STATE_DONE;
                handle_cooking_complete();
            }
            break;
            
        case STATE_DONE:
            printf("DONE\n");
            break;
            
        default:
            printf("UNKNOWN\n");
            break;
    }
    
    // Error detection
    if (current_state == STATE_COOKING && elapsed_minutes > (profile->cooking_time_minutes + 10)) {
        if (temperature < 60) {
            current_state = STATE_ERROR;
            printf("ERROR: Temperature too low - check rice cooker connection!\n");
            send_sms("ERROR: Rice cooker temperature too low. Please check device.");
            sound_buzzer(5000); // Long error beep
        }
    }
}

void handle_cooking_complete(void) {
    printf("\n=== RICE IS READY! ===\n");
    printf("Rice Type: %s\n", rice_profiles[selected_rice_type].name);
    
    time_t total_time = time(NULL) - cooking_start_time;
    int total_minutes = (int)(total_time / 60);
    
    printf("Total cooking time: %d minutes\n", total_minutes);
    
    // Send completion SMS
    char sms_message[200];
    snprintf(sms_message, sizeof(sms_message), 
             "🍚 Your %s rice is ready! Cooked in %d minutes. Enjoy your meal!", 
             rice_profiles[selected_rice_type].name, 
             total_minutes);
    send_sms(sms_message);
    
    // Alert sequence
    for (int i = 0; i < 3; i++) {
        sound_buzzer(1000);
        set_led_status(true);
        sleep(1);
        set_led_status(false);
        sleep(1);
    }
    
    set_led_status(true); // Keep LED on to indicate ready status
    log_event("Cooking completed");
    
    printf("Press any key to return to menu...\n");
    getchar();
    getchar(); // Clear buffer
    
    current_state = STATE_IDLE;
    set_led_status(false);
}

void log_event(const char* event) {
    FILE* log_file = fopen("rice_cooker.log", "a");
    if (log_file) {
        time_t now = time(NULL);
        char* time_str = ctime(&now);
        time_str[strlen(time_str) - 1] = '\0'; // Remove newline
        
        fprintf(log_file, "[%s] %s\n", time_str, event);
        fclose(log_file);
    }
}

int main(void) {
    printf("Smart Rice Cooker Controller v1.0\n");
    printf("Supporting Indian Rice Varieties\n");
    printf("================================\n");
    
    srand(time(NULL)); // Initialize random seed for simulation
    init_hardware();
    
    while (1) {
        if (current_state == STATE_IDLE) {
            rice_type_t selected = select_rice_type();
            start_cooking(selected);
        }
        
        if (current_state == STATE_COOKING || current_state == STATE_STEAMING) {
            monitor_cooking_process();
            sleep(5); // Check every 5 seconds
        }
        
        if (current_state == STATE_ERROR) {
            printf("System in error state. Resetting...\n");
            current_state = STATE_IDLE;
            set_led_status(false);
        }
    }
    
    return 0;
}

// Configuration functions (can be expanded for user settings)
void update_phone_number(const char* new_number) {
    strncpy(device_config.phone_number, new_number, sizeof(device_config.phone_number) - 1);
    device_config.phone_number[sizeof(device_config.phone_number) - 1] = '\0';
    printf("Phone number updated to: %s\n", device_config.phone_number);
}

void toggle_sms_notifications(void) {
    device_config.sms_enabled = !device_config.sms_enabled;
    printf("SMS notifications: %s\n", device_config.sms_enabled ? "Enabled" : "Disabled");
}
