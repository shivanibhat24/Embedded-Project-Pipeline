TARGET = flash_integrity_engine
DEBUG = 1
OPT = -Og

BUILD_DIR = build
C_SOURCES = \\
Core/Src/main.c \\
Core/Src/stm32f4xx_it.c \\
Core/Src/stm32f4xx_hal_msp.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_crc.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_uart.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_rcc_ex.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_flash_ex.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_gpio.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma_ex.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_dma.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_pwr_ex.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_cortex.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal.c \\
Drivers/STM32F4xx_HAL_Driver/Src/stm32f4xx_hal_exti.c \\
Core/Src/system_stm32f4xx.c

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size

MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard

C_DEFS = \\
-DUSE_HAL_DRIVER \\
-DSTM32F411xE

C_INCLUDES = \\
-ICore/Inc \\
-IDrivers/STM32F4xx_HAL_Driver/Inc \\
-IDrivers/STM32F4xx_HAL_Driver/Inc/Legacy \\
-IDrivers/CMSIS/Device/ST/STM32F4xx/Include \\
-IDrivers/CMSIS/Include

CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections

ifeq ($(DEBUG), 1)
CFLAGS += -g -gdwarf-2
endif

CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"

LDSCRIPT = STM32F411RETX_FLASH.ld
LIBS = -lc -lm -lnosys
LIBDIR = 
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections

all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin
