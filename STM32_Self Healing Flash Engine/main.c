#include "stm32f4xx_hal.h"
#include "stm32f4xx_ll_flash.h"
#include <string.h>
#include <stdio.h>

/* Private defines */
#define FLASH_USER_START_ADDR     0x08020000  /* Start of user flash area */
#define FLASH_USER_END_ADDR       0x08080000  /* End of user flash */
#define FLASH_SECTOR_SIZE         0x20000     /* 128KB sectors for F411RE */
#define FLASH_PAGE_SIZE           0x4000      /* 16KB pages */

#define PRIMARY_DATA_START        0x08020000  /* Primary data blocks start */
#define PRIMARY_DATA_SIZE         0x20000     /* 128KB for primary data */
#define BACKUP_DATA_START         0x08040000  /* Backup data blocks start */
#define BACKUP_DATA_SIZE          0x20000     /* 128KB for backup data */

#define MAX_DATA_BLOCKS           8           /* Number of data blocks */
#define BLOCK_SIZE               (PRIMARY_DATA_SIZE / MAX_DATA_BLOCKS)

#define CRC_TABLE_START          0x08060000   /* CRC lookup table location */
#define STATUS_LOG_START         0x08070000   /* Status log location */

/* Status definitions */
typedef enum {
    INTEGRITY_OK = 0,
    INTEGRITY_CORRUPTED,
    INTEGRITY_REPAIRED,
    INTEGRITY_REPAIR_FAILED
} IntegrityStatus_t;

/* Block information structure */
typedef struct {
    uint32_t start_addr;
    uint32_t backup_addr;
    uint32_t size;
    uint32_t expected_crc;
    uint32_t actual_crc;
    IntegrityStatus_t status;
} FlashBlock_t;

/* Global variables */
CRC_HandleTypeDef hcrc;
UART_HandleTypeDef huart2;
FlashBlock_t flash_blocks[MAX_DATA_BLOCKS];
char uart_buffer[256];

/* Private function prototypes */
static void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_CRC_Init(void);
static void MX_USART2_UART_Init(void);
static void Flash_Integrity_Init(void);
static HAL_StatusTypeDef Flash_Unlock_Safe(void);
static HAL_StatusTypeDef Flash_Lock_Safe(void);
static uint32_t Calculate_CRC32(uint32_t start_addr, uint32_t size);
static IntegrityStatus_t Check_Block_Integrity(uint8_t block_index);
static HAL_StatusTypeDef Repair_Corrupted_Block(uint8_t block_index);
static HAL_StatusTypeDef Erase_Flash_Sector(uint32_t sector_addr);
static HAL_StatusTypeDef Program_Flash_Block(uint32_t dest_addr, uint32_t src_addr, uint32_t size);
static void Log_Status_UART(const char* message);
static void Log_Block_Status(uint8_t block_index);
static void Initialize_Test_Data(void);
static void Run_Integrity_Check_Cycle(void);

/**
 * @brief Main program entry point
 */
int main(void)
{
    /* MCU Configuration */
    HAL_Init();
    SystemClock_Config();
    
    /* Initialize peripherals */
    MX_GPIO_Init();
    MX_CRC_Init();
    MX_USART2_UART_Init();
    
    /* Initialize flash integrity system */
    Flash_Integrity_Init();
    
    Log_Status_UART("\r\n=== STM32 Self-Healing Flash Integrity Engine ===\r\n");
    Log_Status_UART("System initialized successfully\r\n");
    
    /* Initialize test data (for demonstration) */
    Initialize_Test_Data();
    
    /* Main integrity monitoring loop */
    while (1)
    {
        Run_Integrity_Check_Cycle();
        HAL_Delay(5000); /* Check every 5 seconds */
    }
}

/**
 * @brief Initialize the flash integrity system
 */
static void Flash_Integrity_Init(void)
{
    /* Initialize flash block structures */
    for (uint8_t i = 0; i < MAX_DATA_BLOCKS; i++)
    {
        flash_blocks[i].start_addr = PRIMARY_DATA_START + (i * BLOCK_SIZE);
        flash_blocks[i].backup_addr = BACKUP_DATA_START + (i * BLOCK_SIZE);
        flash_blocks[i].size = BLOCK_SIZE;
        flash_blocks[i].status = INTEGRITY_OK;
        
        /* Calculate expected CRC from backup area */
        flash_blocks[i].expected_crc = Calculate_CRC32(flash_blocks[i].backup_addr, BLOCK_SIZE);
    }
    
    Log_Status_UART("Flash integrity system initialized\r\n");
}

/**
 * @brief Safe flash unlock with error handling
 */
static HAL_StatusTypeDef Flash_Unlock_Safe(void)
{
    if (LL_FLASH_IsActiveFlag_EOP())
        LL_FLASH_ClearFlag_EOP();
    
    if (LL_FLASH_IsActiveFlag_OPERR())
        LL_FLASH_ClearFlag_OPERR();
    
    if (LL_FLASH_IsActiveFlag_WRPERR())
        LL_FLASH_ClearFlag_WRPERR();
    
    if (LL_FLASH_IsActiveFlag_PGAERR())
        LL_FLASH_ClearFlag_PGAERR();
    
    return HAL_FLASH_Unlock();
}

/**
 * @brief Safe flash lock
 */
static HAL_StatusTypeDef Flash_Lock_Safe(void)
{
    return HAL_FLASH_Lock();
}

/**
 * @brief Calculate CRC32 for a flash memory block
 * @param start_addr: Starting address of the block
 * @param size: Size of the block in bytes
 * @return Calculated CRC32 value
 */
static uint32_t Calculate_CRC32(uint32_t start_addr, uint32_t size)
{
    uint32_t crc_result = 0;
    uint32_t *data_ptr = (uint32_t*)start_addr;
    uint32_t word_count = size / 4;
    
    /* Reset CRC peripheral */
    __HAL_CRC_DR_RESET(&hcrc);
    
    /* Calculate CRC for the block */
    for (uint32_t i = 0; i < word_count; i++)
    {
        crc_result = HAL_CRC_Accumulate(&hcrc, &data_ptr[i], 1);
    }
    
    return crc_result;
}

/**
 * @brief Check integrity of a specific flash block
 * @param block_index: Index of the block to check
 * @return Integrity status
 */
static IntegrityStatus_t Check_Block_Integrity(uint8_t block_index)
{
    if (block_index >= MAX_DATA_BLOCKS)
        return INTEGRITY_REPAIR_FAILED;
    
    /* Calculate actual CRC of the primary block */
    flash_blocks[block_index].actual_crc = Calculate_CRC32(
        flash_blocks[block_index].start_addr, 
        flash_blocks[block_index].size
    );
    
    /* Compare with expected CRC */
    if (flash_blocks[block_index].actual_crc == flash_blocks[block_index].expected_crc)
    {
        flash_blocks[block_index].status = INTEGRITY_OK;
        return INTEGRITY_OK;
    }
    else
    {
        flash_blocks[block_index].status = INTEGRITY_CORRUPTED;
        return INTEGRITY_CORRUPTED;
    }
}

/**
 * @brief Repair a corrupted flash block from backup
 * @param block_index: Index of the block to repair
 * @return HAL status
 */
static HAL_StatusTypeDef Repair_Corrupted_Block(uint8_t block_index)
{
    HAL_StatusTypeDef status = HAL_OK;
    
    if (block_index >= MAX_DATA_BLOCKS)
        return HAL_ERROR;
    
    snprintf(uart_buffer, sizeof(uart_buffer), 
             "Repairing block %d from backup...\r\n", block_index);
    Log_Status_UART(uart_buffer);
    
    /* Unlock flash for writing */
    if (Flash_Unlock_Safe() != HAL_OK)
    {
        Log_Status_UART("Failed to unlock flash\r\n");
        return HAL_ERROR;
    }
    
    /* Erase the corrupted sector */
    status = Erase_Flash_Sector(flash_blocks[block_index].start_addr);
    if (status != HAL_OK)
    {
        Log_Status_UART("Failed to erase corrupted sector\r\n");
        Flash_Lock_Safe();
        return status;
    }
    
    /* Program the block with backup data */
    status = Program_Flash_Block(
        flash_blocks[block_index].start_addr,
        flash_blocks[block_index].backup_addr,
        flash_blocks[block_index].size
    );
    
    if (status == HAL_OK)
    {
        /* Verify repair was successful */
        if (Check_Block_Integrity(block_index) == INTEGRITY_OK)
        {
            flash_blocks[block_index].status = INTEGRITY_REPAIRED;
            Log_Status_UART("Block repair successful\r\n");
        }
        else
        {
            flash_blocks[block_index].status = INTEGRITY_REPAIR_FAILED;
            Log_Status_UART("Block repair verification failed\r\n");
            status = HAL_ERROR;
        }
    }
    else
    {
        flash_blocks[block_index].status = INTEGRITY_REPAIR_FAILED;
        Log_Status_UART("Failed to program repaired block\r\n");
    }
    
    /* Lock flash */
    Flash_Lock_Safe();
    
    return status;
}

/**
 * @brief Erase a flash sector using LL APIs
 * @param sector_addr: Address within the sector to erase
 * @return HAL status
 */
static HAL_StatusTypeDef Erase_Flash_Sector(uint32_t sector_addr)
{
    uint32_t sector_number;
    
    /* Determine sector number based on address */
    if (sector_addr >= 0x08020000 && sector_addr < 0x08040000)
        sector_number = FLASH_SECTOR_5;
    else if (sector_addr >= 0x08040000 && sector_addr < 0x08060000)
        sector_number = FLASH_SECTOR_6;
    else if (sector_addr >= 0x08060000 && sector_addr < 0x08080000)
        sector_number = FLASH_SECTOR_7;
    else
        return HAL_ERROR;
    
    /* Wait for any ongoing operation */
    while (LL_FLASH_IsActiveFlag_BSY());
    
    /* Set sector to erase */
    LL_FLASH_SetSectorToErase(sector_number);
    
    /* Start erase operation */
    LL_FLASH_StartErase();
    
    /* Wait for completion */
    while (LL_FLASH_IsActiveFlag_BSY());
    
    /* Check for errors */
    if (LL_FLASH_IsActiveFlag_OPERR() || 
        LL_FLASH_IsActiveFlag_WRPERR() ||
        LL_FLASH_IsActiveFlag_PGAERR())
    {
        return HAL_ERROR;
    }
    
    return HAL_OK;
}

/**
 * @brief Program a flash block using LL APIs
 * @param dest_addr: Destination address
 * @param src_addr: Source address
 * @param size: Size in bytes
 * @return HAL status
 */
static HAL_StatusTypeDef Program_Flash_Block(uint32_t dest_addr, uint32_t src_addr, uint32_t size)
{
    uint32_t *src_ptr = (uint32_t*)src_addr;
    uint32_t *dest_ptr = (uint32_t*)dest_addr;
    uint32_t word_count = size / 4;
    
    /* Set programming word size */
    LL_FLASH_SetProgrammingSize(LL_FLASH_PROGRAM_WORD);
    
    for (uint32_t i = 0; i < word_count; i++)
    {
        /* Wait for any ongoing operation */
        while (LL_FLASH_IsActiveFlag_BSY());
        
        /* Program word */
        *dest_ptr = *src_ptr;
        
        /* Wait for completion */
        while (LL_FLASH_IsActiveFlag_BSY());
        
        /* Check for errors */
        if (LL_FLASH_IsActiveFlag_OPERR() || 
            LL_FLASH_IsActiveFlag_WRPERR() ||
            LL_FLASH_IsActiveFlag_PGAERR())
        {
            return HAL_ERROR;
        }
        
        /* Verify programming */
        if (*dest_ptr != *src_ptr)
        {
            return HAL_ERROR;
        }
        
        src_ptr++;
        dest_ptr++;
    }
    
    return HAL_OK;
}

/**
 * @brief Log status message via UART
 * @param message: Message to log
 */
static void Log_Status_UART(const char* message)
{
    HAL_UART_Transmit(&huart2, (uint8_t*)message, strlen(message), HAL_MAX_DELAY);
}

/**
 * @brief Log detailed block status
 * @param block_index: Index of the block
 */
static void Log_Block_Status(uint8_t block_index)
{
    const char* status_str;
    
    switch (flash_blocks[block_index].status)
    {
        case INTEGRITY_OK:
            status_str = "OK";
            break;
        case INTEGRITY_CORRUPTED:
            status_str = "CORRUPTED";
            break;
        case INTEGRITY_REPAIRED:
            status_str = "REPAIRED";
            break;
        case INTEGRITY_REPAIR_FAILED:
            status_str = "REPAIR_FAILED";
            break;
        default:
            status_str = "UNKNOWN";
            break;
    }
    
    snprintf(uart_buffer, sizeof(uart_buffer),
             "Block %d: Addr=0x%08lX, Expected CRC=0x%08lX, Actual CRC=0x%08lX, Status=%s\r\n",
             block_index,
             flash_blocks[block_index].start_addr,
             flash_blocks[block_index].expected_crc,
             flash_blocks[block_index].actual_crc,
             status_str);
    
    Log_Status_UART(uart_buffer);
}

/**
 * @brief Initialize test data in flash (for demonstration)
 */
static void Initialize_Test_Data(void)
{
    Log_Status_UART("Initializing test data...\r\n");
    
    /* This would typically be done during firmware deployment */
    /* For this demo, we assume backup data already exists */
    
    Log_Status_UART("Test data initialization complete\r\n");
}

/**
 * @brief Run one cycle of integrity checking
 */
static void Run_Integrity_Check_Cycle(void)
{
    uint8_t corrupted_blocks = 0;
    uint8_t repaired_blocks = 0;
    
    Log_Status_UART("\r\n--- Starting Integrity Check Cycle ---\r\n");
    
    /* Check integrity of all blocks */
    for (uint8_t i = 0; i < MAX_DATA_BLOCKS; i++)
    {
        IntegrityStatus_t status = Check_Block_Integrity(i);
        
        if (status == INTEGRITY_CORRUPTED)
        {
            corrupted_blocks++;
            Log_Block_Status(i);
            
            /* Attempt repair */
            if (Repair_Corrupted_Block(i) == HAL_OK)
            {
                repaired_blocks++;
            }
        }
        else if (status == INTEGRITY_OK)
        {
            /* Optionally log OK blocks (comment out to reduce output) */
            // Log_Block_Status(i);
        }
    }
    
    /* Summary */
    snprintf(uart_buffer, sizeof(uart_buffer),
             "Cycle complete: %d corrupted, %d repaired\r\n",
             corrupted_blocks, repaired_blocks);
    Log_Status_UART(uart_buffer);
    
    /* Toggle LED to indicate activity */
    HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
}

/**
 * @brief System Clock Configuration
 */
static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ = 4;
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        Error_Handler();
    }
    
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
 * @brief CRC Initialization Function
 */
static void MX_CRC_Init(void)
{
    hcrc.Instance = CRC;
    if (HAL_CRC_Init(&hcrc) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
 * @brief USART2 Initialization Function
 */
static void MX_USART2_UART_Init(void)
{
    huart2.Instance = USART2;
    huart2.Init.BaudRate = 115200;
    huart2.Init.WordLength = UART_WORDLENGTH_8B;
    huart2.Init.StopBits = UART_STOPBITS_1;
    huart2.Init.Parity = UART_PARITY_NONE;
    huart2.Init.Mode = UART_MODE_TX_RX;
    huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart2.Init.OverSampling = UART_OVERSAMPLING_16;
    
    if (HAL_UART_Init(&huart2) != HAL_OK)
    {
        Error_Handler();
    }
}

/**
 * @brief GPIO Initialization Function
 */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
    
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
    
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    
    GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

/**
 * @brief Error Handler
 */
void Error_Handler(void)
{
    __disable_irq();
    while (1)
    {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
        HAL_Delay(100);
    }
}

/**
 * @brief Assert failed handler
 */
#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
}
#endif
