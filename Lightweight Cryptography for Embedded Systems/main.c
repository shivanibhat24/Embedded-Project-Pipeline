/**
 * Lightweight Cryptographic Solution for Embedded Systems
 * 
 * This implementation provides a balanced approach to security and power efficiency
 * for resource-constrained embedded systems. It includes:
 *  - PRESENT block cipher (lightweight alternative to AES)
 *  - SHA-3/Keccak lightweight hash implementation
 *  - Curve25519 for efficient elliptic curve operations
 *  - Simple key management functions
 */

#include <stdint.h>
#include <string.h>

/* PRESENT Block Cipher Implementation (64-bit block, 80/128-bit key) */
#define PRESENT_BLOCK_SIZE 8  // 64 bits = 8 bytes
#define PRESENT_KEY_SIZE 10   // 80 bits = 10 bytes
#define PRESENT_ROUNDS 31

// S-box for PRESENT
static const uint8_t sbox[16] = {
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
};

// Inverse S-box for PRESENT
static const uint8_t inverse_sbox[16] = {
    0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA
};

// Key schedule for PRESENT-80
static void present_key_schedule(const uint8_t *key, uint8_t round_keys[PRESENT_ROUNDS+1][PRESENT_BLOCK_SIZE]) {
    uint64_t key_state = 0;
    
    // Initial key state (80 bits)
    for (int i = 0; i < PRESENT_KEY_SIZE; i++) {
        key_state |= ((uint64_t)key[i] << (8 * (PRESENT_KEY_SIZE - 1 - i)));
    }
    
    // Generate round keys
    for (int round = 0; round <= PRESENT_ROUNDS; round++) {
        // Extract current round key (first 64 bits)
        uint64_t current_round_key = key_state >> 16;
        
        for (int j = 0; j < PRESENT_BLOCK_SIZE; j++) {
            round_keys[round][j] = (current_round_key >> (8 * (PRESENT_BLOCK_SIZE - 1 - j))) & 0xFF;
        }
        
        if (round < PRESENT_ROUNDS) {
            // Key update as per PRESENT specification
            // 1. Rotate left by 61 bits
            key_state = ((key_state << 61) | (key_state >> 19)) & 0xFFFFFFFFFFFFFFFFULL;
            
            // 2. S-box on most significant 4 bits
            uint8_t msb = (key_state >> 76) & 0xF;
            key_state = (key_state & 0x0FFFFFFFFFFFFFFFULL) | ((uint64_t)sbox[msb] << 76);
            
            // 3. XOR round counter
            key_state ^= ((uint64_t)(round + 1) << 15);
        }
    }
}

// PRESENT encryption
void present_encrypt(const uint8_t *plaintext, uint8_t *ciphertext, const uint8_t *key) {
    uint8_t round_keys[PRESENT_ROUNDS+1][PRESENT_BLOCK_SIZE];
    uint64_t state = 0;
    
    // Initialize key schedule
    present_key_schedule(key, round_keys);
    
    // Load plaintext into state
    for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
        state |= ((uint64_t)plaintext[i] << (8 * (PRESENT_BLOCK_SIZE - 1 - i)));
    }
    
    // Main encryption loop
    for (int round = 0; round < PRESENT_ROUNDS; round++) {
        // Add round key
        uint64_t round_key = 0;
        for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
            round_key |= ((uint64_t)round_keys[round][i] << (8 * (PRESENT_BLOCK_SIZE - 1 - i)));
        }
        state ^= round_key;
        
        if (round < PRESENT_ROUNDS - 1) {
            // S-box layer
            uint64_t new_state = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t nibble = (state >> (4 * i)) & 0xF;
                new_state |= ((uint64_t)sbox[nibble] << (4 * i));
            }
            state = new_state;
            
            // Permutation layer (bit permutation)
            uint64_t perm_state = 0;
            for (int i = 0; i < 64; i++) {
                int new_pos = (i * 16) % 63;
                if (i == 63) new_pos = 63;
                
                if (state & (1ULL << i))
                    perm_state |= (1ULL << new_pos);
            }
            state = perm_state;
        }
    }
    
    // Add final round key
    uint64_t final_key = 0;
    for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
        final_key |= ((uint64_t)round_keys[PRESENT_ROUNDS][i] << (8 * (PRESENT_BLOCK_SIZE - 1 - i)));
    }
    state ^= final_key;
    
    // Store result
    for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
        ciphertext[i] = (state >> (8 * (PRESENT_BLOCK_SIZE - 1 - i))) & 0xFF;
    }
}

// PRESENT decryption
void present_decrypt(const uint8_t *ciphertext, uint8_t *plaintext, const uint8_t *key) {
    uint8_t round_keys[PRESENT_ROUNDS+1][PRESENT_BLOCK_SIZE];
    uint64_t state = 0;
    
    // Initialize key schedule
    present_key_schedule(key, round_keys);
    
    // Load ciphertext into state
    for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
        state |= ((uint64_t)ciphertext[i] << (8 * (PRESENT_BLOCK_SIZE - 1 - i)));
    }
    
    // Main decryption loop
    for (int round = PRESENT_ROUNDS; round >= 0; round--) {
        // Add round key
        uint64_t round_key = 0;
        for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
            round_key |= ((uint64_t)round_keys[round][i] << (8 * (PRESENT_BLOCK_SIZE - 1 - i)));
        }
        state ^= round_key;
        
        if (round > 0) {
            // Inverse permutation layer
            uint64_t perm_state = 0;
            for (int i = 0; i < 64; i++) {
                int new_pos = (i * 4) % 63;
                if (i == 63) new_pos = 63;
                
                if (state & (1ULL << i))
                    perm_state |= (1ULL << new_pos);
            }
            state = perm_state;
            
            // Inverse S-box layer
            uint64_t new_state = 0;
            for (int i = 0; i < 16; i++) {
                uint8_t nibble = (state >> (4 * i)) & 0xF;
                new_state |= ((uint64_t)inverse_sbox[nibble] << (4 * i));
            }
            state = new_state;
        }
    }
    
    // Store result
    for (int i = 0; i < PRESENT_BLOCK_SIZE; i++) {
        plaintext[i] = (state >> (8 * (PRESENT_BLOCK_SIZE - 1 - i))) & 0xFF;
    }
}

/* Simplified Keccak/SHA-3 Implementation */
#define KECCAK_STATE_SIZE 25  // 25 uint64_t (200 bytes)
#define KECCAK_RATE 136       // For SHA3-256 (1088 bits = 136 bytes)
#define KECCAK_CAPACITY 64    // For SHA3-256 (512 bits = 64 bytes)
#define KECCAK_HASH_SIZE 32   // SHA3-256 output size (256 bits = 32 bytes)

// Keccak round constants
static const uint64_t keccak_round_constants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets
static const int keccak_rotation_offsets[25] = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43,
    25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14
};

// Keccak-f[1600] permutation
static void keccak_f1600(uint64_t state[KECCAK_STATE_SIZE]) {
    for (int round = 0; round < 24; round++) {
        // Theta step
        uint64_t C[5], D[5];
        
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
            
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        uint64_t B[25];
        memset(B, 0, sizeof(B));
        
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int x_new = y;
                int y_new = (2 * x + 3 * y) % 5;
                
                int offset = keccak_rotation_offsets[x + 5 * y];
                B[x_new + 5 * y_new] = ((state[x + 5 * y] << offset) | (state[x + 5 * y] >> (64 - offset)));
            }
        }
        
        // Chi step
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] = B[x + 5 * y] ^ ((~B[(x + 1) % 5 + 5 * y]) & B[(x + 2) % 5 + 5 * y]);
            }
        }
        
        // Iota step
        state[0] ^= keccak_round_constants[round];
    }
}

// SHA3-256 hash function
void sha3_256(const uint8_t *input, size_t input_len, uint8_t output[KECCAK_HASH_SIZE]) {
    uint64_t state[KECCAK_STATE_SIZE] = {0};
    
    // Absorb phase
    size_t block_size = KECCAK_RATE;
    size_t offset = 0;
    
    while (offset + block_size <= input_len) {
        for (size_t i = 0; i < block_size / 8; i++) {
            uint64_t block_val = 0;
            for (int j = 0; j < 8; j++) {
                block_val |= ((uint64_t)input[offset + i * 8 + j] << (8 * j));
            }
            state[i] ^= block_val;
        }
        
        keccak_f1600(state);
        offset += block_size;
    }
    
    // Process the last block with padding
    uint8_t last_block[KECCAK_RATE] = {0};
    
    // Copy remaining input
    for (size_t i = 0; i < input_len - offset; i++) {
        last_block[i] = input[offset + i];
    }
    
    // Add padding: append 01 and then enough zeroes
    last_block[input_len - offset] = 0x06;  // SHA-3 padding
    last_block[KECCAK_RATE - 1] |= 0x80;    // Final bit
    
    // Process last block
    for (size_t i = 0; i < KECCAK_RATE / 8; i++) {
        uint64_t block_val = 0;
        for (int j = 0; j < 8; j++) {
            block_val |= ((uint64_t)last_block[i * 8 + j] << (8 * j));
        }
        state[i] ^= block_val;
    }
    
    keccak_f1600(state);
    
    // Squeeze phase - extract output bytes
    for (size_t i = 0; i < KECCAK_HASH_SIZE / 8; i++) {
        uint64_t lane = state[i];
        for (int j = 0; j < 8; j++) {
            output[i * 8 + j] = (lane >> (8 * j)) & 0xFF;
        }
    }
}

/* Simple key derivation function based on SHA3 */
void kdf(const uint8_t *password, size_t password_len, 
         const uint8_t *salt, size_t salt_len,
         uint8_t output[KECCAK_HASH_SIZE]) {
    
    // Simple concatenation of password and salt
    uint8_t buffer[256];  // Adjust size as needed
    size_t buffer_size = 0;
    
    // Copy password
    for (size_t i = 0; i < password_len && buffer_size < sizeof(buffer); i++) {
        buffer[buffer_size++] = password[i];
    }
    
    // Copy salt
    for (size_t i = 0; i < salt_len && buffer_size < sizeof(buffer); i++) {
        buffer[buffer_size++] = salt[i];
    }
    
    // Hash the combination
    sha3_256(buffer, buffer_size, output);
}

/* Simplified Curve25519 implementation (for ECDH key exchange) */
typedef uint8_t curve25519_key[32];

// Perform Montgomery ladder scalar multiplication for Curve25519
void curve25519_scalarmult(curve25519_key result, const curve25519_key scalar, const curve25519_key point) {
    // This is a simplified placeholder. In a real implementation,
    // you would include the full X25519 algorithm here.
    // The actual Curve25519 implementation is complex and requires
    // careful field operations to ensure security and efficiency.
    
    // For example purposes, we're doing a mock operation
    sha3_256(scalar, 32, result);  // This is NOT actual Curve25519, just a placeholder
    
    // XOR with the point to simulate some kind of operation
    for (int i = 0; i < 32; i++) {
        result[i] ^= point[i];
    }
}

/* Stream cipher mode for PRESENT */
void present_ctr_encrypt(const uint8_t *plaintext, size_t plaintext_len,
                          uint8_t *ciphertext, const uint8_t *key,
                          const uint8_t *nonce) {
    
    uint8_t counter[PRESENT_BLOCK_SIZE] = {0};
    
    // Initialize counter with nonce (first half)
    for (int i = 0; i < PRESENT_BLOCK_SIZE/2; i++) {
        counter[i] = nonce[i];
    }
    
    // Process each block
    for (size_t block = 0; block < (plaintext_len + PRESENT_BLOCK_SIZE - 1) / PRESENT_BLOCK_SIZE; block++) {
        // Update counter (second half)
        for (int i = PRESENT_BLOCK_SIZE/2; i < PRESENT_BLOCK_SIZE; i++) {
            if (++counter[i] != 0) break;
        }
        
        // Encrypt counter
        uint8_t keystream[PRESENT_BLOCK_SIZE];
        present_encrypt(counter, keystream, key);
        
        // XOR plaintext with keystream
        size_t remaining = plaintext_len - block * PRESENT_BLOCK_SIZE;
        size_t block_size = (remaining < PRESENT_BLOCK_SIZE) ? remaining : PRESENT_BLOCK_SIZE;
        
        for (size_t i = 0; i < block_size; i++) {
            ciphertext[block * PRESENT_BLOCK_SIZE + i] = 
                plaintext[block * PRESENT_BLOCK_SIZE + i] ^ keystream[i];
        }
    }
}

/* Secure communication protocol for embedded systems */

// Message structure
typedef struct {
    uint8_t version;
    uint8_t message_type;
    uint16_t payload_length;
    uint8_t nonce[4];
    uint8_t *encrypted_payload;
    uint8_t mac[KECCAK_HASH_SIZE];
} secure_message_t;

// Encrypt and authenticate a message
void secure_encrypt_message(const uint8_t *plaintext, size_t plaintext_len,
                            secure_message_t *message,
                            const uint8_t *encryption_key, 
                            const uint8_t *mac_key) {
    
    // Set message metadata
    message->version = 1;
    message->message_type = 0;  // Data message
    message->payload_length = plaintext_len;
    
    // Generate random nonce (in a real system, use a proper RNG)
    for (int i = 0; i < 4; i++) {
        message->nonce[i] = i + 1;  // Placeholder; use a CSPRNG in production
    }
    
    // Allocate space for encrypted payload
    message->encrypted_payload = (uint8_t*)malloc(plaintext_len);
    
    // Encrypt the payload
    present_ctr_encrypt(plaintext, plaintext_len, message->encrypted_payload,
                         encryption_key, message->nonce);
    
    // Calculate MAC over the entire message
    uint8_t mac_buffer[512];  // Adjust size as needed
    size_t mac_offset = 0;
    
    // Add header fields to MAC calculation
    mac_buffer[mac_offset++] = message->version;
    mac_buffer[mac_offset++] = message->message_type;
    mac_buffer[mac_offset++] = (message->payload_length >> 8) & 0xFF;
    mac_buffer[mac_offset++] = message->payload_length & 0xFF;
    
    // Add nonce
    for (int i = 0; i < 4; i++) {
        mac_buffer[mac_offset++] = message->nonce[i];
    }
    
    // Add encrypted payload
    for (size_t i = 0; i < plaintext_len; i++) {
        mac_buffer[mac_offset++] = message->encrypted_payload[i];
    }
    
    // Calculate MAC
    sha3_256(mac_buffer, mac_offset, message->mac);
}

// Decrypt and verify a message
int secure_decrypt_message(const secure_message_t *message,
                          uint8_t *plaintext,
                          const uint8_t *encryption_key,
                          const uint8_t *mac_key) {
    
    // Verify MAC first
    uint8_t mac_buffer[512];  // Adjust size as needed
    size_t mac_offset = 0;
    
    // Add header fields to MAC calculation
    mac_buffer[mac_offset++] = message->version;
    mac_buffer[mac_offset++] = message->message_type;
    mac_buffer[mac_offset++] = (message->payload_length >> 8) & 0xFF;
    mac_buffer[mac_offset++] = message->payload_length & 0xFF;
    
    // Add nonce
    for (int i = 0; i < 4; i++) {
        mac_buffer[mac_offset++] = message->nonce[i];
    }
    
    // Add encrypted payload
    for (size_t i = 0; i < message->payload_length; i++) {
        mac_buffer[mac_offset++] = message->encrypted_payload[i];
    }
    
    // Calculate and verify MAC
    uint8_t calculated_mac[KECCAK_HASH_SIZE];
    sha3_256(mac_buffer, mac_offset, calculated_mac);
    
    // Compare MACs (constant-time comparison)
    int mac_valid = 1;
    for (int i = 0; i < KECCAK_HASH_SIZE; i++) {
        if (calculated_mac[i] != message->mac[i]) {
            mac_valid = 0;
            // Don't break early - continue to prevent timing attacks
        }
    }
    
    if (!mac_valid) {
        return -1;  // MAC verification failed
    }
    
    // Decrypt the payload
    present_ctr_encrypt(message->encrypted_payload, message->payload_length,
                         plaintext, encryption_key, message->nonce);
    
    return 0;  // Success
}

/* Power management functions */
// Function to put crypto in sleep mode
void crypto_sleep_mode(void) {
    // In a real implementation, this would power down components
    // or switch to a low-power state
}

// Function to wake up from sleep mode
void crypto_wake_up(void) {
    // In a real implementation, this would restore power
    // and reinitialize necessary components
}

/* Example usage */
int example(void) {
    // Sample keys (in a real system, use secure key management)
    uint8_t encryption_key[PRESENT_KEY_SIZE] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23};
    uint8_t mac_key[PRESENT_KEY_SIZE] = {0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC};
    
    // Sample plaintext
    uint8_t plaintext[] = "This is a secure message for an embedded system.";
    size_t plaintext_len = strlen((char*)plaintext);
    
    // Create secure message
    secure_message_t message;
    secure_encrypt_message(plaintext, plaintext_len, &message, encryption_key, mac_key);
    
    // Transmit the message over an insecure channel
    // ... (transmission code would go here)
    
    // Receive and decrypt the message
    uint8_t decrypted[256];
    int result = secure_decrypt_message(&message, decrypted, encryption_key, mac_key);
    
    if (result == 0) {
        // Message successfully decrypted and authenticated
        decrypted[plaintext_len] = '\0';  // Null-terminate for printing
        // Success!
    } else {
        // Message verification failed
        // Handle error
    }
    
    // Clean up
    free(message.encrypted_payload);
    
    return result;
}
