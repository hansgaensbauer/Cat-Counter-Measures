#ifndef UART_DEBUG_H
#define UART_DEBUG_H

#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>

/**
 * @file uart_debug.h
 * @brief SAMD21 UART driver using ASF (usart_serial / sercom) with
 *        a sprintf-style debug print function.
 *
 * Assumptions:
 *   - Atmel Software Framework (ASF) v3 is included in the project.
 *   - The SERCOM instance, baud rate, and pin mux are configured below
 *     via the UART_DEBUG_* macros — adjust for your board.
 *   - CONF_STDIO_USART is NOT used here; we manage our own instance.
 */

/* -----------------------------------------------------------------------
 * Board / peripheral configuration — edit these for your hardware
 * --------------------------------------------------------------------- */

/** SERCOM instance to use (SERCOM0 … SERCOM5) */
#define UART_DEBUG_SERCOM       SERCOM0

/** Baud rate */
#define UART_DEBUG_BAUD         115200UL
#define UART_DEBUG_MUX          USART_RX_3_TX_2_XCK_3

#define UART_DEBUG_TX_PIN       PIN_PA10C_SERCOM0_PAD2   /* TX */
#define UART_DEBUG_TX_MUX       MUX_PA10C_SERCOM0_PAD2

#define UART_DEBUG_RX_PIN       PIN_PA11C_SERCOM0_PAD3   /* RX */
#define UART_DEBUG_RX_MUX       MUX_PA11C_SERCOM0_PAD3

#define UART_DEBUG_BUF_SIZE     256

/* -----------------------------------------------------------------------
 * Public API
 * --------------------------------------------------------------------- */

/**
 * @brief  Initialise the debug UART peripheral.
 *
 * Call once during system startup, after system_init().
 *
 * @return true on success, false if ASF configuration fails.
 */
bool uart_debug_init(void);

/**
 * @brief  Transmit a single byte (blocking).
 *
 * @param  byte  Character to transmit.
 */
void uart_debug_putc(uint8_t byte);

/**
 * @brief  Transmit a null-terminated string (blocking).
 *
 * @param  str  Pointer to the string.
 */
void uart_debug_puts(const char *str);

/**
 * @brief  Transmit a raw byte buffer (blocking).
 *
 * @param  buf   Pointer to data.
 * @param  len   Number of bytes to send.
 */
void uart_debug_write(const uint8_t *buf, uint16_t len);

/**
 * @brief  sprintf-style debug print (blocking).
 *
 * Formats the string into a local buffer (UART_DEBUG_BUF_SIZE bytes)
 * then transmits it over the debug UART.
 *
 * Usage:
 *   debug_printf("ADC value: %d, voltage: %.3f V\r\n", raw, voltage);
 *
 * @param  fmt  printf-compatible format string.
 * @param  ...  Variadic arguments matching the format string.
 */
void debug_printf(const char *fmt, ...);

/**
 * @brief  va_list variant of debug_printf — useful when wrapping
 *         debug_printf inside another variadic function.
 */
void debug_vprintf(const char *fmt, va_list args);

#endif /* UART_DEBUG_H */