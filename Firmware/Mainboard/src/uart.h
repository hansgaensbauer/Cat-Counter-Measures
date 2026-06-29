#ifndef CCM_UART_H
#define CCM_UART_H

#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h> 

#define UART_BUFFER_SIZE 256

void uart_putc(char c);
void uart_init(void);
void debug_printf(const char *fmt, ...);

#endif /* CCM_UART_H */