
#include <samd21.h>
#include "uart.h"
#include <stdio.h>
#include <string.h> 
#include <stdarg.h>

void uart_init(){

    //Enable SERCOM0 in PM
    PM->APBCMASK.reg |= PM_APBCMASK_SERCOM0;

    //PORT
    //TXD is PA10
    //RXD is PA11
    PORT->Group[0].DIRSET.reg = PORT_PA10;
    PORT->Group[0].DIRCLR.reg = PORT_PA11;

    PORT->Group[0].PINCFG[10].reg = PORT_PINCFG_PMUXEN;
    PORT->Group[0].PINCFG[11].reg = PORT_PINCFG_PMUXEN;

    PORT->Group[0].PMUX[5].reg = PORT_PMUX_PMUXE(MUX_PA10C_SERCOM0_PAD2)
                            | PORT_PMUX_PMUXO(MUX_PA11C_SERCOM0_PAD3);

    //GCLK is assumed to already be set up

    //CTRLA. MSB first. Asynchronous, TX on PAD2, RX on PAD3
    SERCOM0->USART.CTRLA.reg = SERCOM_USART_CTRLA_DORD |
        SERCOM_USART_CTRLA_RXPO(0x3) |
        SERCOM_USART_CTRLA_TXPO(0x1) |
        SERCOM_USART_CTRLA_SAMPR(0x0)|
        SERCOM_USART_CTRLA_MODE_USART_INT_CLK ;

    //CTRLB. 
    SERCOM0->USART.CTRLB.reg = SERCOM_USART_CTRLB_CHSIZE(0x0) |
        SERCOM_USART_CTRLB_TXEN |
        SERCOM_USART_CTRLB_RXEN ;

    //calculate baud rate. BAUD = 65536*(1-S*115200/48000000)
    SERCOM0->USART.BAUD.reg = 63019;
    while(SERCOM0->USART.SYNCBUSY.bit.CTRLB);

    //Enable the UART
    SERCOM0->USART.CTRLA.reg |= SERCOM_USART_CTRLA_ENABLE;
    while(SERCOM0->USART.SYNCBUSY.reg & SERCOM_USART_SYNCBUSY_ENABLE);
}

void uart_putc(char c){
    SERCOM0->USART.DATA.reg = c;
    while(!SERCOM0->USART.INTFLAG.bit.DRE);
}

void debug_printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    char buf[UART_BUFFER_SIZE];
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    if (len <= 0) return;
    if ((size_t)len >= sizeof(buf)) {
        len = sizeof(buf) - 1;
    }
    int i = 0;
    while (buf[i]) {
        uart_putc((char)(buf[i]));
        i = i + 1;
    }
}
