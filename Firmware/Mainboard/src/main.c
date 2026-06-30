/*
 * main.c
 */
#include <samd21.h>
#include "uart.h"
#include "mlx90640.h"
#include "main.h"
#include <stdio.h>
#include <string.h> 
#include <stdarg.h>

int main(void)
{
    clock_init();
    uart_init();
    mlx90640_init();
    debug_printf("hello world!\n\r\n\r");
    uint16_t rdata = 0;
    i2c_read_reg(IR_I2C_ADDR, 0x800D, &rdata);

    debug_printf("rdata: %x", rdata);

    while (1) {
    }
}

void clock_init(){

    NVMCTRL->CTRLB.bit.RWS = 1;
    
    //Enable XOSC32K
    SYSCTRL->XOSC32K.reg = SYSCTRL_XOSC32K_STARTUP(0x6)
                         | SYSCTRL_XOSC32K_XTALEN
                         | SYSCTRL_XOSC32K_EN32K
                         | SYSCTRL_XOSC32K_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.XOSC32KRDY);

    // GCLK1 is div 1
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(1) | GCLK_GENDIV_DIV(1);

    // Enable GCLK1
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(1)
                    | GCLK_GENCTRL_SRC_XOSC32K
                    | GCLK_GENCTRL_GENEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    //Enable MUX 0 (DFLL) with clock sourced from GCLK1
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID(0)
                      | GCLK_CLKCTRL_GEN_GCLK1
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    //Enable DFLL to get 48MHz
    SYSCTRL->DFLLCTRL.reg;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.bit.ONDEMAND = 0;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    uint32_t coarse = (*(volatile uint32_t *)0x806020UL >> 26) & 0x3FUL;
    SYSCTRL->DFLLVAL.reg = SYSCTRL_DFLLVAL_COARSE(coarse);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_MUL(1465)
                         | SYSCTRL_DFLLMUL_FSTEP(10)
                         | SYSCTRL_DFLLMUL_CSTEP(10);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE
                          | SYSCTRL_DFLLCTRL_MODE
                          | SYSCTRL_DFLLCTRL_WAITLOCK;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    while (!SYSCTRL->PCLKSR.bit.DFLLLCKC || !SYSCTRL->PCLKSR.bit.DFLLLCKF);

    //Enable GCLK0
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(0) | GCLK_GENDIV_DIV(1);

    //Set source for GCLK0 to DFLL48M
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(0)
                      | GCLK_GENCTRL_SRC_DFLL48M
                      | GCLK_GENCTRL_IDC
                      | GCLK_GENCTRL_GENEN;

    while (GCLK->STATUS.bit.SYNCBUSY);

    //Connect SERCOM0 (debug UART) to GCLK0
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_SERCOM0_CORE
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    //Connect SERCOM2 to GCLK0
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_SERCOM2_CORE
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);
}