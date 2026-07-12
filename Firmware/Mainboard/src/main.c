/*
 * main.c
 */
#include <samd21.h>
#include "uart.h"
#include "mlx90640.h"
#include "main.h"
#include "boost.h"
#include <stdio.h>
#include <string.h> 
#include <stdarg.h>

uint16_t img_buff[24*32];

//State variables and flags
volatile char charge_done = 0;

int main(void)
{
    clock_init();
    __enable_irq();
    uart_init();

    boost_init();
    uint16_t adc = read_boost_voltage();
    debug_printf("BV: %d\n\r", adc);
    // boost_start_charge();
    // while(!charge_done);
    // debug_printf("Charge done.\n\r");
    // for(int i = 0; i < 100; i++){
    //     debug_printf("hello world!");
    // }
    // boost_stop_charge();

    // mlx90640_init();
    // debug_printf("hello world!\n\r\n\r");
    // uint16_t adc = read_boost_voltage();
    // debug_printf('ADC Reading:  %d', adc);

    

    while (1) {
        // boost_start_charge();
        for(int i = 0; i < 1000; i++){
            uint16_t cnt = 0;
            for(int j = 0; j < 20000; j++){
                cnt = cnt + i + j;
            }
        }
        // boost_stop_charge();
        uint16_t adc = read_boost_voltage();
        debug_printf("BV: %d\n\r", adc);
    }
}

void print_image(int16_t* img){
    debug_printf("\n\r");
    for(int i = 0; i < 24; i ++){
        for(int j = 0; j < 32; j ++){
            debug_printf("%d,", *(img + 32*i + j));
        }
        debug_printf("\n\r");
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

    //Clock for DAC, ADC, AC
    //AC clk - 48MHz
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_AC_DIG
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    // Feed GCLK1 to the AC ANA (32.786kHz)
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_AC_ANA |
                        GCLK_CLKCTRL_GEN_GCLK1 |
                        GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    // Configure GCLK4 divisor
    GCLK->GENDIV.reg = GCLK_GENDIV_ID(4) | GCLK_GENDIV_DIV(3);

    // Enable GCLK4, source = DFLL48M
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(4) |
                        GCLK_GENCTRL_GENEN |
                        GCLK_GENCTRL_SRC_DFLL48M |
                        GCLK_GENCTRL_DIVSEL;

    while (GCLK->STATUS.bit.SYNCBUSY);

    // Feed GCLK4 to the DAC
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_DAC |
                        GCLK_CLKCTRL_GEN_GCLK4 |
                        GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_ADC
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    // Clock for boost switch
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_TCC0_TCC1
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);
}

void HardFault_Handler(){
    //Turn off boost converter switch!
    PORT->Group[0].PINCFG[7].reg = 0;
    PORT->Group[0].OUTCLR.reg = PORT_PA07;
}