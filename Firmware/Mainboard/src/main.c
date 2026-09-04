/*
 * main.c
 */
#include <samd21.h>
#include "uart.h"
#include "mlx90640.h"
#include "MLX90640_API.h"
#include "main.h"
#include "boost.h"
#include <stdio.h>
#include <string.h> 
#include <stdarg.h>
#include <math.h>


static float mlx90640Image[768];

//State variables and flags
volatile char charge_done = 0;

int main(void)
{
    initialize();

    MLX90640_SetChessMode(IR_I2C_ADDR);
    MLX90640_SetResolution(IR_I2C_ADDR, IR_RES_18_BIT);
    MLX90640_SetRefreshRate(IR_I2C_ADDR, IR_REFRESH_RATE_2Hz);
    MLX90640_SynchFrame(IR_I2C_ADDR);

    // boost_init();
    // uint16_t adc = read_boost_voltage();
    // debug_printf("BV: %d\n\r", adc);
    // boost_start_charge();
    // while(!charge_done);
    // debug_printf("Charge done.\n\r");
    // for(int i = 0; i < 100; i++){
    //     debug_printf("hello world!");
    // }
    // boost_stop_charge();

    
    // debug_printf("hello world!\n\r\n\r");
    // uint16_t adc = read_boost_voltage();
    // debug_printf('ADC Reading:  %d', adc);

    

    while (1) {
        // boost_start_charge();

        PORT->Group[0].OUTTGL.reg = PORT_PA18;
        // uint16_t adc = read_boost_voltage();
        // debug_printf("BV: %d\n\r", adc);

        // for(volatile int i = 0; i < 100; i++){
        //     uint16_t cnt = 0;
        //     for(volatile int j = 0; j < 20000; j++){
        //         cnt = cnt + i + j;
        //     }
        // }

        mlx90640_read_image(mlx90640Image);
        float testfloat = 22.3;
        for(int i = 0; i < 20; i++){
            debug_printf("%d,", (int)(mlx90640Image[i]*100));
        }
        // if(center_pixel > -10){
        //     // fire();
        // }
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

void initialize(){
    clock_init();
    __enable_irq();
    uart_init();
    PORT->Group[0].OUTCLR.reg = PORT_PA16;
    PORT->Group[0].DIRSET.reg = PORT_PA16;
    PORT->Group[0].DIRSET.reg = PORT_PA18;
    if(mlx90640_init()){
        debug_printf("Camera initialization failed");
    }
}

void fire(){
    PORT->Group[0].OUTSET.reg = PORT_PA16;
    for(volatile int i = 0; i < 100; i++){
        uint16_t cnt = 0;
        for(volatile int j = 0; j < 20000; j++){
            cnt = cnt + i + j;
        }
    }
    PORT->Group[0].OUTCLR.reg = PORT_PA16;
}

void clock_init(){

    NVMCTRL->CTRLB.bit.RWS = 1;
    
    //Enable OSC32K
    SYSCTRL->OSC32K.reg = SYSCTRL_OSC32K_STARTUP(0x6)
                         | SYSCTRL_OSC32K_EN32K;

    SYSCTRL->OSC32K.bit.ENABLE = 1;
    while (!SYSCTRL->PCLKSR.bit.OSC32KRDY);

    // GCLK1 is div 1
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(1) | GCLK_GENDIV_DIV(1);

    // Enable GCLK1
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(1)
                    | GCLK_GENCTRL_SRC_OSCULP32K
                    | GCLK_GENCTRL_IDC
                    | GCLK_GENCTRL_GENEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    //Enable MUX 0 (DFLL) with clock sourced from GCLK1
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_DFLL48
                      | GCLK_CLKCTRL_GEN_GCLK1
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    //Enable DFLL to get 48MHz
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);
    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_MUL(1465)
                         | SYSCTRL_DFLLMUL_FSTEP(511)
                         | SYSCTRL_DFLLMUL_CSTEP(31);

    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    uint32_t coarse = (*((uint32_t *)FUSES_DFLL48M_COARSE_CAL_ADDR) & FUSES_DFLL48M_COARSE_CAL_Msk) >> FUSES_DFLL48M_COARSE_CAL_Pos;
    SYSCTRL->DFLLVAL.bit.COARSE = coarse;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE
                          | SYSCTRL_DFLLCTRL_MODE
                          | SYSCTRL_DFLLCTRL_WAITLOCK;

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