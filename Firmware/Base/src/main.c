/*
 * main.c
 */
#include <samd21.h>
#include "tusb.h"
    
void clock_init(){
    
    PM->APBBMASK.reg |= PM_APBBMASK_USB;
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

    //Connect USB (6) to GCLK0
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_USB
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);
    
    //Connect SERCOM0 (debug UART) ot GCLK
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_SERCOM0_CORE
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

}

void usb_pin_init(void)
{
    PORT->Group[0].PINCFG[24].reg = PORT_PINCFG_PMUXEN;
    PORT->Group[0].PINCFG[25].reg = PORT_PINCFG_PMUXEN;

    PORT->Group[0].PMUX[12].reg = PORT_PMUX_PMUXE(MUX_PA24G_USB_DM)
                                | PORT_PMUX_PMUXO(MUX_PA25G_USB_DP);
}

int main(void)
{
    system_init();
    usb_clock_init_xosc32k();
    tc3_init();

    __enable_irq();
    usb_pin_init();
    
    if(tusb_init()){
        REG_PORT_OUT0 &= ~PORT_PA17;
    }

    while (1) {
        tud_task();
    }
}

void USB_Handler(void)
{
    tusb_int_handler(0, true);
}