/*
 * main.c
 */
#include <samd21.h>
#include "tusb.h"

static void usb_clock_init_usbcrm(void)
{
    /* Must set NVM wait state before increasing clock speed */
    NVMCTRL->CTRLB.bit.RWS = 1;

    /* Errata 9905 — dummy read first, then wait for ready */
    SYSCTRL->DFLLCTRL.reg;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Clear ONDEMAND explicitly */
    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.bit.ONDEMAND = 0;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Load factory coarse calibration */
    uint32_t coarse = (*(volatile uint32_t *)0x806020UL >> 26) & 0x3FUL;
    SYSCTRL->DFLLVAL.reg = SYSCTRL_DFLLVAL_COARSE(coarse);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Set multiplier for USBCRM (48MHz / 1kHz SOF = 48000) */
    SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_MUL(48000)
                         | SYSCTRL_DFLLMUL_FSTEP(1)
                         | SYSCTRL_DFLLMUL_CSTEP(1);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Enable DFLL with USBCRM */
    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE
                          | SYSCTRL_DFLLCTRL_USBCRM
                          | SYSCTRL_DFLLCTRL_MODE
                          | SYSCTRL_DFLLCTRL_CCDIS
                          | SYSCTRL_DFLLCTRL_BPLCKC;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Switch GCLK0 to DFLL48M */
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(0) | GCLK_GENDIV_DIV(1);
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(0)
                      | GCLK_GENCTRL_SRC_DFLL48M
                      | GCLK_GENCTRL_IDC
                      | GCLK_GENCTRL_GENEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    /* Route GCLK0 to USB peripheral */
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID(6)
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    PM->APBBMASK.reg |= PM_APBBMASK_USB;
}

static void usb_clock_init_xosc32k(void)
{
    PM->APBBMASK.reg |= PM_APBBMASK_USB;
    /* Must set NVM wait state before increasing clock speed */
    NVMCTRL->CTRLB.bit.RWS = 1;

    /* Enable XOSC32K */
    SYSCTRL->XOSC32K.reg = SYSCTRL_XOSC32K_STARTUP(0x6)
                         | SYSCTRL_XOSC32K_XTALEN
                         | SYSCTRL_XOSC32K_EN32K
                         | SYSCTRL_XOSC32K_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.XOSC32KRDY);

    /* Route XOSC32K to GCLK1 */
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(1) | GCLK_GENDIV_DIV(1);
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(1)
                      | GCLK_GENCTRL_SRC_XOSC32K
                      | GCLK_GENCTRL_GENEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    /* Route GCLK1 to DFLL48M reference (channel 0) */
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID(0)
                      | GCLK_CLKCTRL_GEN_GCLK1
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    /* Errata 9905 — dummy read before any DFLL register access */
    SYSCTRL->DFLLCTRL.reg;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    SYSCTRL->DFLLCTRL.bit.ONDEMAND = 0;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Load factory coarse calibration */
    uint32_t coarse = (*(volatile uint32_t *)0x806020UL >> 26) & 0x3FUL;
    SYSCTRL->DFLLVAL.reg = SYSCTRL_DFLLVAL_COARSE(coarse);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* 48MHz / 32768Hz = 1465 */
    SYSCTRL->DFLLMUL.reg = SYSCTRL_DFLLMUL_MUL(1465)
                         | SYSCTRL_DFLLMUL_FSTEP(10)
                         | SYSCTRL_DFLLMUL_CSTEP(10);
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Enable DFLL in closed-loop mode */
    SYSCTRL->DFLLCTRL.reg = SYSCTRL_DFLLCTRL_ENABLE
                          | SYSCTRL_DFLLCTRL_MODE
                          | SYSCTRL_DFLLCTRL_WAITLOCK;
    while (!SYSCTRL->PCLKSR.bit.DFLLRDY);

    /* Wait for coarse and fine lock */
    while (!SYSCTRL->PCLKSR.bit.DFLLLCKC || !SYSCTRL->PCLKSR.bit.DFLLLCKF);

    /* Switch GCLK0 to DFLL48M */
    GCLK->GENDIV.reg  = GCLK_GENDIV_ID(0) | GCLK_GENDIV_DIV(1);
    GCLK->GENCTRL.reg = GCLK_GENCTRL_ID(0)
                      | GCLK_GENCTRL_SRC_DFLL48M
                      | GCLK_GENCTRL_IDC
                      | GCLK_GENCTRL_GENEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    /* Route GCLK0 to USB peripheral */
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID(6)
                      | GCLK_CLKCTRL_GEN_GCLK0
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);
}

void tc3_init(void)
{
    /* Route GCLK1 (32.768 kHz) to TC3 */
    GCLK->CLKCTRL.reg = GCLK_CLKCTRL_ID_TCC2_TC3_Val
                      | GCLK_CLKCTRL_GEN_GCLK1
                      | GCLK_CLKCTRL_CLKEN;
    while (GCLK->STATUS.bit.SYNCBUSY);

    /* Enable TC3 in power manager */
    PM->APBCMASK.reg |= PM_APBCMASK_TC3;

    /* Reset TC3 */
    TC3->COUNT16.CTRLA.reg = TC_CTRLA_SWRST;
    while (TC3->COUNT16.STATUS.bit.SYNCBUSY);
    while (TC3->COUNT16.CTRLA.bit.SWRST);

    /* 32768 / 1 = 32768 Hz
     * Match frequency mode, no prescaler
     * CC0 = 16383 → toggles every 16384 ticks = 2 Hz */
    TC3->COUNT16.CTRLA.reg = TC_CTRLA_MODE_COUNT16
                           | TC_CTRLA_WAVEGEN_MFRQ
                           | TC_CTRLA_PRESCALER_DIV1;

    TC3->COUNT16.CC[0].reg = 16383;
    while (TC3->COUNT16.STATUS.bit.SYNCBUSY);

    /* Enable match interrupt */
    TC3->COUNT16.INTENSET.reg = TC_INTENSET_MC0;

    /* Enable TC3 */
    TC3->COUNT16.CTRLA.reg |= TC_CTRLA_ENABLE;
    while (TC3->COUNT16.STATUS.bit.SYNCBUSY);

    /* Enable TC3 interrupt in NVIC */
    NVIC_SetPriority(TC3_IRQn, 3);
    NVIC_EnableIRQ(TC3_IRQn);

    /* Configure PA17 as output */
    PORT->Group[0].DIRSET.reg = (1 << 17);
    PORT->Group[0].OUTCLR.reg = (1 << 17);
}

void usb_pin_init(void)
{
    /* PA24 and PA25 to peripheral function G (USB) */
    PORT->Group[0].PINCFG[24].reg = PORT_PINCFG_PMUXEN;
    PORT->Group[0].PINCFG[25].reg = PORT_PINCFG_PMUXEN;

    /* PMUX index = pin/2 = 12, PA24 is even (PMUXE), PA25 is odd (PMUXO) */
    PORT->Group[0].PMUX[12].reg = PORT_PMUX_PMUXE(MUX_PA24G_USB_DM)
                                | PORT_PMUX_PMUXO(MUX_PA25G_USB_DP);
}

void TC3_Handler(void)
{
    /* Clear the match interrupt flag */
    TC3->COUNT16.INTFLAG.reg = TC_INTFLAG_MC0;

    /* Toggle PA17 */
    PORT->Group[0].OUTTGL.reg = (1 << 17);
}

int main(void)
{
    system_init();
    usb_clock_init_xosc32k();
    tc3_init();

    __enable_irq();
    usb_pin_init();
    // usb_clock_init_usbcrm();
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