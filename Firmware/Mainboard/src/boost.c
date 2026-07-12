#include <samd21.h>
#include "uart.h"
#include "boost.h"

//comparator init from DAC
void comparator_init(){
    PM->APBCMASK.reg |= PM_APBCMASK_AC | PM_APBCMASK_DAC | PM_APBCMASK_ADC;

    //Set up DAC
    DAC->CTRLB.reg = (
        DAC_CTRLB_IOEN |
        DAC_CTRLB_EOEN | //Necessary to drive ADC
        DAC_CTRLB_REFSEL_AVCC
    );
    while(DAC->STATUS.bit.SYNCBUSY);

    DAC->DATA.reg = BOOST_TOP_DAC_VALUE;
    while(DAC->STATUS.bit.SYNCBUSY);

    DAC->CTRLA.reg |= DAC_CTRLA_ENABLE;
    while(DAC->STATUS.bit.SYNCBUSY);

    //PA7 as comparator input
    PORT->Group[0].DIRCLR.reg = PORT_PA06;
    PORT->Group[0].PINCFG[6].reg = PORT_PINCFG_PMUXEN;
    PORT->Group[0].PMUX[3].reg = PORT_PMUX_PMUXE(MUX_PA06B_ADC_AIN6);

    //Set up input and output events
    // AC->COMPCTRL[0].reg = (AC_COMPCTRL_HYST | 
    //                         AC_COMPCTRL_MUXPOS_PIN2 | 
    //                         AC_COMPCTRL_MUXNEG_DAC |
    //                         AC_COMPCTRL_INTSEL_RISING
    //                     );
    // while(AC->STATUSB.bit.SYNCBUSY);

    // AC->COMPCTRL[0].reg |= AC_COMPCTRL_ENABLE;
    // while(AC->STATUSB.bit.SYNCBUSY);

    // //Enable COMP1 interrupt
    // AC->INTENSET.reg = AC_INTENSET_COMP0;

    // //Enable the comparator
    // AC->CTRLA.reg |= AC_CTRLA_ENABLE;
    // while(AC->STATUSB.bit.SYNCBUSY);

    //Set up the ADC
    ADC->REFCTRL.reg = ADC_REFCTRL_REFSEL_INTVCC1;
    ADC->AVGCTRL.reg = ADC_AVGCTRL_SAMPLENUM_1;
    ADC->CTRLB.reg = (ADC_CTRLB_PRESCALER_DIV128 |
                 ADC_CTRLB_RESSEL_12BIT);
    while(ADC->STATUS.bit.SYNCBUSY);
    ADC->INPUTCTRL.reg = ADC_INPUTCTRL_GAIN_DIV2 |
                        ADC_INPUTCTRL_MUXNEG_GND |
                        ADC_INPUTCTRL_MUXPOS_PIN6;
    while(ADC->STATUS.bit.SYNCBUSY);
    // ADC->SAMPCTRL.reg = ADC_SAMPCTRL_SAMPLEN(20); // pick x based on your actual divider resistance
    while (ADC->STATUS.bit.SYNCBUSY);

    //Enable the ADC
    ADC->CTRLA.reg |= ADC_CTRLA_ENABLE;
    while (ADC->STATUS.bit.SYNCBUSY);

    //Throw out the first reading!
    read_boost_voltage();
}

//adc read
uint16_t read_boost_voltage(){
    ADC->SWTRIG.reg = ADC_SWTRIG_START;
    while(!ADC->INTFLAG.bit.RESRDY);
    ADC->INTFLAG.reg = ADC_INTFLAG_RESRDY;
    return ADC->RESULT.reg;
}

//boost/timer init
void boost_init(){
    comparator_init();
    
    PORT->Group[0].DIRSET.reg = PORT_PA07;
    PORT->Group[0].OUTCLR.reg = PORT_PA07;

    //TCC1
    PM->APBCMASK.reg |= PM_APBCMASK_TCC1;
    TCC1->CTRLA.reg = (TCC_CTRLA_PRESCALER_DIV1);
    TCC1->WAVE.reg = (TCC_WAVE_WAVEGEN_NPWM);

    //Set period and compare value
    TCC1->PER.reg = 250; //3us
    while(TCC1->SYNCBUSY.bit.PER);
    //Worst case: 0V difference, so I should start with 
    TCC1->CC[1].reg = TCC_CC_CC(23); //Should maybe be 48, try 24 first. 5V, 1us = 5A
    while(TCC1->SYNCBUSY.bit.CC1);
}

//Start charge
void boost_start_charge(){
    //Check if charging is already done
    char chargestate = AC->STATUSA.bit.STATE0;
    char readystate = AC->STATUSB.bit.READY0;
    if(!readystate){
        debug_printf("Comparator not ready.\n\r");
    }else{
        if(chargestate){
            debug_printf("Charging Done.\n\r");
            charge_done = 1;
        }else{
            // TCC1->COUNT.reg = 0;
            // TCC1->CTRLA.reg |= TCC_CTRLA_ENABLE;
            // while(TCC1->SYNCBUSY.bit.ENABLE);
            // PORT->Group[0].PINCFG[7].reg = PORT_PINCFG_PMUXEN | PORT_PINCFG_DRVSTR;
            // PORT->Group[0].PMUX[3].reg = PORT_PMUX_PMUXO(MUX_PA07E_TCC1_WO1);
            // PORT->Group[0].DIRSET.reg = PORT_PA07;  //Careful!
        }
    }
}

void boost_stop_charge(){
    PORT->Group[0].PINCFG[7].reg = 0;
    PORT->Group[0].OUTCLR.reg = PORT_PA07;
    TCC1->CTRLA.reg &= ~TCC_CTRLA_ENABLE;
}

void AC_Handler(){
    boost_stop_charge();
    charge_done = 1;
}


//ISR to stop charge