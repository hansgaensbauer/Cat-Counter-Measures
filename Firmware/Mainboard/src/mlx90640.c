#include <samd21.h>
#include "mlx90640.h"

void mlx90640_init(){
    //Camera enable pin
    PORT->Group[0].DIRSET.reg = CAMERA_EN_PIN;

    //Turn on camera
    PORT->Group[0].OUTCLR.reg = CAMERA_EN_PIN;

    i2c_init();

}

uint8_t mlx90640_read_image(int16_t* image_buff){

    //Poll "new data in RAM"
    uint16_t status_reg = 0;
    while(!(status_reg & IR_NEW_DATA_BIT)){
        i2c_read_addr(IR_I2C_ADDR, IR_STATUS_REG, &status_reg, 1);
    }

    //Possibly - magic undocumented init status value (0x30) to STATUS_REG???
    i2c_write_reg(IR_I2C_ADDR, IR_STATUS_REG, 0x30);

    //Read frame 1 (chess pattern)
    i2c_read_addr(IR_I2C_ADDR, IR_IMG_RAM_START, (uint16_t*) image_buff, IR_NUM_PIXELS);

    //Clear bit "new data in RAM"
    i2c_read_addr(IR_I2C_ADDR, IR_STATUS_REG, &status_reg, 1);
    i2c_write_reg(IR_I2C_ADDR, IR_STATUS_REG, status_reg & ~IR_NEW_DATA_BIT);

    return 0;
}

void i2c_init(){
    //Initialize SERCOM2 for I2C
    PM->APBCMASK.reg |= PM_APBCMASK_SERCOM2;

    //PORT
    //SDA is PA8
    //SCL is PA9

    PORT->Group[0].PINCFG[8].reg = PORT_PINCFG_PMUXEN;
    PORT->Group[0].PINCFG[9].reg = PORT_PINCFG_PMUXEN;

    PORT->Group[0].PMUX[4].reg = PORT_PMUX_PMUXE(MUX_PA08D_SERCOM2_PAD0)
                            | PORT_PMUX_PMUXO(MUX_PA09D_SERCOM2_PAD1);

    //CTRLA
    SERCOM2->I2CM.CTRLA.reg = SERCOM_I2CM_CTRLA_MODE_I2C_MASTER;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //CTRLB - enable smart mode
    // SERCOM2->I2CM.CTRLB.reg |= SERCOM_I2CM_CTRLB_SMEN;
    // while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //BAUD: 48000000/(10+2*250) is about 94117
    SERCOM2->I2CM.BAUD.reg = SERCOM_I2CM_BAUD_BAUD(40);
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //Enable
    SERCOM2->I2CM.CTRLA.reg |= SERCOM_SPI_CTRLA_ENABLE;
    while(SERCOM2->I2CM.SYNCBUSY.bit.ENABLE);

    //Force bus state to idle
    SERCOM2->I2CM.STATUS.bit.BUSSTATE = 0x1;

}

uint16_t i2c_read_addr(uint8_t addr, uint16_t reg_addr, uint16_t* dataptr, uint16_t len){
    //write address packet, 0 for write
    SERCOM2->I2CM.ADDR.reg = (addr << 1) | 0;

    //check INTFLAG.MB and STATUS.RXNACK
    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = reg_addr >> 8;

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = reg_addr & 0xFF;

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //write address packet, 1 for read
    SERCOM2->I2CM.ADDR.reg = (addr << 1) | 1;

    //Wait for SB
    while(!SERCOM2->I2CM.INTFLAG.bit.SB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    for(int n = 0; n < len * 2 - 1; n++){
        //ACK...
        SERCOM2->I2CM.CTRLB.reg &= ~SERCOM_I2CM_CTRLB_ACKACT;
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //...then one more
        SERCOM2->I2CM.CTRLB.bit.CMD = 0x2;
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //Swap endian-ness
        if(n % 2){
            *(((uint8_t*) dataptr) + n - 1) = SERCOM2->I2CM.DATA.reg;
        }else{
            *(((uint8_t*) dataptr) + n + 1) = SERCOM2->I2CM.DATA.reg;
        }
            
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //Wait for SB
        while(!SERCOM2->I2CM.INTFLAG.bit.SB);
        if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
            return 1;
        }
    }
    //last byte

    //NACK...
    SERCOM2->I2CM.CTRLB.reg |= SERCOM_I2CM_CTRLB_ACKACT;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //...and stop
    SERCOM2->I2CM.CTRLB.bit.CMD = 0x3;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //Read in data
    *(((uint8_t*) dataptr) + 2*len - 2) = SERCOM2->I2CM.DATA.reg;

    return 0;

}

uint8_t i2c_write_reg(uint8_t addr, uint16_t reg_addr, uint16_t data){
    //write address packet
    SERCOM2->I2CM.ADDR.reg = (addr << 1) | 0;;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //check INTFLAG.MB and STATUS.RXNACK
    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = reg_addr >> 8;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = reg_addr & 0xFF;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put data MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = data >> 8;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    //put data LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = data & 0xFF;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return 1;
    }

    return 0;
}