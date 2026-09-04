#include <samd21.h>
#include "mlx90640.h"
#include "MLX90640_API.h"
#include "uart.h"
#include <math.h>


float ta = -999.0;
uint16_t serialNumber[3];
paramsMLX90640 _params;
static uint16_t eeMLX90640[832];
static uint16_t mlx90640Frame[834];

int mlx90640_init(){
    //Camera enable pin
    PORT->Group[0].DIRSET.reg = CAMERA_EN_PIN;

    //Turn on camera
    PORT->Group[0].OUTCLR.reg = CAMERA_EN_PIN;
    debug_printf("Initializing I2C\n\r");
    MLX90640_I2CInit();

    
    if (MLX90640_DumpEE(IR_I2C_ADDR, eeMLX90640) != 0) {
        debug_printf("EEPROM Read Fail");
        return 1;
    }
    debug_printf("EEPROM Read Successful\n\r");

    MLX90640_ExtractParameters(eeMLX90640, &_params);

    return 0;
}

int mlx90640_read_image(float* image_buff){

    float emissivity = 0.95;
    MLX90640_GetFrameData(IR_I2C_ADDR, mlx90640Frame); //page 0
    ta = MLX90640_GetTa(mlx90640Frame, &_params);
    float tr = ta-8;
    MLX90640_CalculateTo(mlx90640Frame, &_params, emissivity, tr, image_buff);

    MLX90640_GetFrameData(IR_I2C_ADDR, mlx90640Frame); //Page 1
    ta = MLX90640_GetTa(mlx90640Frame, &_params);
    tr = ta-8;
    MLX90640_CalculateTo(mlx90640Frame, &_params, emissivity, tr, image_buff);

    return 0;
}

//------------------------------------------------------------------------------

void MLX90640_GetImage_INT16(uint16_t *frameData, const paramsMLX90640 *params, int16_t *result)
{
    float vdd;
    float ta;
    float gain;
    float irDataCP[2];
    float irData;
    float alphaCompensated;
    uint8_t mode;
    int8_t ilPattern;
    int8_t chessPattern;
    int8_t pattern;
    int8_t conversionPattern;
    float image;
    uint16_t subPage;
    float ktaScale;
    float kvScale;
    float kta;
    float kv;
    
    subPage = frameData[833];
    vdd = MLX90640_GetVdd(frameData, params);
    ta = MLX90640_GetTa(frameData, params);
    
    ktaScale = POW2(params->ktaScale);
    kvScale = POW2(params->kvScale);
    
//------------------------- Gain calculation -----------------------------------    
    
    gain = (float)params->gainEE / (int16_t)frameData[778]; 
  
//------------------------- Image calculation -------------------------------------    
    
    mode = (frameData[832] & MLX90640_CTRL_MEAS_MODE_MASK) >> 5;
    
    irDataCP[0] = (int16_t)frameData[776] * gain;
    irDataCP[1] = (int16_t)frameData[808] * gain;
    
    irDataCP[0] = irDataCP[0] - params->cpOffset[0] * (1 + params->cpKta * (ta - 25)) * (1 + params->cpKv * (vdd - 3.3));
    if( mode ==  params->calibrationModeEE)
    {
        irDataCP[1] = irDataCP[1] - params->cpOffset[1] * (1 + params->cpKta * (ta - 25)) * (1 + params->cpKv * (vdd - 3.3));
    }
    else
    {
      irDataCP[1] = irDataCP[1] - (params->cpOffset[1] + params->ilChessC[0]) * (1 + params->cpKta * (ta - 25)) * (1 + params->cpKv * (vdd - 3.3));
    }

    for( int pixelNumber = 0; pixelNumber < 768; pixelNumber++)
    {
        ilPattern = pixelNumber / 32 - (pixelNumber / 64) * 2; 
        chessPattern = ilPattern ^ (pixelNumber - (pixelNumber/2)*2); 
        conversionPattern = ((pixelNumber + 2) / 4 - (pixelNumber + 3) / 4 + (pixelNumber + 1) / 4 - pixelNumber / 4) * (1 - 2 * ilPattern);
        
        if(mode == 0)
        {
          pattern = ilPattern; 
        }
        else 
        {
          pattern = chessPattern; 
        }
        
        if(pattern == frameData[833])
        {    
            irData = (int16_t)frameData[pixelNumber] * gain;
            
            kta = params->kta[pixelNumber]/ktaScale;
            kv = params->kv[pixelNumber]/kvScale;
            irData = irData - params->offset[pixelNumber]*(1 + kta*(ta - 25))*(1 + kv*(vdd - 3.3));

            if(mode !=  params->calibrationModeEE)
            {
              irData = irData + params->ilChessC[2] * (2 * ilPattern - 1) - params->ilChessC[1] * conversionPattern; 
            }
            
            irData = irData - params->tgc * irDataCP[subPage];
                        
            alphaCompensated = params->alpha[pixelNumber];
            
            image = irData*alphaCompensated;
            
            result[pixelNumber] = (int16_t) image;
        }
    }
}

void MLX90640_I2CGeneralReset(){}

void MLX90640_I2CInit(){
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

int MLX90640_I2CRead(uint8_t slaveAddr, uint16_t startAddress, uint16_t nMemAddressRead, uint16_t* data){
    //write address packet, 0 for write
    SERCOM2->I2CM.ADDR.reg = (slaveAddr << 1) | 0;

    //check INTFLAG.MB and STATUS.RXNACK
    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = startAddress >> 8;

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = startAddress & 0xFF;

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //write address packet, 1 for read
    SERCOM2->I2CM.ADDR.reg = (slaveAddr << 1) | 1;

    //Wait for SB
    while(!SERCOM2->I2CM.INTFLAG.bit.SB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    for(int n = 0; n < nMemAddressRead * 2 - 1; n++){
        //ACK...
        SERCOM2->I2CM.CTRLB.reg &= ~SERCOM_I2CM_CTRLB_ACKACT;
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //...then one more
        SERCOM2->I2CM.CTRLB.bit.CMD = 0x2;
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //Swap endian-ness
        if(n % 2){
            *(((uint8_t*) data) + n - 1) = SERCOM2->I2CM.DATA.reg;
        }else{
            *(((uint8_t*) data) + n + 1) = SERCOM2->I2CM.DATA.reg;
        }
            
        while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

        //Wait for SB
        while(!SERCOM2->I2CM.INTFLAG.bit.SB);
        if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
            return -1;
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
    *(((uint8_t*) data) + 2*nMemAddressRead - 2) = SERCOM2->I2CM.DATA.reg;

    return 0;

}

int MLX90640_I2CWrite(uint8_t slaveAddr, uint16_t writeAddress, uint16_t data){
    //write address packet
    SERCOM2->I2CM.ADDR.reg = (slaveAddr << 1) | 0;;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    //check INTFLAG.MB and STATUS.RXNACK
    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = writeAddress >> 8;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = writeAddress & 0xFF;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put data MSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = data >> 8;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    //put data LSB into DATA.DATA
    SERCOM2->I2CM.DATA.reg = data & 0xFF;
    while(SERCOM2->I2CM.SYNCBUSY.bit.SYSOP);

    while(!SERCOM2->I2CM.INTFLAG.bit.MB);
    if(SERCOM2->I2CM.STATUS.bit.RXNACK || SERCOM2->I2CM.STATUS.bit.BUSERR){
        return -1;
    }

    return 0;
}

void MLX90640_I2CFreqSet(int freq){}