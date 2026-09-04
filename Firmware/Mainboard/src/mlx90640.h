#ifndef CCM_IR_H
#define CCM_IR_H

#include "MLX90640_API.h"

#define IR_I2C_ADDR 0x33
#define CAMERA_EN_PIN PORT_PA28

#define IR_NUM_PIXELS 768

#define IR_STATUS_REG 0x8000
#define IR_CTRL_REG_1 0x800D
#define IR_I2C_CONF_REG 0x800F
#define IR_IMG_RAM_START 0x0400

#define IR_NEW_DATA_BIT (1 << 3)
#define IR_SUBPAGE_MODE_BIT (1<<0)
#define IR_SUBPAGE_REPEAT_BIT (1<<3)

#define IR_RES_16_BIT 0x00
#define IR_RES_17_BIT 0x01
#define IR_RES_18_BIT 0x02
#define IR_RES_19_BIT 0x03

#define IR_REFRESH_RATE_0_5Hz 0x00
#define IR_REFRESH_RATE_1Hz 0x01
#define IR_REFRESH_RATE_2Hz 0x02
#define IR_REFRESH_RATE_4Hz 0x03
#define IR_REFRESH_RATE_8Hz 0x04
#define IR_REFRESH_RATE_16Hz 0x05
#define IR_REFRESH_RATE_32Hz 0x06
#define IR_REFRESH_RATE_64Hz 0x07

int mlx90640_init();
void MLX90640_I2CInit();
void MLX90640_I2CGeneralReset();

uint16_t i2c_read_addr(uint8_t addr, uint16_t reg_addr, uint16_t* dataptr, uint16_t len);
uint8_t i2c_write_reg(uint8_t addr, uint16_t reg_addr, uint16_t data);
int mlx90640_read_image(float* image_buff);
void MLX90640_GetImage_INT16(uint16_t *frameData, const paramsMLX90640 *params, int16_t *result);

#endif