#ifndef CCM_IR_H
#define CCM_IR_H

#define IR_I2C_ADDR 0x33
#define CAMERA_EN_PIN PORT_PA28

void mlx90640_init();
void i2c_init();

uint16_t i2c_read_reg(uint8_t addr, uint16_t reg_addr, uint16_t* dataptr);
uint8_t i2c_write_reg(uint8_t addr, uint16_t reg_addr, uint16_t data);

#endif