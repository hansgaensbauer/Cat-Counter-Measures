#ifndef CCM_IR_H
#define CCM_IR_H

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

void mlx90640_init();
void i2c_init();

uint16_t i2c_read_addr(uint8_t addr, uint16_t reg_addr, uint16_t* dataptr, uint16_t len);
uint8_t i2c_write_reg(uint8_t addr, uint16_t reg_addr, uint16_t data);
uint8_t mlx90640_read_image(int16_t* image_buff);

#endif