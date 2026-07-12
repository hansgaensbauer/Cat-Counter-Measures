##############################################################################
# Paths
##############################################################################
TINYUSB_ROOT := tinyusb
ASF_ROOT      := /home/hans/Documents/xdk-asf-3.52.0
BUILD_DIR     := build

##############################################################################
# Toolchain
##############################################################################
CC      := arm-none-eabi-gcc
OBJCOPY := arm-none-eabi-objcopy
SIZE    := arm-none-eabi-size

##############################################################################
# MCU flags
##############################################################################
MCU_FLAGS := \
  -mcpu=cortex-m0plus \
  -mthumb \
  -mfloat-abi=soft

##############################################################################
# Source files
##############################################################################
SRC_C := \
  src/main.c \
  src/uart.c \
  src/mlx90640.c \
  src/boost.c \
  $(ASF_ROOT)/sam0/drivers/system/system.c \
  $(ASF_ROOT)/sam0/drivers/system/interrupt/system_interrupt.c \
  $(ASF_ROOT)/sam0/utils/cmsis/samd21/source/gcc/startup_samd21.c \
  $(ASF_ROOT)/sam0/utils/syscalls/gcc/syscalls.c

##############################################################################
# Include paths
##############################################################################
INC := \
  src \
  $(ASF_ROOT)/sam0/utils/cmsis/samd21/include \
  $(ASF_ROOT)/sam0/utils/cmsis/samd21/source \
  $(ASF_ROOT)/sam0/utils/header_files \
  $(ASF_ROOT)/sam0/utils \
  $(ASF_ROOT)/sam0/utils/preprocessor \
  $(ASF_ROOT)/sam0/drivers/system \
  $(ASF_ROOT)/sam0/drivers/system/pinmux \
  $(ASF_ROOT)/sam0/drivers/system/reset/reset_sam_d_r_h \
  $(ASF_ROOT)/sam0/drivers/system/power/power_sam_d_r_h \
  $(ASF_ROOT)/sam0/drivers/system/clock \
  $(ASF_ROOT)/sam0/drivers/system/clock/clock_samd21_r21_da_ha1 \
  $(ASF_ROOT)/sam0/drivers/system/interrupt \
  $(ASF_ROOT)/sam0/drivers/system/interrupt/system_interrupt_samd21 \
  $(ASF_ROOT)/thirdparty/CMSIS/Include \
  $(ASF_ROOT)/common/utils

##############################################################################
# Defines
##############################################################################
DEFS := \
  -D__SAMD21E17D__ \
  -DARM_MATH_CM0PLUS \
  -DBOARD=USER_BOARD

##############################################################################
# Flags
##############################################################################
CFLAGS := $(MCU_FLAGS) $(DEFS) \
  $(addprefix -I, $(INC)) \
  -Os -g \
  -ffunction-sections -fdata-sections \
  -Wall -Wextra \
  -std=gnu11

LDFLAGS := $(MCU_FLAGS) \
  -T$(ASF_ROOT)/sam0/utils/linker_scripts/samd21/gcc/samd21e17d_flash.ld \
  -Wl,--gc-sections \
  -Wl,-Map=$(BUILD_DIR)/output.map \
  --specs=nosys.specs

##############################################################################
# Build rules
##############################################################################
OBJS := $(patsubst %.c, $(BUILD_DIR)/%.o, $(SRC_C))

all: $(BUILD_DIR)/firmware.elf $(BUILD_DIR)/firmware.bin
	$(SIZE) $(BUILD_DIR)/firmware.elf

$(BUILD_DIR)/firmware.elf: $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/firmware.bin: $(BUILD_DIR)/firmware.elf
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

flash: $(BUILD_DIR)/firmware.bin
	openocd -f interface/cmsis-dap.cfg -f target/at91samdXX.cfg \
	  -c "program $(BUILD_DIR)/firmware.bin verify reset exit 0x00000000"

.PHONY: all clean flash