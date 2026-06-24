// src/usb_descriptors.c
#include "tusb.h"

// String indices
enum { STRID_LANGID = 0, STRID_MANUFACTURER, STRID_PRODUCT, STRID_SERIAL };

#define PID_MAP(itf, n)  ((CFG_TUD_##itf) ? (1 << (n)) : 0)
#define USB_PID           (0x4000 | PID_MAP(CDC, 0) | PID_MAP(MSC, 1) | PID_MAP(HID, 2) | \
                           PID_MAP(MIDI, 3) | PID_MAP(VENDOR, 4) )

#define USB_VID   0xCafe
#define USB_BCD   0x0200

//--------------------------------------------------------------------+
// Device Descriptors
//--------------------------------------------------------------------+
static tusb_desc_device_t const desc_device = {
    .bLength            = sizeof(tusb_desc_device_t),
    .bDescriptorType    = TUSB_DESC_DEVICE,
    .bcdUSB             = USB_BCD,

    // Use Interface Association Descriptor (IAD) for CDC
    // As required by USB Specs IAD's subclass must be common class (2) and protocol must be IAD (1)
    .bDeviceClass       = TUSB_CLASS_MISC,
    .bDeviceSubClass    = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol    = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0    = CFG_TUD_ENDPOINT0_SIZE,

    .idVendor           = USB_VID,
    .idProduct          = USB_PID,
    .bcdDevice          = 0x0100,

    .iManufacturer      = 0x01,
    .iProduct           = 0x02,
    .iSerialNumber      = 0x03,

    .bNumConfigurations = 0x01
};

uint8_t const *tud_descriptor_device_cb(void) {
    return (uint8_t const *)&desc_device;
}

// CDC uses 2 interfaces, 3 endpoints (notif EP + bulk RX/TX)
#define CONFIG_TOTAL_LEN  (TUD_CONFIG_DESC_LEN + TUD_CDC_DESC_LEN)
#define EP_CDC_NOTIF   0x81
#define EP_CDC_OUT     0x02
#define EP_CDC_IN      0x82

uint8_t const desc_config[] = {
    TUD_CONFIG_DESCRIPTOR(1, 2, 0, CONFIG_TOTAL_LEN, 0x00, 100),
    TUD_CDC_DESCRIPTOR(0, 4, EP_CDC_NOTIF, 8, EP_CDC_OUT, EP_CDC_IN, 64),
};

uint8_t const *tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_config;
}

// array of pointer to string descriptors
static char const *string_desc_arr[] = {
    (const char[]) { 0x09, 0x04 }, // 0: is supported language is English (0x0409)
    "TinyUSB",                     // 1: Manufacturer
    "TinyUSB Device",              // 2: Product
    NULL,                          // 3: Serials will use unique ID if possible
    "TinyUSB CDC",                 // 4: CDC Interface
    "TinyUSB MSC",                 // 5: MSC Interface
};

static uint16_t _desc_str[32];

uint16_t const *tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;
    uint8_t chr_count;
    if (index == 0) {
        memcpy(&_desc_str[1], string_desc_arr[0], 2);
        chr_count = 1;
    } else {
        if (index >= sizeof(string_desc_arr) / sizeof(string_desc_arr[0])) return NULL;
        const char *str = string_desc_arr[index];
        chr_count = strlen(str);
        for (uint8_t i = 0; i < chr_count; i++)
            _desc_str[1 + i] = str[i];
    }
    _desc_str[0] = (TUSB_DESC_STRING << 8) | (2 * chr_count + 2);
    return _desc_str;
}