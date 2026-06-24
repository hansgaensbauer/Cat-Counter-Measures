// src/conf_clocks.h
#ifndef CONF_CLOCKS_H
#define CONF_CLOCKS_H

#include <clock.h>

// Keep ASF from touching DFLL — we configure it manually
#define CONF_CLOCK_DFLL_ENABLE                  false

// GCLK0 — ASF will set this to OSC8M/1 as a safe default
// We switch it to DFLL48M ourselves after system_init()
#define CONF_CLOCK_GCLK_0_ENABLE                true
#define CONF_CLOCK_GCLK_0_CLOCK_SOURCE          SYSTEM_CLOCK_SOURCE_OSC8M
#define CONF_CLOCK_GCLK_0_PRESCALER             1
#define CONF_CLOCK_GCLK_0_OUTPUT_ENABLE         false

// Disable everything else we don't use
#define CONF_CLOCK_GCLK_1_ENABLE                false
#define CONF_CLOCK_GCLK_2_ENABLE                false
#define CONF_CLOCK_GCLK_3_ENABLE                false
#define CONF_CLOCK_GCLK_4_ENABLE                false
#define CONF_CLOCK_GCLK_5_ENABLE                false
#define CONF_CLOCK_GCLK_6_ENABLE                false
#define CONF_CLOCK_GCLK_7_ENABLE                false
#define CONF_CLOCK_GCLK_8_ENABLE                false

#endif