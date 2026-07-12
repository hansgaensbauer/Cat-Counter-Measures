#ifndef BOOST_H
#define BOOST_H

#define BOOST_TOP_DAC_VALUE 300

extern volatile char charge_done;

uint16_t read_boost_voltage();
void comparator_init();
void boost_init();
void boost_start_charge();
void boost_stop_charge();

#endif