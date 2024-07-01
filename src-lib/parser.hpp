#pragma once

//#include "network.hpp"

#ifdef __cplusplus
extern "C" {
#endif
network parse_network_cfg(const char * filename);
network parse_network_cfg_custom(const char * filename, int batch, int time_steps);
void save_network(network net, char *filename);
void save_weights(network net, char *filename);
void save_weights_upto(network net, char *filename, int cutoff, int save_ema);
void save_weights_double(network net, char *filename);
void load_weights(network * net, const char * filename);
void load_weights_upto(network * net, const char * filename, int cutoff);

#ifdef __cplusplus
}
#endif
