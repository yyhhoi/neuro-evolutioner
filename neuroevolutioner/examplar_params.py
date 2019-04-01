

class Params():
    
    # LIF
    u_rest = -70
    u_reset = -70
    u_threshold = -50
    tau_m = 50
    r_m = 10
    
    # Synapses
    E_syn_inhibit = -75
    E_syn_excite = 0
    tau_syn = 50
    g_syn = 400

    # AdEx: Initiaial bursting
    sharpness = 2
    tau_w = 100
    a = 0.5
    b = 7

    step_current = 30

# class Params():
    
#     # LIF
#     u_rest = -70e-3
#     u_reset = -70e-3
#     u_threshold = -50e-3
#     tau_m = 50e-3
#     r_m = 500e6
    
#     # Synapses
#     E_syn_inhibit = -75e-3
#     E_syn_excite = 0
#     tau_syn = 5e-3
#     g_syn = 40e-12

#     # AdEx: Initiaial bursting
#     sharpness = 2e-3
#     tau_w = 100e-3
#     a = 0.5e-9
#     b = 7e-12

#     step_current = 46e-12