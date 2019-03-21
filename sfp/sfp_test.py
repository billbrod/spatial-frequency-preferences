import shutil
import pytest
from . import stimuli as sfp_stimuli
import numpy as np


def test_mkr():
    sfp_stimuli.mkR(50)


def test_mkangle():
    sfp_stimuli.mkAngle(50)


def test_log_polar_grating():
    sfp_stimuli.log_polar_grating(1080, 6)


def test_check_aliasing():
    sfp_stimuli.check_aliasing(100, 0, 6, check_scale_factor=11)


def test_check_aliasing_with_mask():
    sfp_stimuli.check_aliasing_with_mask(100, 0, 6, check_scale_factor=11)


def test_gen_log_polar_stim_set():
    sfp_stimuli.gen_log_polar_stim_set(32, freqs_ra=[(4, 0), (0, 4), (4, 4)],
                                       phi=np.array(range(8))/8.*2*np.pi)
    sfp_stimuli.gen_log_polar_stim_set(1080, [(0, 128)])


def test_gen_constant_stim_set():
    _, mask = sfp_stimuli.create_antialiasing_mask(32)
    sfp_stimuli.gen_constant_stim_set(32, mask, freqs_xy=[(.004, 0), (0, .004), (.004, .004)],
                                      phi=np.array(range(8))/8.*2*np.pi)
    _, mask = sfp_stimuli.create_antialiasing_mask(1080)
    sfp_stimuli.gen_constant_stim_set(1080, mask, [(0, .1)])


def test_check_stim_properties():
    mask_df, sf_df = sfp_stimuli.check_stim_properties(1080, None, 24, w_r=range(0, 128, 20),
                                                       w_a=range(0, 128, 10))
    sfp_stimuli.plot_stim_properties(mask_df, size=8, data_label='mask_radius_pix',
                                     title_text='Mask radius in degrees')
    sfp_stimuli.plot_stim_properties(mask_df, data_label='cpp_masked_max',
                                     title_text="Max masked frequency in cpp", size=8)


def test_create_sf_maps():
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_r=6, w_a=6)
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_r=6, w_a=0)
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_r=6, w_a=6, stim_type='pilot')
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_r=6, w_a=0, stim_type='pilot')
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_x=.1, w_y=.5, stim_type='constant')
    sfp_stimuli.create_sf_maps_cpd(1080, 12, w_x=.1, w_y=.1, stim_type='constant')
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_r=6, w_a=6)
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_r=6, w_a=0)
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_r=6, w_a=6, stim_type='pilot')
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_r=6, w_a=0, stim_type='pilot')
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_x=.1, w_y=.5, stim_type='constant')
    sfp_stimuli.create_sf_origin_polar_maps_cpd(1080, 12, w_x=.1, w_y=.1, stim_type='constant')

# THESE REQUIRE MORE MEMORY THAN TRAVIS CI CAN GIVE US

# def test_stim_main():
#     sfp_stimuli.main('test', "data/test/")
#     shutil.rmtree('data/test')


# def test_stim_main_exception():
#     sfp_stimuli.main('test', "data/test/")
#     with pytest.raises(Exception):
#         sfp_stimuli.main('test', "data/test/", create_stim=False)
#     with pytest.raises(Exception):
#         sfp_stimuli.main('test', "data/test/", create_idx=False)
#     shutil.rmtree('data/test')
