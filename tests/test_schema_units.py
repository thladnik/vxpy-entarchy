"""Schema-level helpers: hierarchy declaration, metadata flattening, frame timing."""
import os
import numpy as np
import pytest
import yaml

import h5py

import entarchy
from vxentarchy import schema
from vxentarchy.schema import (Animal, ClockDivisionTiming, Experiment,
                                          Imaging, Layer,
                                          Phase, Recording, Roi, VxEntarchy,
                                          ca_frame_times_from_start_pulse,
                                          ca_frame_times_from_toggle,
                                          ca_frame_times_from_y_mirror,
                                          frame_time_methods, unravel_dict)


class TestHierarchyDeclaration:

    def test_resolved_hierarchy(self):
        hierarchy, entity_map = VxEntarchy._declared_hierarchy()

        assert hierarchy['Experiment'] == {'Animal': {
            'Recording': {'Imaging': {'Layer': {'Roi': {}}}, 'Phase': {}}}}
        for name in ('Experiment', 'Animal', 'Recording', 'Imaging', 'Layer',
                     'Roi', 'Phase'):
            assert name in entity_map

    def test_child_entity_types(self):
        assert Experiment.get_child_entity_types() == [Animal]
        assert Animal.get_child_entity_types() == [Recording]
        assert Recording.get_child_entity_types() == [Imaging, Phase]
        assert Imaging.get_child_entity_types() == [Layer]
        assert Layer.get_child_entity_types() == [Roi]
        assert Roi.get_child_entity_types() is None

    def test_collection_types_are_registered(self):
        from vxentarchy.schema import (AnimalCollection,
                                                  ExperimentCollection,
                                                  ImagingCollection,
                                                  LayerCollection, PhaseCollection,
                                                  RecordingCollection, RoiCollection)
        assert Experiment.get_collection_type() is ExperimentCollection
        assert Animal.get_collection_type() is AnimalCollection
        assert Recording.get_collection_type() is RecordingCollection
        assert Layer.get_collection_type() is LayerCollection
        assert Roi.get_collection_type() is RoiCollection
        assert Phase.get_collection_type() is PhaseCollection
        assert Imaging.get_collection_type() is ImagingCollection

    def test_entities_subclass_entarchy_entity(self):
        for entity_type in (Animal, Recording, Imaging, Layer, Roi, Phase):
            assert issubclass(entity_type, entarchy.Entity)


class TestUnravelDict:

    class _Recorder(dict):
        """Stands in for an Entity - only __setitem__ is used."""

    def test_flat_dict(self):
        target = self._Recorder()
        unravel_dict({'a': 1, 'b': 'x'}, target, 'meta')
        assert target == {'meta/a': 1, 'meta/b': 'x'}

    def test_nested_dict_is_flattened_with_slashes(self):
        target = self._Recorder()
        unravel_dict({'outer': {'inner': {'leaf': 5}}, 'top': 1}, target, 'meta')
        assert target == {'meta/outer/inner/leaf': 5, 'meta/top': 1}

    def test_empty_dict_writes_nothing(self):
        target = self._Recorder()
        unravel_dict({}, target, 'meta')
        assert target == {}

    def test_empty_nested_dict_writes_nothing(self):
        target = self._Recorder()
        unravel_dict({'empty': {}}, target, 'meta')
        assert target == {}

    def test_values_are_passed_through_unchanged(self):
        target = self._Recorder()
        array = np.arange(3)
        unravel_dict({'arr': array, 'none': None}, target, 's2p')
        assert target['s2p/none'] is None
        assert np.array_equal(target['s2p/arr'], array)


class TestAddMetadata:

    def test_reads_and_flattens_metadata_files(self, tmp_path):
        (tmp_path / 'rec_metadata.yaml').write_text(
            yaml.safe_dump({'strain': 'wt', 'nested': {'age': 5}}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))

        assert target['metadata/strain'] == 'wt'
        assert target['metadata/nested/age'] == 5

    def test_merges_multiple_files(self, tmp_path):
        (tmp_path / 'a_metadata.yaml').write_text(yaml.safe_dump({'a': 1}))
        (tmp_path / 'b_metadata.yaml').write_text(yaml.safe_dump({'b': 2}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))

        assert target['metadata/a'] == 1
        assert target['metadata/b'] == 2

    def test_no_metadata_files_is_not_an_error(self, tmp_path):
        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))
        assert target == {}

    def test_ignores_unrelated_yaml(self, tmp_path):
        (tmp_path / 'config.yaml').write_text(yaml.safe_dump({'ignored': True}))

        target = TestUnravelDict._Recorder()
        schema.add_metadata(target, str(tmp_path))
        assert target == {}


def triangle_mirror_signal(frequency=10.0, duration=2.0, sample_rate=1000.0):
    """Galvo mirror trace that starts falling, so a trough precedes the first peak."""
    scipy_signal = pytest.importorskip('scipy.signal')
    times = np.arange(0, duration, 1 / sample_rate)
    position = scipy_signal.sawtooth(2 * np.pi * frequency * times - np.pi / 2, width=0.5)
    return position, times


class TestFrameTimesFromYMirror:

    def test_detects_every_half_cycle(self):
        position, times = triangle_mirror_signal(frequency=10.0, duration=2.0)
        indices, frame_times = ca_frame_times_from_y_mirror(position, times)

        # 20 troughs and 20 peaks over 2 s at 10 Hz; the first trough is dropped
        assert len(frame_times) == 39
        assert len(indices) == len(frame_times)

    def test_frame_times_are_monotonic(self):
        position, times = triangle_mirror_signal()
        _, frame_times = ca_frame_times_from_y_mirror(position, times)
        assert np.all(np.diff(frame_times) > 0)

    def test_intervals_are_half_a_cycle(self):
        position, times = triangle_mirror_signal(frequency=10.0, duration=2.0)
        _, frame_times = ca_frame_times_from_y_mirror(position, times)

        intervals = np.diff(frame_times)
        assert np.allclose(intervals, 0.05, atol=2e-3)

    def test_indices_reference_the_source_signal(self):
        position, times = triangle_mirror_signal()
        indices, frame_times = ca_frame_times_from_y_mirror(position, times)
        assert np.array_equal(times[indices], frame_times)

    def test_amplitude_independent(self):
        position, times = triangle_mirror_signal()
        _, small = ca_frame_times_from_y_mirror(position * 0.01, times)
        _, large = ca_frame_times_from_y_mirror(position * 100, times)
        assert np.array_equal(small, large)

    def test_offset_independent(self):
        position, times = triangle_mirror_signal()
        _, centred = ca_frame_times_from_y_mirror(position, times)
        _, offset = ca_frame_times_from_y_mirror(position + 5.0, times)
        assert np.array_equal(centred, offset)


class TestFrameTimesFromStartPulse:
    """A line that pulses once per frame: the rise is the frame start."""

    def test_detects_rising_edges(self):
        """The frame time is the first sample recorded high, not the last one
        recorded low: the line goes high when the frame starts."""
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
        times = np.arange(len(signal), dtype=float)

        indices, frame_times = ca_frame_times_from_start_pulse(signal, times)
        assert np.array_equal(frame_times, [2.0, 6.0])

    def test_ignores_falling_edges(self):
        signal = np.array([1, 1, 0, 0, 0])
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_start_pulse(signal, times)
        assert len(frame_times) == 0

    def test_constant_signal_yields_no_frames(self):
        times = np.arange(10, dtype=float)
        _, frame_times = ca_frame_times_from_start_pulse(np.ones(10), times)
        assert len(frame_times) == 0

    def test_a_boolean_line_counts_each_frame_once(self):
        """vxpy records a digital line as bool, and np.diff of a bool array is
        XOR - so without widening, every fall reads as a rise and the frame
        count comes out doubled."""
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_start_pulse(signal, times)
        assert np.array_equal(frame_times, [2.0, 6.0])

    def test_a_boolean_line_matches_an_integer_one(self):
        rng = np.random.default_rng(0)
        signal = rng.integers(0, 2, size=500).astype(bool)
        times = np.arange(len(signal), dtype=float)

        _, from_bool = ca_frame_times_from_start_pulse(signal, times)
        _, from_int = ca_frame_times_from_start_pulse(signal.astype(np.int8), times)
        assert np.array_equal(from_bool, from_int)


class TestFrameTimesFromToggle:
    """A line that flips state at each frame start: the level means nothing,
    the change does."""

    def test_every_flip_is_a_frame(self):
        signal = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0])
        times = np.arange(len(signal), dtype=float)

        indices, frame_times = ca_frame_times_from_toggle(signal, times)
        assert np.array_equal(indices, [2, 4, 6, 8])
        assert np.array_equal(frame_times, [2.0, 4.0, 6.0, 8.0])

    def test_a_fall_counts_as_a_rise_does(self):
        signal = np.array([1, 1, 0, 0, 0])
        times = np.arange(len(signal), dtype=float)

        _, frame_times = ca_frame_times_from_toggle(signal, times)
        assert np.array_equal(frame_times, [2.0])

    def test_constant_signal_yields_no_frames(self):
        times = np.arange(10, dtype=float)
        _, frame_times = ca_frame_times_from_toggle(np.ones(10), times)
        assert len(frame_times) == 0

    def test_a_boolean_line_matches_an_integer_one(self):
        rng = np.random.default_rng(0)
        signal = rng.integers(0, 2, size=500).astype(bool)
        times = np.arange(len(signal), dtype=float)

        _, from_bool = ca_frame_times_from_toggle(signal, times)
        _, from_int = ca_frame_times_from_toggle(signal.astype(np.int8), times)
        assert np.array_equal(from_bool, from_int)

    def test_it_is_not_the_start_pulse_reading(self):
        """The same line read both ways: a toggle has a frame on every edge, a
        start pulse only on the rises - which is why the metadata has to say
        which of the two the rig recorded."""
        signal = np.tile([0, 0, 1, 1], 10)
        times = np.arange(len(signal), dtype=float)

        _, toggled = ca_frame_times_from_toggle(signal, times)
        _, pulsed = ca_frame_times_from_start_pulse(signal, times)
        assert len(toggled) == 19
        assert len(pulsed) == 10
        assert set(pulsed) < set(toggled)


def _sync_line(tmp_path, edge_num: int, samples_per_period: int = 40,
               sample_rate: float = 600.0, signal: str = 'di_frame_sync'):
    """An Io.hdf5 holding a uniform square wave, as a free-running clock is.

    Built from sample counts rather than a modulo of the times, so that the
    number of rising edges is exactly `edge_num` and the period is exact.
    At the defaults that is a 15 Hz clock sampled at 600 Hz.
    """
    samples = (edge_num + 1) * samples_per_period
    times = np.arange(samples) / sample_rate
    line = np.zeros(samples, dtype=bool)

    # Start one sample in, so the first rise is an edge rather than the
    #  recording already having begun high
    for index in range(edge_num):
        start = 1 + index * samples_per_period
        line[start:start + samples_per_period // 2] = True

    with h5py.File(tmp_path / 'Io.hdf5', 'w') as h5file:
        h5file.create_dataset(signal, data=line[:, None])
        h5file.create_dataset(f'{signal}_time', data=times[:, None])

    return str(tmp_path)


class TestClockDivisionTiming:
    """A sync line that ticks faster than frames arrive, and not by a whole
    number - the tailtracking setup free-runs at 7.5 ticks per volume."""

    def test_divides_the_clock(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=150)
        timing = ClockDivisionTiming(7.5)

        times = timing.frame_times(path, layer_num=5)

        assert len(times) == 5
        # 149 gaps between 150 edges, at 7.5 per volume, plus the first volume
        assert len(times[0]) == 20

    def test_a_fractional_ratio_does_not_drift(self, tmp_path):
        """Interpolating along the edges keeps the frames on the clock; a
        nominal rate would accumulate error over a long recording."""
        path = _sync_line(tmp_path, edge_num=1500)
        timing = ClockDivisionTiming(7.5)

        times = timing.frame_times(path, layer_num=5)[0]

        gaps = np.diff(times)
        assert np.allclose(gaps, gaps[0], atol=1e-3)
        # 7.5 ticks of a 15 Hz clock is half a second
        assert gaps[0] == pytest.approx(0.5, abs=1e-3)

    def test_layers_are_offset_within_the_volume(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=150)

        times = ClockDivisionTiming(7.5).frame_times(path, layer_num=5)

        offsets = [t[0] - times[0][0] for t in times]
        assert offsets[0] == 0
        assert np.allclose(np.diff(offsets), 0.5 / 5, atol=1e-3)

    def test_a_whole_number_ratio_works_too(self, tmp_path):
        path = _sync_line(tmp_path, edge_num=100)

        times = ClockDivisionTiming(4).frame_times(path, layer_num=1)[0]

        assert len(times) == 25
        assert np.allclose(np.diff(times), 4 / 15, atol=1e-3)

    def test_a_silent_line_is_refused(self, tmp_path):
        with h5py.File(tmp_path / 'Io.hdf5', 'w') as h5file:
            h5file.create_dataset('di_frame_sync', data=np.zeros((100, 1), dtype=bool))
            h5file.create_dataset('di_frame_sync_time',
                                  data=np.arange(100, dtype=float)[:, None])

        with pytest.raises(ValueError, match='cannot time anything'):
            ClockDivisionTiming(7.5).frame_times(str(tmp_path), layer_num=1)

    def test_a_nonsense_ratio_is_refused(self):
        with pytest.raises(ValueError, match='must be positive'):
            ClockDivisionTiming(0)


class TestLongPath:
    """Windows stops opening files past 260 characters, and registration output
    nests two long image names, so it lands there."""

    def test_a_short_path_is_left_alone(self, tmp_path):
        assert not schema.long_path(str(tmp_path)).startswith('\\\\?\\')

    def test_a_long_path_is_marked_on_windows(self, tmp_path):
        deep = tmp_path.joinpath(*['a_reasonably_long_folder_name'] * 12)

        marked = schema.long_path(str(deep))

        if os.name == 'nt':
            assert marked.startswith('\\\\?\\')
            assert marked.endswith(str(deep).replace('/', '\\'))
        else:
            assert marked == str(deep)

    @pytest.mark.skipif(os.name != 'nt', reason='the limit is a Windows one')
    def test_a_file_past_the_limit_can_be_opened(self, tmp_path):
        deep = tmp_path.joinpath(*['a_reasonably_long_folder_name'] * 12)
        os.makedirs(schema.long_path(str(deep)), exist_ok=True)

        target = os.path.join(str(deep), 'metadata.yaml')
        assert len(target) > 260

        with open(schema.long_path(target), 'w') as f:
            f.write('written: true\n')

        # Which is exactly what fails without it
        with pytest.raises(OSError):
            open(target)

        with open(schema.long_path(target)) as f:
            assert yaml.safe_load(f) == {'written': True}

    def test_forward_slashes_are_resolved(self, tmp_path):
        """The marked form is handed to the filesystem unnormalised, so it has
        to be an absolute path with no forward slashes left in it."""
        deep = '/'.join([str(tmp_path)] + ['a_reasonably_long_folder_name'] * 12)

        marked = schema.long_path(deep)

        if os.name == 'nt':
            assert '/' not in marked


class TestFrameTimeMethodRegistry:

    def test_registered_methods(self):
        assert set(frame_time_methods) == {'ymirror', 'toggle', 'startpulse'}
        assert frame_time_methods['ymirror'] is ca_frame_times_from_y_mirror
        assert frame_time_methods['toggle'] is ca_frame_times_from_toggle
        assert frame_time_methods['startpulse'] is ca_frame_times_from_start_pulse

    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            frame_time_methods['does_not_exist']


class TestMissingSyncChannel:
    """A timing that reads a channel the rig did not record says so usefully.

    Metadata that names the galvo mirror on a rig that records a digital frame
    clock instead used to fail with h5py's bare "object doesn't exist".
    """

    @staticmethod
    def write_io(tmp_path, **channels):
        import h5py
        with h5py.File(tmp_path / 'Io.hdf5', 'w') as io_file:
            for name, data in channels.items():
                io_file.create_dataset(name, data=data)
        return tmp_path.as_posix()

    def test_the_error_names_the_channels_present(self, tmp_path):
        from vxentarchy.schema import SyncSignalTiming
        path = self.write_io(tmp_path, di_frame_sync=np.zeros((10, 1), dtype=bool),
                             di_frame_sync_time=np.arange(10.0)[:, None], ai_other=np.zeros(10))

        with pytest.raises(KeyError, match='ai_y_mirror_in.*di_frame_sync, di_frame_sync_time'):
            SyncSignalTiming('ymirror', 'ai_y_mirror_in').frame_times(path, 1)

    def test_the_error_points_at_the_metadata(self, tmp_path):
        from vxentarchy.schema import SyncSignalTiming
        path = self.write_io(tmp_path, di_frame_sync=np.zeros((10, 1), dtype=bool),
                             di_frame_sync_time=np.arange(10.0)[:, None])

        with pytest.raises(KeyError, match='metadata.yaml.*frame_sync_signal_type'):
            SyncSignalTiming('ymirror', 'ai_y_mirror_in').frame_times(path, 1)

    def test_the_other_way_round(self, tmp_path):
        from vxentarchy.schema import ClockDivisionTiming
        path = self.write_io(tmp_path, ai_y_mirror_in=np.zeros(10), ai_y_mirror_in_time=np.arange(10.0))

        with pytest.raises(KeyError, match='di_frame_sync.*ai_y_mirror_in'):
            ClockDivisionTiming(7.5).frame_times(path, 1)

    def test_a_present_channel_reads_as_before(self, tmp_path):
        from vxentarchy.schema import read_io_channel
        path = self.write_io(tmp_path, di_frame_sync=np.array([[0], [1], [1], [0]], dtype=bool),
                             di_frame_sync_time=np.arange(4.0)[:, None])

        signal, times = read_io_channel(path, 'di_frame_sync', 'di_frame_sync_time')
        assert signal.shape == (4,) and times.shape == (4,)
        assert signal.tolist() == [False, True, True, False]


class TestSyncSignalTimingHasNoDefault:
    """Which line a rig recorded is written down, never assumed."""

    def test_method_and_signal_are_required(self):
        from vxentarchy.schema import SyncSignalTiming

        with pytest.raises(TypeError):
            SyncSignalTiming()

    def test_an_unknown_method_names_the_known_ones(self):
        from vxentarchy.schema import SyncSignalTiming

        with pytest.raises(ValueError, match='startpulse, toggle, ymirror'):
            SyncSignalTiming('y_mirror', 'ai_y_mirror_in')

    def test_the_time_channel_follows_the_signal_unless_named(self):
        from vxentarchy.schema import SyncSignalTiming

        assert SyncSignalTiming('toggle', 'di_frame_sync').signal_time == 'di_frame_sync_time'
        assert SyncSignalTiming('toggle', 'di_frame_sync', 'clock').signal_time == 'clock'


class TestTimingFromMetadata:
    """What a recording folder's metadata.yaml says about its frame timing."""

    METADATA = {'frame_sync_signal_type': 'ymirror',
                'frame_sync_signal': 'ai_y_mirror_in',
                'frame_sync_signal_timing': 'ai_y_mirror_in_time'}

    def test_metadata_that_says_nothing_gives_none(self):
        assert schema.timing_from_metadata({'repeat_num': 2}) is None

    @pytest.mark.parametrize('kind', ['ymirror', 'toggle', 'startpulse'])
    def test_the_signal_types(self, kind):
        timing = schema.timing_from_metadata({'frame_sync_signal_type': kind,
                                              'frame_sync_signal': 'di_frame_sync',
                                              'frame_sync_signal_timing': 'di_frame_sync_time'})

        assert isinstance(timing, schema.SyncSignalTiming)
        assert timing.config == {'method': kind, 'signal': 'di_frame_sync',
                                 'signal_time': 'di_frame_sync_time', 'frame_avg_num': 1}

    def test_the_time_points_are_the_ones_named(self):
        timing = schema.timing_from_metadata({**self.METADATA,
                                              'frame_sync_signal_timing': 'some_clock'})

        assert timing.signal_time == 'some_clock'

    def test_frame_avg_num_is_optional(self):
        timing = schema.timing_from_metadata({**self.METADATA, 'frame_avg_num': 4})

        assert timing.frame_avg_num == 4

    def test_the_type_is_read_without_regard_to_case(self):
        timing = schema.timing_from_metadata({**self.METADATA,
                                              'frame_sync_signal_type': 'YMirror'})

        assert timing.method == 'ymirror'

    def test_an_unknown_type_names_the_known_ones(self):
        with pytest.raises(ValueError, match='unknown frame_sync_signal_type "galvo".*'
                                             'startpulse, toggle, ymirror'):
            schema.timing_from_metadata({**self.METADATA,
                                         'frame_sync_signal_type': 'galvo'}, '/rec')

    def test_a_type_without_its_signal_is_an_error_not_a_silence(self):
        with pytest.raises(ValueError, match='frame_sync_signal and frame_sync_signal_timing'):
            schema.timing_from_metadata({'frame_sync_signal_type': 'toggle'}, '/rec')

    def test_a_type_without_its_time_points(self):
        with pytest.raises(ValueError, match='no frame_sync_signal_timing'):
            schema.timing_from_metadata({'frame_sync_signal_type': 'toggle',
                                         'frame_sync_signal': 'di_frame_sync'}, '/rec')

    def test_a_divided_clock_can_be_written_down_too(self):
        timing = schema.timing_from_metadata({'frame_sync_signal_type': 'clockdivision',
                                              'frame_sync_signal': 'di_frame_sync',
                                              'frame_sync_signal_timing': 'di_frame_sync_time',
                                              'frame_sync_edges_per_volume': 7.5})

        assert isinstance(timing, ClockDivisionTiming)
        assert timing.config == {'edges_per_volume': 7.5, 'signal': 'di_frame_sync',
                                 'signal_time': 'di_frame_sync_time'}

    def test_a_divided_clock_needs_its_ratio(self):
        with pytest.raises(ValueError, match='frame_sync_edges_per_volume'):
            schema.timing_from_metadata({'frame_sync_signal_type': 'clockdivision',
                                         'frame_sync_signal': 'di_frame_sync',
                                         'frame_sync_signal_timing': 'di_frame_sync_time'})


class TestReadMetadata:

    def test_a_folder_without_metadata_reads_as_empty(self, tmp_path):
        assert schema.read_metadata(tmp_path.as_posix()) == {}
        assert schema.metadata_files_in(tmp_path.as_posix()) == []

    def test_the_files_are_merged(self, tmp_path):
        with open(tmp_path / 'metadata.yaml', 'w') as f:
            yaml.safe_dump({'frame_sync_signal_type': 'toggle'}, f)
        with open(tmp_path / 'recording_metadata.yaml', 'w') as f:
            yaml.safe_dump({'repeat_num': 2}, f)
        (tmp_path / 'notes.txt').write_text('not metadata')

        assert schema.read_metadata(tmp_path.as_posix()) == {
            'frame_sync_signal_type': 'toggle', 'repeat_num': 2}
        assert schema.metadata_files_in(tmp_path.as_posix()) == [
            'metadata.yaml', 'recording_metadata.yaml']

    def test_an_empty_file_contributes_nothing(self, tmp_path):
        (tmp_path / 'metadata.yaml').write_text('')

        assert schema.read_metadata(tmp_path.as_posix()) == {}
