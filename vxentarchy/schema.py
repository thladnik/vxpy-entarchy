from __future__ import annotations
import os
import pathlib
import time
import traceback
from typing import TypeVar, cast, overload

import h5py
import numpy as np
import pandas as pd
import scipy
import tifffile
import tqdm
import yaml

import entarchy


__all__ = ['VxEntarchy',
           'Experiment', 'Animal', 'Recording', 'Imaging', 'Layer', 'Roi', 'Phase',
           'ExperimentCollection', 'AnimalCollection', 'RecordingCollection',
           'ImagingCollection', 'LayerCollection', 'RoiCollection', 'PhaseCollection',
           'ImagingSource', 'Suite2pSource', 'ImagingSpec', 'imaging_sources',
           'FrameTiming', 'SyncSignalTiming', 'ClockDivisionTiming', 'CameraTiming',
           'is_recording_folder', 'scan_experiment']


# Each names the entity it holds, so that indexing and iterating one gives that
#  entity rather than the base class - recording.phases[0] is a Phase, and a
#  Phase's own properties complete. The names are quoted because every
#  collection here is declared before the entity it holds.


class ExperimentCollection(entarchy.Collection['Experiment']):
    pass


class AnimalCollection(entarchy.Collection['Animal']):
    pass


class RecordingCollection(entarchy.Collection['Recording']):
    pass


class LayerCollection(entarchy.Collection['Layer']):
    pass


class RoiCollection(entarchy.Collection['Roi']):
    pass


class PhaseCollection(entarchy.Collection['Phase']):
    pass


class ImagingCollection(entarchy.Collection['Imaging']):

    @overload
    def __getitem__(self, item: str) -> 'Imaging': ...

    @overload
    def __getitem__(self, item: int) -> 'Imaging': ...

    @overload
    def __getitem__(self, item: slice) -> list['Imaging']: ...

    @overload
    def __getitem__(self, item: list) -> pd.DataFrame: ...

    def __getitem__(self, item):  # type: ignore[override]
        """Also addressable by source name, since that is what an Imaging is.

        Which is why the overloads are spelled out here rather than inherited:
        on any other collection a string names an attribute and gives its
        values, and on this one it names a source and gives that source.

        A type checker is right to call that an incompatible override, and the
        ignore above is the record of it being deliberate. The way to be rid of
        it would be a named method - imaging.by_source('ca') - leaving the
        brackets meaning what they mean everywhere else.
        """
        if isinstance(item, str):
            matching = [imaging for imaging in self if imaging.id == item]
            if len(matching) == 0:
                available = ', '.join(sorted(imaging.id for imaging in self)) or 'none'
                raise KeyError(f'No imaging source "{item}". Available: {available}.')
            return matching[0]

        return super().__getitem__(item)


class Experiment(entarchy.Entity):
    """One experiment: a set of animals recorded under one protocol.

    The raw tree has always had this level - `<experiment>/<animal>/<recording>`
    - and the ingest used to flatten it onto a string attribute. Two things
    make it worth an entity of its own. Which animals belong together is a
    fact about the data rather than a label, so `[Experiment]id == "cmn"`
    reads like every other ancestor query here. And how frames were timed is
    chosen per experiment and cannot be detected, so an Experiment is where
    that choice is recorded, under `imaging/`.
    """

    collection_type = ExperimentCollection

    @property
    def animals(self) -> AnimalCollection:
        return self.get(Animal)  # type: ignore[return-value]

    @property
    def recordings(self) -> RecordingCollection:
        return self.get(Recording)  # type: ignore[return-value]

    @property
    def phases(self) -> PhaseCollection:
        return self.get(Phase)  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.get(Layer)  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.get(Roi)  # type: ignore[return-value]


class Animal(entarchy.Entity):
    collection_type = AnimalCollection

    @property
    def experiment(self) -> Experiment:
        return self.parent  # type: ignore[return-value]

    @property
    def recordings(self) -> RecordingCollection:
        return self.get(Recording)  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.get(Layer)  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.get(Roi)  # type: ignore[return-value]


class Recording(entarchy.Entity):
    collection_type = RecordingCollection
    # # Child entity types may be added by name meta // Does not work yet
    # _child_entity_types = [Roi, Phase]

    @property
    def animal(self) -> Animal:
        return self.parent  # type: ignore[return-value]

    @property
    def phases(self) -> PhaseCollection:
        return self.get(Phase)  # type: ignore[return-value]

    @property
    def imaging(self) -> ImagingCollection:
        """The imaging sources of this recording. May be empty."""
        return self.get(Imaging)  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.get(Layer)  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.get(Roi)  # type: ignore[return-value]

    def sole_imaging(self) -> 'Imaging':
        """The one imaging source, when asking without naming one is unambiguous."""
        sources = list(self.imaging)
        if len(sources) == 1:
            return sources[0]

        if len(sources) == 0:
            raise LookupError(f'{self} has no imaging data.')

        names = ', '.join(sorted(source.id for source in sources))
        raise LookupError(f'{self} has several imaging sources ({names}); '
                          f'say which one.')


class Imaging(entarchy.Entity):
    """One imaging source of a recording: an acquisition and what read it.

    Its id is the source name - `suite2p`, `caiman`, `widefield` - so a
    recording processed twice holds two of these rather than two sets of
    colliding plane ids. Everything about when frames happened and what
    produced the signals lives here rather than on the Recording, which knows
    only what the acquisition software wrote.
    """
    collection_type = ImagingCollection

    @property
    def recording(self) -> 'Recording':
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> 'Animal':
        return self.recording.animal  # type: ignore[return-value]

    @property
    def layers(self) -> LayerCollection:
        return self.get(Layer)  # type: ignore[return-value]

    @property
    def rois(self) -> RoiCollection:
        return self.get(Roi)  # type: ignore[return-value]


class Phase(entarchy.Entity):
    collection_type = PhaseCollection

    @property
    def recording(self) -> Recording:
        return self.parent  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.parent  # type: ignore[return-value]

    def frames_in(self, imaging: 'Imaging' = None):
        """Which frames of an imaging source fall inside this phase.

        The window belongs to the pair, not to the phase: two sources of one
        recording sample it differently. It is stored as a link for that reason,
        and this hides that. Omit `imaging` when the recording has one source.

        Returns:
            (start_index, end_index), or None if the phase covers no frames.
        """
        imaging = self.recording.sole_imaging() if imaging is None else imaging

        link = self.entarchy.get_link(self, imaging, PHASE_FRAMES_LINK)
        if link is None:
            return None

        return int(link['start_index']), int(link['end_index'])


class Layer(entarchy.Entity):
    collection_type = LayerCollection

    @property
    def rois(self) -> RoiCollection:
        return self.get(Roi)  # type: ignore[return-value]
        # return get_collection_as[RoiCollection](self.schema, Roi, f'[Recording]uuid == "{self.uuid}"')

    @property
    def imaging(self) -> Imaging:
        return self.parent  # type: ignore[return-value]

    @property
    def recording(self) -> Recording:
        return self.imaging.recording  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


class Roi(entarchy.Entity):
    collection_type = RoiCollection

    @property
    def layer(self) -> Layer:
        return self.parent  # type: ignore[return-value]

    @property
    def imaging(self) -> Imaging:
        return self.layer.imaging  # type: ignore[return-value]

    @property
    def recording(self) -> Recording:
        return self.layer.recording  # type: ignore[return-value]

    @property
    def animal(self) -> Animal:
        return self.recording.animal  # type: ignore[return-value]


# Establish hierarchy
Experiment.add_child_entity_type(Animal)
Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Imaging)
Imaging.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)
Recording.add_child_entity_type(Phase)

# Which frames of an imaging source a stimulation phase covers. A link, because
#  the window belongs to the pair: the same phase falls on different frames of
#  two sources that sampled it differently.
PHASE_FRAMES_LINK = 'phase_frames' 


class FrameTiming:
    """When each imaging frame was acquired, on the recording's timebase.

    Separate from the source that reads the signals, because the two are
    independent choices: suite2p output may be timed from a galvo mirror, a
    frame sync toggle or a timestamps file, and the same timing may serve
    suite2p and CaImAn alike.
    """
    name: str = 'timing'

    def frame_times(self, path: str, layer_num: int) -> list:
        """One array of frame times per layer, in acquisition order."""
        raise NotImplementedError

    @property
    def config(self) -> dict:
        """What this timing was configured with.

        Frame timing is a choice rather than something readable off the data -
        the ratio a divided clock is read at was measured by hand - so an
        ingest writes this onto the Experiment. Every subclass keeps its
        configuration in plain attributes named after its arguments, which is
        all this collects.
        """
        return {name: value for name, value in vars(self).items()
                if not name.startswith('_')}


def read_io_channel(path: str, signal: str, signal_time: str) -> tuple[np.ndarray, np.ndarray]:
    """A channel of Io.hdf5 and its timestamps, each squeezed to one dimension.

    A channel that is not there is how a frame timing shows itself to be the
    wrong one for a rig: a recording without a galvo trace cannot be timed from
    one. So the error says what the file does have, and that the timing is a
    choice to pass rather than a defect of the file.
    """
    io_path = os.path.join(path, 'Io.hdf5')
    with h5py.File(io_path, 'r') as io_file:
        missing = [name for name in (signal, signal_time) if name not in io_file]
        if missing:
            present = sorted(name for name in io_file.keys()
                             if 'sync' in name.lower() or 'mirror' in name.lower())
            raise KeyError(
                f'{io_path} has no channel {", ".join(missing)}. Channels in it that '
                f'look like a frame sync: {", ".join(present) if present else "none"}. '
                f'How frames were timed is not something the file says: the recording '
                f'folder\'s metadata.yaml names it, with {TIMING_TYPE_KEY}, '
                f'{TIMING_SIGNAL_KEY} and {TIMING_SIGNAL_TIME_KEY}.')

        return np.squeeze(io_file[signal][:]), np.squeeze(io_file[signal_time][:])


class SyncSignalTiming(FrameTiming):
    """Frame times recovered from a sync channel of Io.hdf5.

    `method` is how the channel marks a frame - `ymirror`, `toggle` or
    `startpulse`, see `frame_time_methods` - and `signal` the channel. Neither
    has a default: which line a rig recorded is written in the recording
    folder's metadata.yaml, not guessed here.

    A scanner that visits planes in turn puts every layer_num-th frame in the
    same plane, which is what the slicing below undoes. `frame_avg_num` is how
    many frames the scanner averaged into one.
    """
    name = 'sync_signal'

    def __init__(self, method: str, signal: str, signal_time: str = None,
                 frame_avg_num: int = 1):
        if method not in frame_time_methods:
            raise ValueError(f'Unknown frame timing method "{method}". '
                             f'Known: {", ".join(sorted(frame_time_methods))}.')

        self.method = method
        self.signal = signal
        self.signal_time = f'{signal}_time' if signal_time is None else signal_time
        self.frame_avg_num = frame_avg_num

    def all_frame_times(self, path: str) -> np.ndarray:
        sync_data, sync_data_times = read_io_channel(path, self.signal, self.signal_time)

        return frame_time_methods[self.method](sync_data, sync_data_times)[1]

    def frame_times(self, path: str, layer_num: int) -> list:
        all_times = self.all_frame_times(path)
        step = layer_num * self.frame_avg_num

        return [all_times[int(layer_idx + self.frame_avg_num // 2)::step]
                for layer_idx in range(layer_num)]


class ClockDivisionTiming(FrameTiming):
    """Frame times from a sync line that ticks at a fixed multiple of the rate.

    Not every frame sync pulses once per acquired slice. A scanner whose slow
    axis free-runs keeps emitting its frame clock through the piezo flyback, so
    the line ticks evenly for exactly as long as the acquisition lasts while
    only some of those ticks are frames that were kept. Such a line still times
    the frames - it has to be divided rather than counted.

    `edges_per_volume` is that ratio and need not be a whole number, which is
    the point: it is the volume period measured in clock ticks, and a volume
    that is not phase-locked to the clock gives a fraction. Measure it as
    (edges / volumes) on a recording where both are known, then check that it
    reproduces the frame count of the others.
    """
    name = 'clock_division'

    def __init__(self, edges_per_volume: float, signal: str = 'di_frame_sync',
                 signal_time: str = None):
        if edges_per_volume <= 0:
            raise ValueError('edges_per_volume must be positive, '
                             f'got {edges_per_volume}.')

        self.edges_per_volume = float(edges_per_volume)
        self.signal = signal
        self.signal_time = f'{signal}_time' if signal_time is None else signal_time

    def edge_times(self, path: str) -> np.ndarray:
        sync_data, sync_data_times = read_io_channel(path, self.signal, self.signal_time)

        return ca_frame_times_from_start_pulse(sync_data, sync_data_times)[1]

    def frame_times(self, path: str, layer_num: int) -> list:
        edges = self.edge_times(path)
        if len(edges) < 2:
            raise ValueError(f'The sync line "{self.signal}" in {path} has '
                             f'{len(edges)} edges, so it cannot time anything.')

        # Ride the clock rather than a nominal rate, so that a clock which is
        #  slightly off nominal does not accumulate drift across the recording
        volume_num = int((len(edges) - 1) / self.edges_per_volume) + 1
        positions = np.arange(volume_num) * self.edges_per_volume
        times = np.interp(positions, np.arange(len(edges)), edges)

        # Planes are visited in turn within a volume
        slice_dt = np.median(np.diff(times)) / layer_num

        return [times + layer_idx * slice_dt for layer_idx in range(layer_num)]


class CameraTiming(FrameTiming):
    """Frame times the camera itself recorded, for a source that is not scanned.

    One plane, so no demultiplexing: the times are the times.
    """
    name = 'camera'

    def __init__(self, device: str, file_name: str = 'Camera.hdf5'):
        self.device = device
        self.file_name = file_name

    def frame_times(self, path: str, layer_num: int) -> list:
        if layer_num != 1:
            raise ValueError(f'{type(self).__name__} times a single plane, '
                             f'but {layer_num} layers were asked for.')

        with h5py.File(os.path.join(path, self.file_name), 'r') as h5file:
            times = np.squeeze(h5file[f'{self.device}_frame_time'][:])

        return [times]


class ImagingSource:
    """Where signals come from: what is on disk and how to read it.

    An implementation must write the ROI contract - VxEntarchy.ROI_REQUIRED -
    whatever its own vocabulary is. Anything else it knows stays namespaced
    under its own name, as suite2p's stats do under `s2p/`.
    """
    name: str = 'source'

    def detect(self, path: str) -> bool:
        """Whether this source has output in a recording folder."""
        raise NotImplementedError

    def layer_names(self, path: str) -> list:
        """The planes this source produced, in index order."""
        raise NotImplementedError

    def ingest(self, imaging: Imaging, path: str, frame_times_by_layer: list,
               options: dict) -> None:
        """Create the layers and ROIs of this source under `imaging`."""
        raise NotImplementedError


class Suite2pSource(ImagingSource):
    """suite2p's plane directories: ops, F, spks, stat, iscell."""
    name = 'suite2p'

    def detect(self, path: str) -> bool:
        return len(self.layer_names(path)) > 0

    def layer_names(self, path: str) -> list:
        suite2p_path = os.path.join(path, 'suite2p')
        if not os.path.isdir(suite2p_path):
            return []

        names = [name for name in os.listdir(suite2p_path)
                 if name.startswith('plane')
                 and os.path.isdir(os.path.join(suite2p_path, name))]

        return sorted(names, key=lambda name: int(name.replace('plane', '')))

    def ingest(self, imaging: Imaging, path: str, frame_times_by_layer: list,
               options: dict) -> None:
        ent = imaging.entarchy
        layer_names = self.layer_names(path)
        layer_num = len(layer_names)
        dt_frames = np.diff(frame_times_by_layer[0].squeeze()).mean()

        frame_times = frame_times_by_layer[0].squeeze()

        for layer_str in layer_names:

            layer = imaging.add_child(Layer, layer_str)
            print(f'> Process {layer}')

            s2p_path = os.path.join(path, 'suite2p', layer_str)
            layer_idx = int(layer_str.replace('plane', ''))
            frame_times = frame_times_by_layer[layer_idx]

            # Load suite2p's analysis options
            print('>> Include suite2p ops')
            ops = np.load(os.path.join(s2p_path, 'ops.npy'), allow_pickle=True).item()
            unravel_dict(ops, layer, 's2p')

            print('>> Load ROI data')
            fluorescence = np.load(os.path.join(s2p_path, 'F.npy'), allow_pickle=True)
            spikes_all = np.load(os.path.join(s2p_path, 'spks.npy'), allow_pickle=True)
            roi_stats_all = np.load(os.path.join(s2p_path, 'stat.npy'), allow_pickle=True)
            # In some suite2p versions the iscell file may be missing?
            try:
                iscell_all = np.load(os.path.join(s2p_path, 'iscell.npy'), allow_pickle=True)
            except Exception:
                iscell_all = None

            # Check if frame times and signal match
            if frame_times.shape[0] != fluorescence.shape[1]:
                print(f'Detected frame times length does not match frame count. '
                      f'Detected frame times: {frame_times.shape[0]} / Frames: {fluorescence.shape[1]}')

                if frame_times.shape[0] < fluorescence.shape[1]:
                    fluorescence = fluorescence[:, :frame_times.shape[0]]
                    print('Truncated signal at end to resolve mismatch. Check debug output to verify')
                else:
                    frame_times = frame_times[:fluorescence.shape[1]]
                    print('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

            layer['index'] = layer_idx
            layer['roi_num'] = fluorescence.shape[0]
            layer['t_offset'] = layer_idx * dt_frames / layer_num

            roi_coordinates = self._registration_coordinates(path, layer_str)

            print('>> Add ROI stats and signals')
            for roi_idx in tqdm.tqdm(range(fluorescence.shape[0])):
                roi = layer.add_child(Roi, f'Roi_{roi_idx}')
                roi['index'] = roi_idx

                # suite2p's own vocabulary, kept where it cannot be mistaken for
                #  the shared contract
                roi.update({f's2p/{k}': v for k, v in roi_stats_all[roi_idx].items()})

                if roi_coordinates is not None:
                    coords = roi_coordinates.iloc[roi_idx]
                    roi.update({'ants/x': float(coords.x), 'ants/y': float(coords.y),
                                'ants/z': float(coords.z)})

                roi['fluorescence'] = fluorescence[roi_idx]
                roi['spikes'] = spikes_all[roi_idx]

                # suite2p packs the classifier's verdict and its confidence into
                #  one two element row. Split, so that a source which classifies
                #  differently - or not at all - writes the same names.
                if iscell_all is not None:
                    roi['is_unit'] = bool(iscell_all[roi_idx][0])
                    roi['unit_probability'] = float(iscell_all[roi_idx][1])

        imaging['frame_times'] = frame_times
        imaging['frame_num'] = frame_times.shape[0]

    def _registration_coordinates(self, path: str, layer_str: str):
        """ANTs-registered ROI coordinates for a plane, if any were computed.

        Two layouts are in use. Registration run per plane names the plane in
        the path; registration run once over the whole extraction names what was
        registered instead, which is the suite2p folder:

            <rec>/suite2p/ants_registration/<plane>/<reference>/mapped_points.h5
            <rec>/ants_registration/suite2p/<reference>/mapped_points.h5

        The second says nothing about planes, so it is only read when there is
        one - otherwise every plane would be handed the same coordinates.
        """
        roots = [os.path.join(path, 'suite2p', 'ants_registration', layer_str)]
        if len(self.layer_names(path)) == 1:
            roots.append(os.path.join(path, 'ants_registration', 'suite2p'))

        for root in roots:
            if not os.path.isdir(long_path(root)):
                continue

            for reference in sorted(os.listdir(long_path(root))):
                mapped_path = long_path(os.path.join(root, reference,
                                                     'mapped_points.h5'))
                if os.path.exists(mapped_path):
                    print(f'>> ANTs ROI coordinates from {mapped_path}')
                    return pd.read_hdf(mapped_path, key='coordinates')

        print(f'>> No ANTs ROI coordinates for {layer_str}')
        return None


# What a recording folder says about how its frames were timed, in its
#  metadata.yaml. A folder with an imaging source needs the first three; the
#  rest are optional. The type names are the recording's vocabulary - a rig
#  records a mirror, a toggling line or a pulse - and `frame_time_methods`
#  maps them to the computation.
TIMING_TYPE_KEY = 'frame_sync_signal_type'          # ymirror, toggle, startpulse, clockdivision
TIMING_SIGNAL_KEY = 'frame_sync_signal'             # channel of Io.hdf5
TIMING_SIGNAL_TIME_KEY = 'frame_sync_signal_timing'  # its time points
TIMING_EDGES_KEY = 'frame_sync_edges_per_volume'    # clockdivision only
FRAME_AVG_KEY = 'frame_avg_num'                     # optional, default 1
LAYER_NUM_KEY = 'layer_num'                         # optional, checked against the source
CLOCK_DIVISION_TYPE = 'clockdivision'


class MissingFrameTiming(FileNotFoundError):
    """A recording folder has an imaging source, but its metadata does not say
    how the frames were timed. Nothing of the recording is ingested; the digest
    reports it and lists every such folder at the end, and a rerun with the
    metadata in place picks it up."""

    def __init__(self, path: str, metadata_files: list):
        self.path = path
        self.metadata_files = metadata_files
        why = (f'{", ".join(metadata_files)} in it has no {TIMING_TYPE_KEY}'
               if metadata_files else 'it has no metadata.yaml')
        super().__init__(
            f'{path} has an imaging source but no frame timing: {why}. A recording '
            f'with imaging needs {TIMING_TYPE_KEY} (ymirror, toggle, startpulse or '
            f'{CLOCK_DIVISION_TYPE}), {TIMING_SIGNAL_KEY} and {TIMING_SIGNAL_TIME_KEY} '
            f'in its metadata.yaml.')


def timing_from_metadata(metadata: dict, path: str = '') -> FrameTiming | None:
    """The FrameTiming a recording's metadata describes, or None if it
    describes none.

    Incomplete or unknown is an error rather than None: a folder that names a
    type and gets the rest wrong is not a folder that said nothing.
    """
    kind = metadata.get(TIMING_TYPE_KEY)
    if kind is None:
        return None
    kind = str(kind).lower()

    known = sorted(frame_time_methods) + [CLOCK_DIVISION_TYPE]
    if kind not in known:
        raise ValueError(f'{path}: unknown {TIMING_TYPE_KEY} "{kind}" in the metadata. '
                         f'Known: {", ".join(known)}.')

    signal = metadata.get(TIMING_SIGNAL_KEY)
    signal_time = metadata.get(TIMING_SIGNAL_TIME_KEY)
    missing = [key for key, value in ((TIMING_SIGNAL_KEY, signal),
                                      (TIMING_SIGNAL_TIME_KEY, signal_time)) if value is None]
    if missing:
        raise ValueError(f'{path}: {TIMING_TYPE_KEY} is "{kind}" but the metadata gives '
                         f'no {" and ".join(missing)}.')

    if kind == CLOCK_DIVISION_TYPE:
        edges = metadata.get(TIMING_EDGES_KEY)
        if edges is None:
            raise ValueError(f'{path}: {TIMING_TYPE_KEY} {CLOCK_DIVISION_TYPE} needs '
                             f'{TIMING_EDGES_KEY}, the volume period in clock ticks.')
        return ClockDivisionTiming(float(edges), signal=str(signal), signal_time=str(signal_time))

    return SyncSignalTiming(method=kind, signal=str(signal), signal_time=str(signal_time),
                            frame_avg_num=int(metadata.get(FRAME_AVG_KEY, 1)))


class ImagingSpec:
    """One imaging source to ingest, and how to time it.

    `source` may be a name from the registry or an instance, so a caller with
    an unregistered source can pass it directly. `timing` is a FrameTiming to
    apply to every recording, or None - the default - to read each recording's
    timing from its own metadata.yaml (see `timing_from_metadata`).
    """

    def __init__(self, source='suite2p', name: str = None, timing: FrameTiming = None,
                 **options):
        if timing is not None and not isinstance(timing, FrameTiming):
            raise TypeError(f'timing is a FrameTiming instance, or None to read it from '
                            f'each recording folder; got {timing!r}.')

        self.source = imaging_sources[source] if isinstance(source, str) else source
        self.name = self.source.name if name is None else name
        self.options = options
        self._timing = timing

    def timing_for(self, path: str) -> FrameTiming:
        """How the frames in this recording folder were timed.

        Raises MissingFrameTiming when nothing names it, which an ingest
        reports rather than fails on.
        """
        if self._timing is not None:
            return self._timing

        timing = timing_from_metadata(read_metadata(path), path)
        if timing is None:
            raise MissingFrameTiming(path, metadata_files_in(path))

        return timing

    def resolved_timing(self) -> FrameTiming | None:
        """The timing applied to every recording, or None when each recording
        brings its own."""
        return self._timing

    def __repr__(self):
        return f'ImagingSpec({self.name}, source={self.source.name})'


imaging_sources = {
    'suite2p': Suite2pSource(),
}


# The files vxpy writes beside one another into a recording folder. Any one of
#  them is enough to tell a recording from whatever else sits in the tree -
#  ants_registration, a zstack, a notes file.
RECORDING_FILE_NAMES = frozenset({'io.hdf5', 'camera.hdf5', 'display.hdf5',
                                  'gui.hdf5'})


def is_recording_folder(path: str) -> bool:
    """Whether `path` looks like a vxpy recording folder."""
    if not os.path.isdir(path):
        return False

    return any(name.lower() in RECORDING_FILE_NAMES for name in os.listdir(path))


def scan_experiment(path: str) -> list[tuple[str, list[str]]]:
    """The animal folders of an experiment folder, and the recordings in each.

    A folder is an animal if it holds at least one recording, which is what
    leaves out anything else sitting at that level. Reading the tree is separate
    from ingesting it, so that a caller can see what a run would do before
    starting one that takes hours:

        for animal_path, recordings in scan_experiment('/data/cmn'):
            print(os.path.basename(animal_path), len(recordings))

    Returns:
        list: one (animal_path, [recording_path, ...]) per animal, in name
            order, holding only animals that have at least one recording.
    """
    contents = []
    for animal_name in sorted(os.listdir(path)):
        animal_path = os.path.join(path, animal_name)
        if not os.path.isdir(animal_path):
            continue

        recordings = [os.path.join(animal_path, name)
                      for name in sorted(os.listdir(animal_path))
                      if is_recording_folder(os.path.join(animal_path, name))]

        if len(recordings) > 0:
            contents.append((animal_path, recordings))

    return contents


def lacks_frame_timing(recording_path: str) -> bool:
    """Whether a recording folder has an imaging source and no frame timing in
    its metadata - the folder an ingest reports and leaves out."""
    if not any(source.detect(recording_path) for source in imaging_sources.values()):
        return False

    return read_metadata(recording_path).get(TIMING_TYPE_KEY) is None


class VxEntarchy(entarchy.Entarchy):

    # A store is compatible when it has this tree, under these parents; anything
    #  else that grew on it comes along. The tree has changed twice: an Imaging
    #  entity went between Recording and Layer, so that imaging is one source
    #  among possibly several rather than a property of the recording, and an
    #  Experiment went above Animal, so that which animals belong together is a
    #  level of the hierarchy rather than a string written on each of them.
    _entity_types = (Experiment,)

    def add_experiment(self, path: str, imaging='auto', name: str = None,
                       with_video: bool = True, skip_broken: bool = True,
                       limit: int = None) -> Experiment:
        """Ingest one experiment folder: `<experiment>/<animal>/<recording>`.

        This is the unit the ingest was missing. `add_animal` and `add_recording`
        each take one folder, and everything above them - which folders are
        animals, which of those hold recordings, what to do when one of
        thirty-eight fails halfway through - was left to whoever wrote the loop.

            ent.add_experiment('/data/cmn')
            ent.add_experiment('/data/rot_trans', imaging=ImagingSpec(
                timing=ClockDivisionTiming(7.5, signal='di_frame_sync')))

        How frames were timed cannot be detected, so each recording folder says
        so in its metadata.yaml - `frame_sync_signal_type`, `frame_sync_signal`,
        `frame_sync_signal_timing` - and a folder with an imaging source and no
        such entry is reported, listed at the end, and left out whole, so that a
        rerun picks it up once the metadata is there. A timing passed here
        applies to every recording instead. Either way the choice is written
        onto the Experiment under `imaging/` and onto each Imaging entity, so the
        entarchy records how it read its own data rather than leaving that in
        the script that happened to run.

        Re-running continues rather than duplicating, since `add_animal` and
        `add_recording` both skip what is already there. An ingest that stopped
        is resumed by running it again.

        Args:
            path: the experiment folder. Its name becomes the experiment id
                unless `name` says otherwise.
            imaging: as in `add_recording`, applied to every recording here.
            with_video: take the behaviour videos in as media.
            skip_broken: report a folder that fails and carry on rather than
                ending the run. False re-raises, which is what a test wants.
            limit: at most this many recordings per animal. A trial run over
                the whole tree before committing hours to it, since a later
                full run picks up the rest.

        Returns:
            Experiment: the experiment, newly created or already present.
        """
        path = pathlib.Path(path).as_posix()
        if not os.path.isdir(path):
            raise FileNotFoundError(f'No experiment folder {path}')

        if limit is not None and limit < 1:
            raise ValueError(f'limit is a number of recordings per animal and '
                             f'is at least one, got {limit}.')

        contents = scan_experiment(path)
        if limit is not None:
            contents = [(animal_path, recordings[:limit])
                        for animal_path, recordings in contents]

        if len(contents) == 0:
            raise FileNotFoundError(
                f'{path} holds no animal folder with a vxpy recording in it. An '
                f'experiment folder is <experiment>/<animal>/<recording>; one '
                f'animal folder goes to add_animal instead.')

        experiment = self.experiment(os.path.basename(path) if name is None else name)
        self._record_imaging_choice(experiment, imaging, path)

        total = sum(len(recordings) for _, recordings in contents)
        print(f'\nIngest experiment {experiment.id} from {path}')
        print(f'{len(contents)} animals, {total} recordings')

        started = time.time()
        added, skipped, failed, untimed = 0, 0, [], []

        for animal_path, recording_paths in contents:
            animal_id = os.path.basename(animal_path)
            print(f'\n{"#" * 78}\n# {experiment.id} / {animal_id}\n{"#" * 78}')

            try:
                animal = self.add_animal(experiment, animal_path)
            except Exception:
                if not skip_broken:
                    raise
                failed.append((animal_path, traceback.format_exc()))
                print(f'FAILED to add animal {animal_id}:\n{failed[-1][1]}')
                continue

            for recording_path in recording_paths:
                done = added + skipped + len(failed)
                print(f'\n----- {os.path.basename(recording_path)}  '
                      f'[{done}/{total}, {(time.time() - started) / 60:.0f} min] -----')

                try:
                    recording = self.add_recording(animal, recording_path,
                                                   imaging=imaging,
                                                   with_video=with_video)
                except MissingFrameTiming as error:
                    if not skip_broken:
                        raise
                    untimed.append(recording_path)
                    print(f'NO FRAME TIMING, left out: {error}')
                    continue
                except Exception:
                    if not skip_broken:
                        raise
                    failed.append((recording_path, traceback.format_exc()))
                    print(f'FAILED {os.path.basename(recording_path)}:\n{failed[-1][1]}')
                    continue

                if recording is None:
                    skipped += 1
                else:
                    added += 1

        print(f'\n{"=" * 78}')
        print(f'{experiment.id}: added {added}, skipped {skipped} already present, '
              f'{len(untimed)} without frame timing, {len(failed)} failed, '
              f'in {(time.time() - started) / 60:.1f} min')

        if len(untimed) > 0:
            print(f'\n{len(untimed)} recordings have an imaging source but no frame timing '
                  f'in their metadata.yaml, and were left out. Each needs {TIMING_TYPE_KEY} '
                  f'(ymirror, toggle, startpulse or {CLOCK_DIVISION_TYPE}), {TIMING_SIGNAL_KEY} '
                  f'and {TIMING_SIGNAL_TIME_KEY}; a rerun picks them up:')
            for path in untimed:
                print(f'  {path}')

        for failed_path, error in failed:
            print(f'\n--- {failed_path}\n{error}')

        return experiment

    def _imaging_record(self, imaging, path: str) -> dict:
        """How frames are timed, flattened into attribute names and values."""
        if imaging is None:
            return {'imaging': 'none'}

        if isinstance(imaging, str) and imaging == 'auto':
            return {'imaging': 'auto'}

        specs = self._imaging_specs(imaging, path)
        record = {'imaging': ', '.join(spec.name for spec in specs)}

        for spec in specs:
            prefix = f'imaging/{spec.name}'
            record[f'{prefix}/source'] = spec.source.name

            # None when each recording brings its own timing, from its
            #  metadata, and then nothing about it holds for the experiment;
            #  each Imaging entity records what it was timed with
            timing = spec.resolved_timing()
            if timing is None:
                record[f'{prefix}/timing/type'] = 'per recording'
                record[f'{prefix}/timing/source'] = 'metadata.yaml'
                continue

            record[f'{prefix}/timing/type'] = type(timing).__name__
            for key, value in timing.config.items():
                record[f'{prefix}/timing/{key}'] = value

        return record

    def _record_imaging_choice(self, experiment: Experiment, imaging,
                               path: str) -> None:
        """Write how this experiment's frames were timed onto the experiment.

        Without it the entarchy holds data it cannot say how it interpreted. The
        ratio a divided clock is read at is a measurement, and it used to exist
        only in whichever script happened to run the ingest.

        An experiment that already carries a different choice is refused rather
        than overwritten. An experiment grows over days, and the second run is
        the one where somebody reaches for the default: that would time the new
        recordings differently from the ones already in there and replace the
        record saying so, which is the exact failure this attribute exists to
        prevent.
        """
        proposed = self._imaging_record(imaging, path)
        recorded = {name: experiment[name] for name in experiment.keys()
                    if name == 'imaging' or name.startswith('imaging/')}

        if len(recorded) > 0 and recorded != proposed:
            differences = '\n'.join(
                f'  {name:<40} was {recorded.get(name, "-")!r}, now {proposed.get(name, "-")!r}'
                for name in sorted(set(recorded) | set(proposed))
                if recorded.get(name) != proposed.get(name))
            raise ValueError(
                f'Experiment "{experiment.id}" was ingested with a different '
                f'imaging choice, and the {len(experiment.recordings)} recordings '
                f'already in it were timed that way:\n{differences}\n'
                f'Pass the same imaging to add to it, or name a new experiment.')

        with self:
            for name, value in proposed.items():
                experiment[name] = value

    def experiment(self, experiment: Experiment | str) -> Experiment:
        """The Experiment of that name, created if there is none yet.

        Idempotent by id, the way `set_current_analysis` is, so an ingest that
        runs twice adds animals to the experiment it made the first time rather
        than to a second one.
        """
        if isinstance(experiment, Experiment):
            return experiment

        if not isinstance(experiment, str):
            raise TypeError(f'An experiment is named by an Experiment or a '
                            f'string, not by {type(experiment).__name__}.')

        existing = self.get(Experiment) & f'id == "{experiment}"'
        if len(existing) > 0:
            return existing[0]

        with self:
            print(f'> Create new entity for experiment {experiment}')
            entity = self.root.add_child(Experiment, experiment)

        return entity

    @entarchy.digest_method
    def add_animal(self, experiment: Experiment | str, path: str,
                   use_anatomy_reference: str = None) -> Animal:
        """Add one animal folder to an experiment.

        The experiment comes first, as the animal does in `add_recording`: an
        animal belongs to one, and naming it by string creates it if it is new.

            ent.add_animal('cmn', '/data/cmn/2024-08-02_fish1')
        """

        experiment = self.experiment(experiment)
        path = pathlib.Path(path).as_posix()

        print(f'> Add animal from path {path}')

        # Create animal
        path_parts = path.split('/')
        animal_id = path_parts[-1]

        # Scoped to the experiment, so that the same animal id under two of them
        #  is two animals rather than a collision
        animal_collection = self.get(
            Animal, f'id == "{animal_id}" and [Experiment]uuid == "{experiment.uuid}"')

        if len(animal_collection) > 0:
            print(f'WARNING: animal with id {animal_id} already exists in '
                  f'experiment {experiment.id}. Skipping.')
            return animal_collection[0]

        with self:

            # Create new animal entity
            print(f'>> Create new entity for animal {animal_id}')
            animal = experiment.add_child(Animal, animal_id)

            # Search for zstacks
            zstack_names = []
            for fn in os.listdir(path):
                _p = os.path.join(path, fn)
                if os.path.isdir(_p):
                    continue
                if 'zstack' in fn:
                    if fn.lower().endswith(('.tif', '.tiff')):
                        zstack_names.append(fn)

            # Add first stack that was detected
            if len(zstack_names) > 0:
                if len(zstack_names) > 1:
                    print(f'WARNING: multiple zstacks detected, using {zstack_names[0]}')

                # Load zstack
                zstack_data = tifffile.imread(os.path.join(path, zstack_names[0]))

                print(f'>> Add zstack {zstack_names[0]} of shape {zstack_data.shape}')
                animal['zstack_fn'] = zstack_names[0]
                animal['zstack'] = zstack_data

        # Add metadata
        add_metadata(animal, path)

        # Search for valid registration path in animal folder
        valid_reg_path = None
        if 'ants_registration' in os.listdir(path):
            for mov_folder in os.listdir(os.path.join(path, 'ants_registration')):
                for ref_folder in os.listdir(os.path.join(path, 'ants_registration', mov_folder)):

                    # Skip if user specified a particular reference name
                    if use_anatomy_reference is not None and use_anatomy_reference not in ref_folder:
                        continue

                    reg_path = os.path.join(path, 'ants_registration', mov_folder, ref_folder)

                    # If there is a transform file, we'll take it
                    if 'Composite.h5' in os.listdir(long_path(reg_path)):
                        valid_reg_path = reg_path
                        break

        # Write registration metadata to animal entity
        if valid_reg_path is not None:
            print(f'Loading ANTs registration metadata at {valid_reg_path}')
            with open(long_path(os.path.join(valid_reg_path, 'metadata.yaml')), 'r') as f:
                ants_metadata = yaml.safe_load(f)
            animal.update({f'ants/{n}': v for n, v in ants_metadata.items()})

        self.commit()

        return animal

    # Behaviour cameras write one video per device beside the HDF5 files, named
    #  after the device that produced it: fish_embedded -> fish_embedded_frame.avi
    VIDEO_SUFFIXES = ('.avi', '.mp4', '.mkv', '.mov')

    def _camera_devices(self, path: str) -> list[str]:
        """The camera devices Camera.hdf5 says this recording used."""
        camera_path = next((os.path.join(path, name) for name in os.listdir(path)
                            if name.lower() == 'camera.hdf5'), None)
        if camera_path is None:
            return []

        with h5py.File(camera_path, 'r') as h5file:
            devices = h5file.attrs.get('__camera_device_list', [])

        return [str(device) for device in devices]

    def _add_recording_videos(self, recording: Recording, path: str) -> list[str]:
        """Take the behaviour videos of a recording into the entarchy.

        Camera.hdf5 already carries the frame times and whatever tracking ran on
        the video; what it does not carry is the video. Without it the frames
        behind `tail_pose_data` cannot be looked at again, and an entarchy is
        meant to hold everything it needs.

        Videos are copied, never moved - an ingest must not consume the raw data
        it was pointed at. Files are matched to the camera devices the recording
        declares, so a video that is an output rather than an acquisition (a
        stimulus animation dropped in the folder, say) is left alone.

        Returns:
            list of str: the attribute names written.
        """
        entries = {name.lower(): name for name in os.listdir(path)}
        written = []

        for device in self._camera_devices(path):
            for suffix in self.VIDEO_SUFFIXES:
                candidate = entries.get(f'{device}_frame{suffix}'.lower())
                if candidate is None:
                    continue

                name = f'camera/{device}/video'
                source = os.path.join(path, candidate)
                print(f'> Take in {candidate} '
                      f'({os.path.getsize(source) / 1024 ** 2:.1f} MB)')
                recording.set_media(name, source)
                written.append(name)
                break
            else:
                print(f'WARNING: camera device "{device}" has no video file in {path}')

        return written

    # What a Roi must carry whatever produced it. A source's own vocabulary stays
    #  namespaced beside these - s2p/npix, caiman/SNR_comp - so nothing is lost;
    #  what these names buy is that analysis can read a Roi without knowing which
    #  software segmented it.
    ROI_REQUIRED = ('index', 'fluorescence')
    ROI_OPTIONAL = ('spikes', 'is_unit', 'unit_probability')

    def _read_record_groups(self, path: str):
        """The stimulation phase id per analog sample, on the io timebase.

        This is what says when each phase actually ran, and it needs no
        microscope - which is why it is read here rather than inside an imaging
        source, where it used to be resampled onto calcium frames.
        """
        io_path = os.path.join(path, 'Io.hdf5')
        if not os.path.exists(io_path):
            print('WARNING: no Io.hdf5; phases will have no time windows')
            return None

        with h5py.File(io_path, 'r') as io_file:
            if '__record_group_id' in io_file:
                ids = io_file['__record_group_id'][:].squeeze()
                times = io_file['__time'][:].squeeze()
            else:
                ids = io_file['record_group_id'][:].squeeze()
                times = io_file['global_time'][:].squeeze()

        return ids, times

    def _ingest_hdf5_files(self, recording: Recording, path: str, record_groups) -> dict:
        """Everything the acquisition software wrote, imaging or not.

        Phases are created here rather than by an imaging source: a stimulation
        phase is a fact about what was shown, and a recording with no microscope
        still has them.

        Returns:
            dict: the Phase entities, keyed by index.
        """
        phase_data = {}

        for data_fn in os.listdir(path):
            if not any([data_fn.lower().endswith(fn) for fn in ['.h5', 'hdf5']]):
                continue

            # Get short name for attribute names
            fn_short = data_fn.split('.')[0].lower()
            with h5py.File(os.path.join(path, data_fn), 'r') as h5file:

                print(f'> {data_fn}')
                # Get attributes
                recording.update({f'{fn_short}/attrs/{k}': v for k, v in h5file.attrs.items()})
                for key1, member1 in tqdm.tqdm(h5file.items()):

                    # If dataset, save to recording directly
                    if isinstance(member1, h5py.Dataset):
                        recording[f'{fn_short}/{key1}'] = np.squeeze(member1[:])
                        continue

                    # Otherwise it's a group -> keep going

                    # Add phase
                    if key1.startswith('phase'):
                        phase_index = int(key1.replace('phase', ''))

                        # Shared across files: two logs describing the same phase
                        #  contribute attributes to one entity rather than
                        #  colliding on a second one with the same id
                        if phase_index in phase_data:
                            phase = phase_data[phase_index]
                        else:
                            phase = recording.add_child(Phase, key1)
                            phase_data[phase_index] = phase
                            phase['index'] = phase_index

                            window = _phase_time_window(record_groups, phase_index)
                            if window is not None:
                                phase['start_time'], phase['end_time'] = window

                        # Write attributes
                        for attr_key, attr_value in member1.attrs.items():
                            phase[f'{fn_short}/{attr_key}'] = attr_value

                        # Write datasets
                        for key2, member2 in member1.items():
                            if isinstance(member2, h5py.Dataset):
                                phase[key2] = np.squeeze(member2[:])
                            else:
                                print('WARNING: nested groups in phase not supported yet')

                    # Add other data
                    else:
                        # Write attributes
                        for k, v in member1.attrs.items():
                            recording[f'{fn_short}/{key1}/{k}'] = v

                        # Write datasets
                        for key2, member2 in member1.items():
                            if isinstance(member2, h5py.Dataset):
                                recording[f'{fn_short}/{key1}/{key2}'] = np.squeeze(member2[:])

        return phase_data

    def _link_phase_frames(self, imaging: Imaging, record_groups) -> int:
        """Record which frames of this source each phase covers.

        A link rather than an attribute on the Phase, because the window belongs
        to the pair - two sources of one recording sample the same phase
        differently. Phase.frames_in() reads it back.
        """
        if record_groups is None:
            return 0

        frame_times = imaging['frame_times']
        ids, times = record_groups
        on_frames = scipy.interpolate.interp1d(times, ids, kind='nearest')(frame_times)

        linked = 0
        for phase in imaging.recording.phases:
            inside = np.where(on_frames == phase['index'])[0]
            if len(inside) == 0:
                print(f'WARNING: phase {phase["index"]} covers no frames of {imaging.id}')
                continue

            self.link(phase, imaging, PHASE_FRAMES_LINK,
                      start_index=int(inside[0]), end_index=int(inside[-1]))
            linked += 1

        return linked

    def _check_roi_contract(self, imaging: Imaging) -> None:
        """Whether a source wrote what analysis is entitled to expect.

        Checked on the first ROI of each layer rather than all of them: a source
        that does not honour the contract fails for every ROI it wrote, and
        reading them all to find that out would cost more than the ingest did.

        Must run after a commit - entities queued in a write context cannot be
        queried, so this would otherwise inspect nothing and pass.
        """
        layers = list(imaging.layers)
        if len(layers) == 0:
            raise RuntimeError(
                f'Imaging source "{imaging.id}" produced no layers.')

        for layer in layers:
            rois = layer.rois
            if len(rois) == 0:
                continue

            missing = [name for name in self.ROI_REQUIRED if name not in rois[0]]
            if len(missing) > 0:
                raise RuntimeError(
                    f'Imaging source "{imaging.id}" produced ROIs without {missing}. '
                    f'Every source must write {list(self.ROI_REQUIRED)} so that '
                    f'analysis can read an ROI without knowing what segmented it.')

    def _imaging_specs(self, imaging, path: str) -> list:
        """Normalise the `imaging` argument into a list of ImagingSpec."""
        if imaging is None:
            return []

        if imaging == 'auto':
            return [ImagingSpec(source=source)
                    for source in imaging_sources.values() if source.detect(path)]

        if isinstance(imaging, (str, ImagingSpec, ImagingSource)):
            imaging = [imaging]

        specs = []
        for item in imaging:
            if isinstance(item, ImagingSpec):
                specs.append(item)
            elif isinstance(item, str):
                if item not in imaging_sources:
                    raise ValueError(
                        f'Unknown imaging source "{item}". '
                        f'Known: {", ".join(sorted(imaging_sources))}.')
                specs.append(ImagingSpec(source=item))
            elif isinstance(item, ImagingSource):
                specs.append(ImagingSpec(source=item))
            else:
                raise TypeError(f'Cannot read {item!r} as an imaging source.')

        return specs

    @entarchy.digest_method
    def add_imaging(self, recording: Recording, source='suite2p', path: str = None,
                    name: str = None, timing: FrameTiming = None, **options) -> Imaging:
        """Add one imaging source to a recording that already exists.

        The recording folder has to be given again rather than remembered: an
        entarchy is self-contained and must not depend on a path outside itself
        for reading, only for provenance.

            ent.add_imaging(recording, 'suite2p', path='/data/fish1/rec_01')
            ent.add_imaging(recording, caiman_source, path=..., name='caiman_v2')

        Returns:
            Imaging: the new source entity.
        """
        if path is None:
            raise ValueError('add_imaging needs the recording folder; an entarchy '
                             'does not keep a path it can read from.')

        spec = source if isinstance(source, ImagingSpec) else ImagingSpec(
            source=source, name=name, timing=timing, **options)

        with self:
            imaging = self._add_imaging(recording, spec, path,
                                        self._read_record_groups(path))

        return imaging

    def _add_imaging(self, recording: Recording, spec: 'ImagingSpec', path: str,
                     record_groups, timing: FrameTiming = None) -> Imaging:
        """Ingest one source. Assumes an open write context.

        `timing` is what `spec.timing_for(path)` gave, when the caller resolved
        it already - add_recording does, before it creates anything.
        """
        existing = [source.id for source in recording.imaging]
        if spec.name in existing:
            raise ValueError(f'{recording} already has an imaging source named '
                             f'"{spec.name}".')

        if timing is None:
            timing = spec.timing_for(path)

        layer_names = spec.source.layer_names(path)
        if len(layer_names) == 0:
            raise FileNotFoundError(
                f'Imaging source "{spec.source.name}" found no data in {path}. '
                f'Use imaging="auto" to ingest whatever is there, or None to skip it.')

        # The folder may say how many layers to expect; a source that wrote
        #  another number is a folder to look at, not one to ingest
        declared = read_metadata(path).get(LAYER_NUM_KEY)
        if declared is not None and int(declared) != len(layer_names):
            raise ValueError(f'{path}: the metadata says {LAYER_NUM_KEY}: {declared}, but '
                             f'{spec.source.name} wrote {len(layer_names)} layers.')

        print(f'> Add imaging source {spec.name} ({spec.source.name})')
        imaging = recording.add_child(Imaging, spec.name)
        imaging['method'] = spec.source.name
        imaging['layer_num'] = len(layer_names)

        # How this recording was timed, on the entity that was timed
        imaging['timing'] = timing.name
        for key, value in timing.config.items():
            imaging[f'timing/{key}'] = value
        frame_times_by_layer = timing.frame_times(path, len(layer_names))

        spec.source.ingest(imaging, path, frame_times_by_layer, spec.options)

        frame_times = imaging['frame_times']
        imaging['rate'] = 1. / np.diff(frame_times).mean()
        print(f'> Estimated, effective imaging rate {imaging["rate"]:.2f}Hz')

        # Both of the following ask the database what was just created, and
        #  entities queued inside a write context are not visible to a query
        #  until they are written
        self.commit()

        self._check_roi_contract(imaging)
        self._link_phase_frames(imaging, record_groups)

        recording['has_imaging'] = True

        return imaging

    @entarchy.digest_method
    def add_recording(self, animal: Animal, path: str, with_video: bool = True,
                      imaging='auto') -> Recording | None:
        """Ingest one vxpy recording folder.

        Args:
            imaging: which imaging sources to take signals from. 'auto' ingests
                every registered source that has data in the folder and skips
                imaging entirely when none does; None ingests none, leaving a
                recording of stimulus, io and behaviour data; a name, an
                ImagingSpec, or a list of either ingests those. How the frames
                were timed comes from the folder's metadata.yaml unless the
                ImagingSpec names a timing.
            with_video: take the behaviour videos into the entarchy as media.

        Raises:
            MissingFrameTiming: the folder has an imaging source and nothing
                says how its frames were timed. Nothing is created.
        """

        path = pathlib.Path(path).as_posix()

        # Create recording
        path_parts = path.split('/')
        recording_id = path_parts[-1]

        # Check if recording with same id already exists for this animal
        if len(self.get(Recording, f'id == "{recording_id}" and [Animal]uuid == "{animal.uuid}"')) > 0:
            print(f'WARNING: recording with id {recording_id} already exists for animal {animal.id}. Skipping.')
            return None

        if not is_recording_folder(path):
            print(f'WARNING: {path} does not appear to be vxpy recording folder. Skipping.')
            return None

        specs = self._imaging_specs(imaging, path)

        if len(specs) > 0 and not os.path.exists(os.path.join(path, 'Io.hdf5')):
            raise FileNotFoundError(
                f'{path} has imaging output but no Io.hdf5, so its frames cannot '
                f'be timed. Pass imaging=None to ingest the rest of it.')

        # Resolved before anything is created, so that a folder which cannot
        #  say how it was timed leaves nothing behind
        timings = {spec.name: spec.timing_for(path) for spec in specs}

        print(f'Process recording folder {path}')

        with self:

            # When each stimulation phase ran, on the io timebase. Needed whether
            #  or not there is imaging, so it is read before any source is.
            record_groups = self._read_record_groups(path)

            # Create new recording entity
            recording = animal.add_child(Recording, recording_id)
            print(f'> Create {recording}')

            # Add metadata
            add_metadata(recording, path)
            recording['has_imaging'] = False

            self._ingest_hdf5_files(recording, path, record_groups)

            if with_video:
                self._add_recording_videos(recording, path)

            for spec in specs:
                self._add_imaging(recording, spec, path, record_groups, timings[spec.name])

        return recording

    @entarchy.digest_method
    def update_roi_coordinates_from_registration(self, recording_path: str,
                                                 imaging: str = None,
                                                 experiment: str = None):
        """Refresh ROI coordinates from ANTs registration output.

        Registration is computed about an extraction rather than being part of
        one, so it can be re-run and written back afterwards. `imaging` names
        which source's ROIs to update; with one source it can be left out.

        A recording is found by the last two parts of its path, which name the
        animal and the recording. An animal id is scoped to its experiment, so
        the same one may exist under two of them - `experiment` says which, and
        is only needed when it does.
        """
        parts = pathlib.Path(recording_path).as_posix().split('/')
        query = f'[Animal]id == "{parts[-2]}" AND id == "{parts[-1]}"'
        if experiment is not None:
            query = f'{query} AND [Experiment]id == "{experiment}"'

        found = self.get(Recording, query)

        if len(found) == 0:
            raise LookupError(f'No recording {parts[-2]}/{parts[-1]} in this '
                              f'entarchy.')

        if len(found) > 1:
            names = ', '.join(sorted(r.animal.experiment.id for r in found))
            raise LookupError(
                f'{parts[-2]}/{parts[-1]} exists under several experiments '
                f'({names}), so its path does not say which one to update. '
                f'Name it with experiment=.')

        recording = found[0]

        source_entity = (recording.sole_imaging() if imaging is None
                         else recording.imaging[imaging])
        source = imaging_sources.get(source_entity['method'])
        if source is None:
            raise ValueError(
                f'No registered source named "{source_entity["method"]}", so its '
                f'registration output cannot be located.')

        for layer in source_entity.layers:
            roi_coordinates = source._registration_coordinates(recording_path, layer.id)

            if roi_coordinates is None:
                continue

            rois = layer.rois
            for k in ['x', 'y', 'z']:
                rois[f'ants/{k}'] = rois['index'].apply(
                    lambda idx: roi_coordinates.iloc[idx][k])


# Where Windows starts refusing paths. Directories cap lower than files, so one
#  threshold below both is what decides when to ask for the long form.
_PATH_LIMIT = 240


def long_path(path: str) -> str:
    """A path Windows will open past its 260 character limit.

    ANTs registration output nests a reference image name inside a moving image
    name, both long and descriptive, so what sits beside them can land a
    character or two over: `2024-08-02_fish1` has its registration metadata at
    261. The failure is quiet and misleading - listing the folder still works,
    while opening a file inside it raises a bare FileNotFoundError - so this is
    applied wherever the ingest reaches into registration output.

    Returns the path unchanged off Windows, and where it is short enough that
    the plain form is the better-trodden one.
    """
    if os.name != 'nt':
        return path

    absolute = os.path.abspath(path)
    if len(absolute) < _PATH_LIMIT or absolute.startswith('\\\\'):
        return absolute

    return f'\\\\?\\{absolute}'


# What a folder may say about itself. Setups differ in what they call the
#  sidecar they write - a recording depth is under `info.yaml` in one and
#  `metadata.yaml` in another - so both are read and neither is lost.
METADATA_FILE_NAMES = ('metadata.yaml', 'info.yaml')


def metadata_files_in(folder_path: str) -> list[str]:
    """The metadata files a folder holds, by name."""
    return sorted(f for f in os.listdir(folder_path) if f.endswith(METADATA_FILE_NAMES))


def read_metadata(folder_path: str) -> dict:
    """What a folder's metadata files say, merged into one mapping.

    A file that is not valid YAML is reported and skipped; one that holds no
    mapping contributes nothing.
    """
    metadata = {}
    for name in metadata_files_in(folder_path):
        with open(os.path.join(folder_path, name), 'r') as stream:
            try:
                content = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                continue
        if isinstance(content, dict):
            metadata.update(content)

    return metadata


def add_metadata(entity: entarchy.Entity, folder_path: str):
    """Write what the folder's metadata files say onto the entity, under
    `metadata/`."""
    print(f'Found {len(metadata_files_in(folder_path))} metadata files in {folder_path}.')

    unravel_dict(read_metadata(folder_path), entity, 'metadata')


def _phase_time_window(record_groups, index: int):
    """When a stimulation phase started and ended, in seconds.

    Read off the record group trace, so it says when the phase actually ran
    rather than what the protocol asked for - and needs no microscope, which is
    the point: a phase is a fact about the stimulus.
    """
    if record_groups is None:
        return None

    ids, times = record_groups
    inside = np.where(ids == index)[0]
    if len(inside) == 0:
        return None

    return float(times[inside[0]]), float(times[inside[-1]])


def unravel_dict(dict_data: dict, entity: entarchy.Entity, path: str):
    for key, item in dict_data.items():
        if isinstance(item, dict):
            unravel_dict(item, entity, f'{path}/{key}')
            continue
        entity[f'{path}/{key}'] = item


# TODO: replace scipy here, it's overkill and only used twice

# Frame time calculation methods

def ca_frame_times_from_y_mirror(mirror_position: np.ndarray, mirror_time: np.ndarray):
    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = scipy.signal.find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]

    # Discard all before (and including) first trough
    trough_idcs = trough_idcs[first_trough < trough_idcs]
    frame_idcs = np.sort(np.concatenate([trough_idcs, peak_idcs]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    return frame_idcs, frame_times


def ca_frame_times_from_start_pulse(sync_signal: np.ndarray, sync_time: np.ndarray):
    """Frame times from a line that pulses once at each frame start.

    The rising edge of the pulse is the start, so only rises count. A digital
    line is recorded as bool, and np.diff of a bool array is XOR: a fall reads
    as True just as a rise does, so every frame would be counted twice.
    Widening to a signed type first makes a fall negative again.

    np.diff reports the gap between samples i and i+1, so the rising sample -
    the first one recorded while the frame was being acquired - is i+1.
    """
    frame_indices = np.where(np.diff(sync_signal.astype(np.int8)) > 0)[0] + 1
    frame_times = sync_time[frame_indices]

    return frame_indices, frame_times


def ca_frame_times_from_toggle(sync_signal: np.ndarray, sync_time: np.ndarray):
    """Frame times from a line that flips state at each frame start.

    The level carries no meaning, the change does: every edge, rising or
    falling, is a frame. The first sample after the flip is the frame's.
    """
    frame_indices = np.where(np.diff(sync_signal.astype(np.int8)) != 0)[0] + 1
    frame_times = sync_time[frame_indices]

    return frame_indices, frame_times


# The recording's vocabulary for its sync line, as metadata.yaml names it
frame_time_methods = {
    'ymirror': ca_frame_times_from_y_mirror,
    'toggle': ca_frame_times_from_toggle,
    'startpulse': ca_frame_times_from_start_pulse,
}


if __name__ == '__main__':
    pass