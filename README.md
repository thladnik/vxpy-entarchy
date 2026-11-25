## Entarchy schema for data analysis using vxpy- and suite2p-based data

Defines the entity hierarchy `Experiment > Animal > Recording > {Layer > Roi, Phase}`
on top of
[entarchy](https://github.com/thladnik/entarchy), ingests vxpy recordings and
suite2p output, and implements the contiguous-motion-noise (CMN) receptive field
analysis in `analysis/cmn`.

### Installation

```sh
pip install vxentarchy
```

This installs [entarchy](https://pypi.org/project/entarchy/) with it and puts a
`vxentarchy` command on the path (`vxentarchy --help`; the same as
`python -m vxentarchy`). `pip install "vxentarchy[cmn]"` adds `torch` and the
rest of what the CMN receptive field analysis needs.

### Ingest

A whole experiment folder goes in with one call:

```python
ent = VxEntarchy.open_or_create(analysis_path)
ent.add_experiment('/data/cmn')
```

`add_experiment` expects `<experiment>/<animal>/<recording>` and walks it: a
folder is a recording if it holds one of the files vxpy writes, and an animal
if it holds at least one recording, which leaves out `ants_registration` and
anything else in the tree. `scan_experiment(path)` returns the same reading
without ingesting anything, for a look before a run that takes hours.

**An experiment is not ingested once.** It is recorded over days, and the folder
is handed back with more in it than last time. Re-running scans the folder again
and takes in what is new: a new animal, or a new recording of an animal already
there. What is already in the entarchy is skipped rather than duplicated, so an
ingest that stopped is also resumed by running it again. A folder that fails is
reported and the run carries on; `skip_broken=False` re-raises instead.

An existing animal is skipped whole, so a zstack or an ANTs registration that
arrived after its first ingest is not picked up by a later run.
`update_roi_coordinates_from_registration` exists for the registration case.

The levels are separately available, each taking its parent first:

```python
animal = ent.add_animal('cmn', '/data/cmn/2024-08-02_fish1')
ent.add_recording(animal, '/data/cmn/2024-08-02_fish1/rec_01')
```

### From the command line

```bash
python -m vxentarchy create /data/entarchy
python -m vxentarchy add /data/entarchy /raw/cmn
python -m vxentarchy scan /raw/cmn
```

`add` takes `--name`, `--limit`, `--no-video`, `--dry-run`, `--stop-on-error`, and
`--create` to make the entarchy on the way in. Without `--create` a path that
holds no entarchy is an error rather than a new empty one, so a typo costs
nothing.

`--imaging` is `auto` for every source found in a recording folder, `none` for
stimulus and behaviour only, or a source name such as `suite2p`. How the frames
were timed is not an option: each recording folder says so in its
`metadata.yaml`, see below. `scan`, `--dry-run` and `add` all name the
recordings that have imaging output but no frame timing before anything is
opened, and `add` exits non-zero and names any recording in the folder that did
not make it in.

`--backend mysql` with `--dbname/--dbhost/--dbuser` creates on a server instead.
The password is never an argument: it comes from `ENTARCHY_DB_PASSWORD` or a
prompt, and is not written to `entarchy.yaml`.

**Frame timing is written in each recording folder.** It cannot be detected
from the data, and guessing which line a rig recorded is how frames end up timed
wrongly without anybody noticing. So a recording folder that holds an imaging
source (a `suite2p` subfolder, say) needs a `metadata.yaml` that names it:

```yaml
frame_sync_signal_type: ymirror                 # ymirror, toggle or startpulse
frame_sync_signal: ai_y_mirror_in               # channel of Io.hdf5
frame_sync_signal_timing: ai_y_mirror_in_time   # its time points
frame_avg_num: 1                                # optional: frames averaged into one
layer_num: 5                                    # optional: checked against the source
```

| `frame_sync_signal_type` | the signal | a frame starts at |
|---|---|---|
| `ymirror` | the galvo y mirror position | each turning point of the mirror |
| `toggle` | a line that flips state once per frame | every edge, rising or falling |
| `startpulse` | a line that pulses once per frame | each rising edge |

`frame_avg_num` is taken as 1 when left out. `layer_num` is never needed - the
number of layers is what the source wrote, e.g. the `plane*` folders of suite2p -
but when it is given and disagrees, the recording fails rather than being
ingested with the wrong one. A fourth type, `clockdivision`, reads a free-running
clock that ticks a non-integer number of times per volume, and takes that
measured ratio as `frame_sync_edges_per_volume`.

A recording that has an imaging source and no `frame_sync_signal_type` is
reported when the ingest reaches it, left out whole - no recording entity
without its imaging - and listed with all the others at the end of the run. A
rerun picks it up once the metadata is there. A folder without an imaging
source needs none of this, and neither does `imaging=None`.

What a recording was timed with is written onto its Imaging entity, and the
Experiment says where that came from:

```python
imaging['timing/method']                     # 'ymirror'
imaging['timing/signal']                     # 'ai_y_mirror_in'
experiment['imaging/suite2p/timing/type']    # 'per recording'
```

A `FrameTiming` passed in applies to every recording of the run instead of what
the folders say, for an experiment whose folders cannot be given metadata:

```python
ent.add_experiment('/data/rot_trans', imaging=ImagingSpec(
    timing=ClockDivisionTiming(7.5, signal='di_frame_sync')))

experiment['imaging/suite2p/timing/type']             # 'ClockDivisionTiming'
experiment['imaging/suite2p/timing/edges_per_volume']  # 7.5
```

An experiment reaches everything below it, and an animal names the experiment
it belongs to:

```python
experiment.animals, experiment.recordings, experiment.rois, experiment.phases
animal.experiment
ent.get(Roi, '[Experiment]id == "cmn"')
```

`add_recording` expects a vxpy recording folder containing `Io.hdf5` (with the
galvo mirror sync signal) and the stimulus log HDF5 files. Behaviour videos
beside them are taken into the entarchy as media attributes; pass
`with_video=False` to skip that.

**Imaging is optional, and its source is a choice.** A recording of stimulus,
io and behaviour data alone ingests completely — phases included, since a
stimulation phase is a fact about what was shown rather than about the
microscope.

```python
ent.add_recording(animal, path)                     # every source found in the folder
ent.add_recording(animal, path, imaging='suite2p')  # this one, and require it
ent.add_recording(animal, path, imaging=None)       # none

# several, and later
ent.add_recording(animal, path, imaging=['suite2p', ImagingSpec(CaImAnSource(),
                                                               name='caiman')])
ent.add_imaging(recording, 'suite2p', path='/data/fish1/rec_01')
```

Each source is an `Imaging` entity under the Recording, named after itself, and
its layers and ROIs hang off it:

```
Experiment > Animal > Recording > Imaging > Layer > Roi
                                > Phase
```

so `plane0` from suite2p and `plane0` from CaImAn are different entities rather
than a collision. Everything about when frames happened belongs to the source —
`imaging['rate']`, `imaging['frame_times']` — because two sources of one
recording need not agree.

```python
recording.imaging               # every source
recording.imaging['suite2p']    # one by name
recording.sole_imaging()        # the only one, or an error saying there are several
recording.rois                  # every ROI, whichever source found it
```

Phases always carry `start_time` and `end_time` in seconds, read off the record
group trace so they say when the phase actually ran. Which *frames* a phase
covers depends on the source, so it is a link rather than an attribute:

```python
phase.frames_in(imaging)    # (start_index, end_index)
phase.frames_in()           # when there is only one source
```

### What an ROI carries

Whatever segmented it, an ROI has:

| attribute | |
|---|---|
| `index` | position within its layer — required |
| `fluorescence` | the trace, one sample per frame — required |
| `spikes` | deconvolved activity, if the method produces it |
| `is_unit` | whether the method calls it a real cell |
| `unit_probability` | how sure it was |

A source's own vocabulary stays namespaced beside these — `s2p/npix`,
`s2p/skew` — so nothing is lost. What the shared names buy is analysis that
reads an ROI without knowing what produced it, which is why the CMN analysis
mentions suite2p nowhere. The required names are checked at ingest.

Adding a source means implementing `ImagingSource` — `detect`, `layer_names`,
`ingest` — and registering it in `imaging_sources`. Frame timing is a separate
interface (`FrameTiming`), because how a scanner was timed and what read its
output are independent choices. The reasoning is in
[the proposal](https://github.com/thladnik/vxpy-entarchy/blob/master/docs/proposals/imaging-sources.md).

### Example notebook

[`examples/01_synthetic_dataset.ipynb`](https://github.com/thladnik/vxpy-entarchy/blob/master/examples/01_synthetic_dataset.ipynb)
builds a small entarchy in this schema with synthetic calcium imaging data, then
works through querying it, DataFrames, parallel analysis with `map_async`, links
between entities, and archiving. It needs no experimental data and runs in about
half a minute.

```sh
pip install -e ".[examples]"
jupyter lab examples/01_synthetic_dataset.ipynb
```

It also runs from a checkout without installing, since the first cell puts the
repository root on the path.

### Running tests

```sh
pip install -e . pytest
pytest
```

The CMN analysis tests require `torch` (part of the `cmn` extra) and are skipped
if it is missing. The ingest tests build a synthetic vxpy + suite2p dataset on
disk, so no experimental data is needed.

A few tests are marked `xfail` to document known defects without failing the
suite; they are listed with a reason and will report as unexpectedly passing once
the underlying issue is fixed.
