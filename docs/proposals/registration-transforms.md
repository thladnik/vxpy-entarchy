# Proposal: registration transforms, mapped at analysis time

Status: **proposed** 2026-11-25. Nothing here is implemented. Seven decisions
are yours; they are collected at the end, each with what I would pick.

## What is there now

Three pieces of code reach into `ants_registration` folders, and each reads
something different:

| where | reads | writes |
|---|---|---|
| `add_animal` | `metadata.yaml` beside the first `Composite.h5` it finds | `animal['ants/*']` |
| `Suite2pSource._registration_coordinates`, at ingest | `mapped_points.h5` | `roi['ants/x\|y\|z']` |
| `update_roi_coordinates_from_registration`, afterwards | `mapped_points.h5` again | the same |

No transform is stored. What is stored is the *result* of a chain of transforms
that the registration UI applied once, and that result does not say which chain.

I looked at what is on `Z:/cluster/data_raw` to see what a design has to hold.

**The chain has three links, and the third is outside the experiment.** For
`cmn_original/2024-07-18_fish1/2024-07-18_rec2`, `mapped_points.h5` is

    plane0  ->  animal z-stack  ->  2025-01-24 tectum reference  ->  mapzebrain

with the links in `<rec>/suite2p/ants_registration/plane0/<zstack>/`,
`<animal>/ants_registration/<zstack>/<reference>/` and
`registration_references/reference_data/2025-01-24_jf7_standard_stack/ants_registration/<reference>/16bit_T_AVG_HuCH2BGCaMP2-tg_ch0.tif/`.

**Several valid transforms per entity is the normal case, not the odd one.**
All six `cmn_original` animals have two: z-stack to the tectum reference
(February 2025) and z-stack to `live_standard_nlsGCaMP` (February 2026). The
stored ROI coordinates still come from the first. Nothing in the entarchy says
so, and the newer registration is not used by anything.

**Most registration folders are not registrations.** Animals have between one
and five `<moving>/<fixed>` folders, of which usually one holds a
`Composite.h5`. The rest are attempts that were abandoned.

**Two layouts at recording level.** `<rec>/suite2p/ants_registration/planeN/<zstack>/`
(the cmn experiments; five planes each in `cmn_texasred`) and
`<rec>/ants_registration/suite2p/<zstack>/` (`rf_location_mapping`). In both,
what is registered is one plane, not the recording.

**`fixed_path` in the registration metadata cannot be followed.** It is relative
to a working directory that differs between versions of the UI, and one of the
two on this animal points at a folder that has since moved. So the next link
cannot be discovered from the previous one; it has to be found by name.

**The transforms can be applied without ANTs, exactly.** I read the three
`InverseComposite.h5` files with h5py, applied them with numpy and
`scipy.ndimage.map_coordinates`, and compared with the stored `mapped_points.h5`
for all 919 ROIs of that recording:

    after plane0 -> z-stack                 median |diff| 187.2 um
    after z-stack -> tectum reference       median |diff| 139.0 um
    after tectum reference -> mapzebrain    median 0.0000 um, max 0.0001 um

So analysis-time mapping needs no new dependency, and the existing
`mapped_points.h5` files are a ready-made test oracle for the implementation.

**The deformable transforms are large.** An affine transform is twelve numbers.
A SyN transform carries a displacement field on the fixed grid: 684 MB for the
tectum reference, 2.4 GB for mapzebrain, 3.7 GB for `live_standard`, and twice
that because both directions are needed. Stored as they are, `cmn_original`
alone is about 50 GB of transforms. The fields are smooth, so I measured what a
coarser copy costs, at 20000 random points per field:

| field | stored as | size | error, 99th percentile | error, max |
|---|---|---|---|---|
| z-stack -> tectum reference | as is, float32 | 684 MB | 0 | 0 |
| | every 2nd sample, float16 | 43 MB | 0.05 um | 0.15 um |
| | every 4th sample, float16 | 5 MB | 0.21 um | 0.61 um |
| | every 8th sample, float16 | 0.7 MB | 0.76 um | 2.83 um |
| tectum reference -> mapzebrain | as is, float32 | 2389 MB | 0 | 0 |
| | every 2nd sample, float16 | 150 MB | 0.07 um | 0.41 um |
| | every 4th sample, float16 | 19 MB | 0.23 um | 0.78 um |
| | every 8th sample, float16 | 2.4 MB | 0.77 um | 2.09 um |

float16 against float32 changes nothing measurable. For scale: the voxels are
0.5 to 1 um and a soma is about 5 um.

## The model

**A transform is an edge between two named coordinate spaces.** That is what the
folder layout already says: `ants_registration/<moving>/<fixed>/`. A space is
named by the image it is the space of.

**A transform belongs to the entity whose folder it was found in**, and may be
used by that entity and everything below it:

| entity | folder scanned | moving space | typical fixed space |
|---|---|---|---|
| Layer | `<rec>/<source>/ants_registration/<plane>/` or `<rec>/ants_registration/<source>/` | the plane | the animal's z-stack |
| Animal | `<animal>/ants_registration/<zstack>/` | the z-stack | a lab reference |
| Experiment | `<experiment>/ants_registration/`, and folders the experiment names | a lab reference | an atlas |

Recording-level transforms attach to the **Layer**, because that is what was
registered: `cmn_texasred` has five per recording, one per plane. A Recording
has no transform of its own; `recording.transforms` is all of its layers'.

**Mapping a point is walking the edges upwards.** From a Layer, the transforms
in reach are its own, its animal's and its experiment's. A route to the space
that was asked for is found by matching names: the layer's transform ends in
`reverse_f1_zstack_...`, the animal has one that starts there, and so on.
Nothing is composed or baked at ingest.

### What if several valid transforms are found for one entity?

That question has three different cases, and only one of them is a conflict.

1. **Different fixed spaces** (z-stack to tectum reference, z-stack to
   `live_standard`). Not a conflict. Both are kept. They are edges to different
   places, and which one is used follows from the space the analysis asks for.
   Choosing between them at ingest is what the current code does, and it is why
   the newer registration is invisible.
2. **The same moving and fixed space twice** (both recording layouts present
   for one plane; a z-stack registered under two file names to one reference).
   This is a conflict, and it is refused: the scan names both folders and adds
   neither. No "first one found". `add_animal` today takes the last valid one in
   directory order, because its `break` only leaves the inner loop.
3. **Two routes to the space asked for** (via the tectum reference, and via
   `live_standard`). The shortest route is taken. Two routes of equal length are
   an error that lists them, and `via=` picks one.

A fourth case is not about several transforms but belongs here: **a
registration that was redone**. Same spaces, newer files. `add_transforms`
refuses to overwrite and says what differs; `replace=True` removes the old one
and adds the new.

## Naming

### In the store

An entity type has exactly one parent type in entarchy, so one `Transform` type
cannot sit under Experiment, Animal and Layer. I propose attributes under a
`transform/` prefix on the owning entity, read through a small view object.
This has a useful side effect: the entity tree does not change, so existing
stores stay compatible and can be given transforms without being rebuilt.

    transform/<fixed>/moving              name of the moving space
    transform/<fixed>/fixed               name of the fixed space
    transform/<fixed>/moving_resolution   um per voxel, x y z
    transform/<fixed>/fixed_resolution
    transform/<fixed>/fixed_shape         voxels, x y z
    transform/<fixed>/type                Affine, SyN, SyNRA, ...
    transform/<fixed>/registered          when Composite.h5 was written
    transform/<fixed>/source_path         where it was read from, for provenance only
    transform/<fixed>/field_step          1 = as registered, 4 = every 4th sample
    transform/<fixed>/init/...            init_translation, init_x_rotation, ...
    transform/<fixed>/settings/...        registration_settings, as the UI wrote them

    transform/<fixed>/to_fixed/<i>/...    step i for points moving -> fixed
    transform/<fixed>/to_moving/<i>/...   step i for points fixed -> moving

    .../<i>/kind                          affine | field
    .../<i>/matrix, translation, center   affine
    .../<i>/field, origin, spacing, direction    field: (z, y, x, 3) displacements in um

`to_fixed` is `InverseComposite.h5` and `to_moving` is `Composite.h5`, decoded
and in the order they are applied. The names say what they do to a point, which
the ANTs names do not: points travel against the direction images are warped.
Mapping ROIs to an atlas uses `to_fixed`; resampling a volume onto the atlas
grid looks up each atlas voxel in the moving image and uses `to_moving`.

**A space name is the file name without its extension**:
`16bit_T_AVG_HuCH2BGCaMP2-tg_ch0`. For a plane it is `<source>/<plane>`, e.g.
`suite2p/plane0`. Short names are aliases, kept on the experiment and editable
at any time without touching a transform:

    experiment['space/mapzebrain'] = '16bit_T_AVG_HuCH2BGCaMP2-tg_ch0'
    experiment['space/tectum']     = '2025-01-24_jf7_6dpf_..._0p88_zstep'
    experiment['space/default']    = 'mapzebrain'

Two names are reserved and need no alias: `plane` is the layer's own space and
`zstack` is the space of the animal's z-stack.

### In the raw folders

Nothing changes for animals and recordings. There are two additions.

An experiment folder may have a `metadata.yaml`, which the ingest does not read
today. It names the short spaces and says where the transforms between
references are, since those live outside the experiment:

```yaml
spaces:
  mapzebrain: 16bit_T_AVG_HuCH2BGCaMP2-tg_ch0
  tectum: 2025-01-24_jf7_6dpf_1p0magn_27laser_611gain_6avg_reference_tectum_200mu_0p88_zstep
default_space: mapzebrain
registration_folders:
  - ../../registration_references/reference_data/2025-01-24_jf7_standard_stack
  - ../../registration_references/reference_data/2025-03-13_jf7_standard_stack
```

A registration folder is **valid** when it holds `Composite.h5`,
`InverseComposite.h5` and a `metadata.yaml` with `moving_resolution` and
`fixed_resolution`, and every step in both files is an affine or a displacement
field. Anything else is listed by the scan as incomplete and skipped.
`mapped_points.h5` is no longer read.

## Syntax

### Adding transforms

During a digest it is automatic. `add_experiment`, `add_animal` and
`add_recording` scan the folder they were given and add every valid transform
to the entity they create. A recording without registration is not an error and
is not reported as one; the scan report says how far each chain reaches.

Afterwards, for entities that already exist, there is one method, and the
digest uses the same one:

```python
ent.add_transforms(animal, '/raw/cmn/2024-07-18_fish1')
ent.add_transforms(recording, '/raw/cmn/2024-07-18_fish1/2024-07-18_rec2')   # every layer of every source
ent.add_transforms(experiment, '/raw/registration_references/reference_data/2025-01-24_jf7_standard_stack')

ent.add_transforms(animal, path, replace=True)     # the registration was redone
ent.add_transforms(animal, path, dry_run=True)     # report only
```

It returns what it added, what it skipped as already present, what it refused
and why. It adds what is new and leaves the rest alone, so it can be run after
every registration session.

The workflow you describe, digest now and register later, is one call over the
experiment folder, which walks every level and matches folders to entities the
way `add_experiment` does:

```python
ent.update_transforms('/raw/cmn')
```

```bash
python -m vxentarchy transforms /data/entarchy /raw/cmn [--replace] [--dry-run]
python -m vxentarchy scan /raw/cmn
```

`scan` gains a registration section, per animal and recording: valid
transforms, incomplete folders, conflicts, and chains that stop short, e.g.
"z-stack is registered to `tectum`, and nothing maps `tectum` onward".

Removing one is `entity.remove_transform('tectum')`, which deletes its
attributes with the removal API that now exists.

### Reading transforms

```python
animal.transforms                    # the views on this entity, by fixed space
animal.transforms['tectum']          # one; alias or full name
animal.transforms['tectum']['init/x_rotation']
animal.transforms['tectum'].registered

layer.spaces()                       # every space reachable from here, and by which route
layer.route('mapzebrain')            # the transforms that would be applied, in order
layer.route('mapzebrain', via='live_standard_nlsGCaMP')
```

### Mapping at analysis time

```python
roi.position()                       # um in its own plane: x, y, z
roi.position('zstack')               # in the animal's z-stack
roi.position('mapzebrain')           # in the atlas

rois.positions('mapzebrain')         # DataFrame x, y, z, one row per ROI, in the collection's order
rois.positions()                     # the experiment's default space
rois.positions('mapzebrain', units='voxel')    # um divided by the space's resolution

layer.map_points(points, to='mapzebrain')              # any N x 3 array of um in this entity's space
animal.map_points(points, to='zstack', source='mapzebrain')   # downwards works too

animal.map_volume(animal['zstack'], to='mapzebrain')   # resampled onto the atlas grid
layer.map_volume(mean_image, to='zstack')              # a plane, placed in the z-stack
animal.map_volume('/raw/.../some_stack.tif', to='tectum')    # a TIF is read first
```

`positions` works on any ROI collection. It groups the ROIs by layer, resolves
one route per layer, loads each transform once, and applies it to all points
together. `experiment.rois.positions('mapzebrain')` with animals registered to
different lab references just works, because each animal takes its own route to
the same named space. ROIs with no route get NaN and are counted in a warning,
or raise with `missing='raise'`.

A route that does not exist is an error that says how far it got: "no route
from `suite2p/plane0` to `mapzebrain`: reached `tectum` via Layer, Animal;
nothing maps `tectum` onward".

`map_volume` returns an array on the fixed space's grid, with linear
interpolation by default and `order=0` for label volumes. Volumes are passed
in, not ingested: the TIFs in recording folders are raw movies, and I would not
copy those into the store as a side effect of this.

### What goes away

`roi['ants/x|y|z']`, `animal['ants/*']`, `update_roi_coordinates_from_registration`
and `Suite2pSource._registration_coordinates` are removed, without a shim. Two
places in the CMN analysis have to change:

- `analysis.py` looks ROIs up in region masks by `ants/x|y|z`. It becomes
  `rois.positions('mapzebrain', units='voxel')`.
- `functions.py` corrects fish pitch with `rec.animal['ants/init_x_rotation']`.
  For the six `cmn_original` animals that value differs between the two
  registrations (-15 and -12 degrees on the animal I looked at), and which one
  is in the store was decided by directory order. It becomes
  `animal.transforms[space]['init/x_rotation']`, with the space named by the
  analysis.

## Implementation order

1. `vxentarchy/registration.py`: read an ITK composite file into steps, apply
   steps to points. Tested on small synthetic transform files, plus a script
   that checks real folders against their `mapped_points.h5`.
2. Scan, validity, conflicts, `add_transforms`, `update_transforms`; digest
   integration; CLI; removal of the old paths.
3. Spaces, aliases, routes, `position`, `positions`, `map_points`.
4. `map_volume`.
5. The two changes to the CMN analysis.

## Decisions

1. **How fields are stored.** As registered (exact, about 50 GB for
   `cmn_original`), or coarser. I would store every 4th sample as float16 by
   default, record `field_step` on the transform, and offer `field_step=1` per
   call. The cost stayed within 0.8 um in my sample, below one voxel. If that is
   unwelcome, `field_step=2` stayed within 0.4 um and is still a sixteenth of the size.
2. **Recording-level transforms on the Layer**, not the Recording. This follows
   from registration being per plane, but it differs from how you phrased it.
3. **Attributes or an entity type.** Attributes need nothing from entarchy and
   keep existing stores compatible. A real `Transform` entity would be
   queryable and removable in one call, but needs entarchy to allow one type
   under several parents first. I would start with attributes.
4. **Where the experiment's transforms come from.** `registration_folders` in
   the experiment's `metadata.yaml`, as above, or an explicit
   `add_transforms(experiment, folder)` only. I would do both; the yaml makes a
   digest reproducible.
5. **Where a plane sits in its moving volume.** The UI embeds the mean image at
   slice 6 of an 11-slice volume and builds ROI points as
   `(x, y, 6) * moving_resolution`. That 6 is in the UI's code and nowhere in
   its output. I would have the UI write `moving_plane_index` into the
   registration `metadata.yaml` from now on, and read existing folders with 6,
   as a named constant tied to that UI. The alternative is to add the key by
   hand, as with the frame timing.
6. **Where an ROI is in its plane.** Only suite2p knows today (`s2p/med`). I
   would add `roi_pixel_positions(rois)` to `ImagingSource` rather than a new
   stored attribute, so existing stores need no re-ingest.
7. **Positions you can query.** Mapping at analysis time means
   `ent.get(Roi, 'ants/z > 100')` is gone, since nothing is stored to query.
   `rois.positions('mapzebrain', store=True)` could write
   `position/mapzebrain/x|y|z` as ordinary analysis attributes, with the route
   recorded beside them so that a redone registration is noticed. I would leave
   this out until it is missed.
