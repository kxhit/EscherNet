### Filter Objaverse Data 
Filter Zero-1-to-3 rendered views (empty images) and save object ids:
```commandline
python objaverse_filter.py --path /data/objaverse/views_release
```
There are 7983/798759 invalid object ids stored in invalid_ids.npy and 8607/798759 empty folders stored in empty_ids.npy. The all_invalid.npy stores 8607 invalid ids.
We finally use 790152 objects from Objaverse Dataset.
Zero-1-to-3's valid_paths.json contains 772870 ids.

### Render GSO Data
We borrowed Zero-1-to-3's blender rendering scripts, set the GSO path and run:
```commandline
python render_all_mvs.py
```