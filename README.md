# curvox
Utility functions for pointclouds, meshes, and voxel grids.

mesh_conversions.py has utilities for converting meshes between ply objects, and ros messages.

cloud_conversions.py has utilities for converting pointclouds between numpy arrays, pointcloud objects, and ros messages

pc_vox_utils.py is for generating binvox files from pointclouds.


## Install
```bash
$ sudo apt install python-pytest
$ pip2 install -e . --user --process-dependency-links
```

## Tests
A few tests have been written in the `tests` folder. In order to run the test suite
```bash
$ cd tests
$ py.test
```