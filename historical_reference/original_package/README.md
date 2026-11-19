# Original flappy-bird-gymnasium 0.4.0 `flappy_bird_env.py`

Recovered on 2026-11-18 from the local pip HTTP cache (the wheel the user installed in June 2026).
Its sha256 (urlsafe-b64 `63fkjKPQo9-Cy5vCSjT6ag0TZW7eOHhowWs0Ciaw1-k`) matches the install-time
`flappy_bird_gymnasium-0.4.0.dist-info/RECORD`, so this is the exact unmodified file.

`vendor_vs_original.diff` shows the only functional difference from `project/vendor/`:
the LiDAR proximity ("private zone") penalty changed from -0.5 to -1 (the user's Experiment C).
The other two files in envs/ (constants.py, lidar.py) still match RECORD.
