# Recording rig

One phone filming side-on: a human hand moves a white bowl onto a white plate, on black card.
Recorded 2026-10-12. The calibration (capture/src/phone_capture/calib.py) fits the camera to this geometry.

## Layout

World frame (cm): origin = plate centre on the floor card; x along the object line toward the plate's right
edge; y away from the camera; z up.

- Floor: A2 black card, 59.4 x 42 cm, long side along the line (ends at x = -44.4 and +15). Backdrop: a second
  A2 card upright behind it; the object line is 18 cm in front of it.
- Plate: 28 cm across, 3 cm tall, fixed at x = 0.
- Bowl: 16.5 cm across, 5.5 cm tall. The LIBERO bowl is 11.2 cm across; the retargeting maps bowl progress
  along the bowl -> plate line and absolute lift, not hand poses, so the size difference does not matter
  except for the hand's own motion near the bowl (scaled by 11.15 / 16.5).
- Start ticks on the line: bowl-to-plate gaps of 16, 12 and 2 cm (recordings G1_gap16, G2_gap12, G3_gap2).
- Lift guide: blue marks on the backdrop at 21.5 / 20 cm (original pair) and 16 cm (added pair).
- Grasp: right hand from the right, black sleeve; pinch the bowl's right rim (plate side), thumb inside, index
  outside; lift, carry, place at the plate's centre, release, withdraw.

## Camera

Samsung Galaxy S23 Ultra, 1x main camera, 1920x1080 H.264, variable frame rate (29.7 fps mean): the pipeline
uses per-frame timestamps. Focal length 1465.6 px, from an earlier test session calibrated with a vertical
tape measure (same lens and settings).

Fitted from the session's own empty frame: lens 17.1 cm left of the plate centre, 56.2 cm in front of it,
34.4 cm up, tilted 23.8 deg down (horizontal field of view 66.5 deg). Checks: the lift-guide marks
back-project to 21.0 / 15.1 / 20.7 / 16.8 cm (21.5 / 16 / 20 / 16), and the bowl rim fits to a 7.58 cm radius
(7.8 cm in the tape-calibrated test session).

The lens was hand-measured at 50-52 cm in front. In the test session, holding it there still fitted the image to
about 1 px but made the resting bowl float 2 cm above the card, so the free fit is used: one view lets focal length and
distance trade off, and the scene's own consistency decides.

Depth (y) is not measurable from one side-on camera. The bowl is modelled as travelling straight from its
measured start to the plate centre; a 3 cm depth error would shift heights by about 1.5 cm.
