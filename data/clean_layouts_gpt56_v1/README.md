# GPT-5.6 Furniture Pilot Clean Layouts v1

This directory contains 30 independently designed, single-room Furniture-stage clean layouts for SceneRepair_v3. Existing Qwen layouts were not used as target answers.

## Contract

- Scene schema: `scene_repair_v3_clean_furniture_scene_v1`
- Coordinate frame: `room_local_z_up`, room center `(0,0)`, x in `[-length/2,length/2]`, y in `[-width/2,width/2]`
- Quaternion order: `wxyz`; yaw is geometric rotation around `+Z`, radians, wrapped to `[-pi,pi)`
- Openings: door and window only; clearance AABBs follow SceneExpert depths (door 0.8 m, window 0.5 m)
- Furniture: at most 17 movable items; no manipulands, wall decoration, or ceiling objects
- HSSD IDs: lookup/provenance/split identity only, never a model feature
- Semantic package hash: `3ab8d60e33e73bceac607544fae3724267e711f86aa467d3c670fed6d3002e11`
- Rules version: `furniture_layout_rules_gpt56_v1`
- Verifier version: `clean_furniture_verifier_gpt56_v1`

## Bbox provenance

Static HSSD bbox values come from `interaction_clearance.nonartic_clearance_v2.object_bbox_m` in Z-up `[x,y,z]`. Official/retrieved articulated records use asset `[x,y_up,z]`; output width/depth/height is deterministically converted to room `[x,z,y]`. Every object records its source path and conversion under `audit`.

## Verification

`validation_report.json` separates deterministic geometry/identity/relation/path checks from GPT semantic visual review. The deterministic verifier checks finite transforms, HSSD identity and bbox equality, room bounds, Furniture OBB penetration, opening clearance, instance-level functional partner orientation/gap, declared use clearances, path reachability, and exact/near duplicate layouts.

## Known limitations

- The pilot uses rectangular rooms and 2D floor-plan OBB checks; it does not run mesh-level collision or physics simulation.
- HSSD canonical front is reliable only where its semantic-front flag is true. Final training pose remains geometric yaw.
- Grid reachability approximates a 0.56 m diameter person and does not model articulated door swing.
- GPT semantic review judges functional plausibility from top-down plots; it is not a deterministic proof and is listed separately in the report.
