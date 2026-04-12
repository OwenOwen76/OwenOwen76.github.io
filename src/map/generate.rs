use crate::map::chunks::*;
use crate::map::noise::*;
use bevy::prelude::*;

pub const MAP_SIZE: i32 = 30;
pub const TILE_SIZE: f32 = 16.0;
pub const NOISE_SCALE: f64 = 0.05;
pub const DECOR_NOISE_SCALE: f64 = 0.08;
pub const GRID_X: u32 = 25;
pub const GRID_Y: u32 = 18;

fn get_tile_type(x: i32, y: i32, perlin: &Perlin) -> (bool, bool) {
    let raw_val = perlin.get([x as f64 * NOISE_SCALE, y as f64 * NOISE_SCALE]);

    let val = (raw_val + 1.0) / 2.0;
    let is_grass = val < 0.65;
    let is_water = val < 0.35;

    (is_grass, is_water)
}

pub fn spawn_chunk(
    commands: &mut Commands,
    cx: i32,
    cy: i32,
    perlin: &Perlin,
    layout: &Handle<TextureAtlasLayout>,
    tex: Handle<Image>,
    cols: u32,
    decor_layout: &Handle<TextureAtlasLayout>,
    decor_tex: Handle<Image>,
) {
    let start_x = cx * CHUNK_SIZE;
    let start_y = cy * CHUNK_SIZE;

    for y in start_y..(start_y + CHUNK_SIZE) {
        for x in start_x..(start_x + CHUNK_SIZE) {
            // --- 1. BASE TERRAIN ---
            let dirt_name = if rand::random::<f32>() > 0.95 {
                "dirt_2"
            } else {
                "dirt_1"
            };
            spawn_sprite(commands, dirt_name, x, y, 0.0, layout, tex.clone(), cols);

            let (is_grass, is_water) = get_tile_type(x, y, perlin);

            // WATER
            if is_water {
                let mask = calculate_mask_water(x, y, perlin);
                let has_grass_neighbor = check_grass_neighbor(x, y, perlin);
                let prefix = if has_grass_neighbor {
                    "water_grass"
                } else {
                    "water_dirt"
                };
                if let Some(s) = match_mask_to_suffix(mask) {
                    let name = if s == "mc" {
                        "water_mc".into()
                    } else {
                        format!("{}_{}", prefix, s)
                    };
                    spawn_sprite(commands, &name, x, y, 2.0, layout, tex.clone(), cols);
                }
            }

            // GRASS
            if is_grass {
                let mask = calculate_mask_grass(x, y, perlin);
                let has_water_neighbor = check_water_neighbor(x, y, perlin);
                let prefix = if has_water_neighbor {
                    "grass_mc"
                } else {
                    "grass"
                };
                if let Some(s) = match_mask_to_suffix(mask) {
                    let name = if s == "mc" {
                        "grass_mc".into()
                    } else {
                        format!("{}_{}", prefix, s)
                    };
                    spawn_sprite(commands, &name, x, y, 1.0, layout, tex.clone(), cols);
                }

                // --- 2. DECORATION ---
                if !is_water {
                    let get_val = |tx: i32, ty: i32| {
                        let raw = perlin
                            .get([tx as f64 * DECOR_NOISE_SCALE, ty as f64 * DECOR_NOISE_SCALE]);
                        (raw + 1.0) / 2.0
                    };
                    let val = get_val(x, y);

                    let local_x = x - start_x;
                    let local_y = y - start_y;

                    let near_edge = local_x < 3
                        || local_x > CHUNK_SIZE - 3
                        || local_y < 1
                        || local_y > CHUNK_SIZE - 6;

                    if !near_edge {
                        // 1. DENSE FOREST
                        if val > 0.55 {
                            if x % 6 == 0 && y % 6 == 0 {
                                let tree_type = if val > 0.80 { "oak_tree" } else { "pine_tree" };
                                spawn_bundle(
                                    commands,
                                    tree_type,
                                    x,
                                    y,
                                    decor_layout,
                                    decor_tex.clone(),
                                );
                            }
                        }
                        // 2. LARGE OBSTACLES
                        else if val > 0.45 && val < 0.5 {
                            if x % 5 == 1 && y % 5 == 1 {
                                let obstacle = if val > 0.58 { "boulder" } else { "hollow_log" };
                                spawn_bundle(
                                    commands,
                                    obstacle,
                                    x,
                                    y,
                                    decor_layout,
                                    decor_tex.clone(),
                                );
                            }
                        }
                        // 3. Small Decors
                        else if val > 0.4 && val < 0.3 {
                            if x % 3 == 0 && y % 3 == 0 {
                                let clutter = match (x + y) % 6 {
                                    0 => "mushrooms",
                                    1 => "large_bush",
                                    2 => "weeds",
                                    3 => "small_bush",
                                    4 => "rock",
                                    _ => "boulder",
                                };
                                spawn_bundle(
                                    commands,
                                    clutter,
                                    x,
                                    y,
                                    decor_layout,
                                    decor_tex.clone(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// Helper Function
fn calculate_mask_water(x: i32, y: i32, perlin: &Perlin) -> i32 {
    let mut mask = 0;
    if get_tile_type(x, y + 1, perlin).1 {
        mask += 1;
    }
    if get_tile_type(x + 1, y, perlin).1 {
        mask += 2;
    }
    if get_tile_type(x, y - 1, perlin).1 {
        mask += 4;
    }
    if get_tile_type(x - 1, y, perlin).1 {
        mask += 8;
    }
    mask
}

fn calculate_mask_grass(x: i32, y: i32, perlin: &Perlin) -> i32 {
    let mut mask = 0;
    if get_tile_type(x, y + 1, perlin).0 {
        mask += 1;
    }
    if get_tile_type(x + 1, y, perlin).0 {
        mask += 2;
    }
    if get_tile_type(x, y - 1, perlin).0 {
        mask += 4;
    }
    if get_tile_type(x - 1, y, perlin).0 {
        mask += 8;
    }
    mask
}

fn check_grass_neighbor(x: i32, y: i32, perlin: &Perlin) -> bool {
    get_tile_type(x, y + 1, perlin).0
        || get_tile_type(x + 1, y, perlin).0
        || get_tile_type(x, y - 1, perlin).0
        || get_tile_type(x - 1, y, perlin).0
}

fn check_water_neighbor(x: i32, y: i32, perlin: &Perlin) -> bool {
    get_tile_type(x, y + 1, perlin).1
        || get_tile_type(x + 1, y, perlin).1
        || get_tile_type(x, y - 1, perlin).1
        || get_tile_type(x - 1, y, perlin).1
}

fn match_mask_to_suffix(mask: i32) -> Option<&'static str> {
    match mask {
        6 => Some("tl"),
        14 => Some("tc"),
        12 => Some("tr"),
        7 => Some("ml"),
        15 => Some("mc"),
        13 => Some("mr"),
        3 => Some("bl"),
        11 => Some("bc"),
        9 => Some("br"),
        _ => None,
    }
}

pub fn spawn_sprite(
    commands: &mut Commands,
    name: &str,
    x: i32,
    y: i32,
    z: f32,
    layout: &Handle<TextureAtlasLayout>,
    tex: Handle<Image>,
    cols: u32,
) {
    use crate::map::assets::MAIN_TILEMAP;

    if let Some(sprite_def) = MAIN_TILEMAP.sprites.iter().find(|s| s.name == name) {
        let index = (sprite_def.pixel_y / 16 * cols) + (sprite_def.pixel_x / 16);

        commands.spawn((
            Sprite {
                image: tex,
                texture_atlas: Some(TextureAtlas {
                    layout: (*layout).clone(),
                    index: index as usize,
                }),
                ..default()
            },
            Transform::from_xyz(x as f32 * TILE_SIZE, y as f32 * TILE_SIZE, z),
        ));
    }
}

pub fn spawn_bundle(
    commands: &mut Commands,
    bundle_type: &str,
    base_x: i32,
    base_y: i32,
    layout: &Handle<TextureAtlasLayout>,
    tex: Handle<Image>,
) {
    match bundle_type {
        "oak_tree" | "oak_tree_bare" => {
            let prefix = if bundle_type == "oak_tree" {
                "oak_g"
            } else {
                "oak_d"
            };
            for row in 0..6 {
                for col in 0..5 {
                    let name = format!("{}_{}_{}", prefix, row, col);
                    spawn_sprite_decor(
                        commands,
                        &name,
                        base_x + col - 2,
                        base_y + (5 - row),
                        3.0,
                        layout,
                        tex.clone(),
                    );
                }
            }
        }
        "pine_tree" | "pine_tree_bare" => {
            let prefix = if bundle_type == "pine_tree" {
                "pine_g"
            } else {
                "pine_b"
            };
            for row in 0..5 {
                for col in 0..3 {
                    let name = format!("{}_{}_{}", prefix, row, col);
                    spawn_sprite_decor(
                        commands,
                        &name,
                        base_x + col - 1,
                        base_y + (4 - row),
                        3.0,
                        layout,
                        tex.clone(),
                    );
                }
            }
        }
        "hollow_log" => {
            for row in 0..2 {
                for col in 0..3 {
                    let name = format!("log_g_{}_{}", row, col);
                    spawn_sprite_decor(
                        commands,
                        &name,
                        base_x + col,
                        base_y - (1 - row),
                        2.5,
                        layout,
                        tex.clone(),
                    );
                }
            }
        }
        "mushrooms" | "large_bush" | "weeds" | "reeds" | "rock" | "boulder" | "small_bush" => {
            let (prefix, z) = match bundle_type {
                "mushrooms" => ("mush_g1", 2.1),
                "large_bush" => ("bush_lg_g", 2.2),
                "weeds" => ("weed_g", 2.1),
                "reeds" => ("reed_g", 2.1),
                "rock" => ("rock_g1", 2.4),
                "boulder" => ("boulder_g1", 2.6),
                "small_bush" => ("bush_sm_g", 2.1),
                _ => ("bush_lg_g", 2.0),
            };

            for row in 0..2 {
                for col in 0..2 {
                    let name = format!("{}_{}_{}", prefix, row, col);
                    spawn_sprite_decor(
                        commands,
                        &name,
                        base_x + col,
                        base_y - (1 - row),
                        z,
                        layout,
                        tex.clone(),
                    );
                }
            }
        }
        _ => {}
    }
}

pub fn spawn_sprite_decor(
    commands: &mut Commands,
    name: &str,
    x: i32,
    y: i32,
    z_base: f32,
    layout: &Handle<TextureAtlasLayout>,
    tex: Handle<Image>,
) {
    use crate::map::assets::DECOR_TILEMAP;

    if let Some(index) = DECOR_TILEMAP.sprite_index(name) {
        let dynamic_z = z_base - (y as f32 * 0.001);

        commands.spawn((
            Sprite {
                image: tex,
                texture_atlas: Some(TextureAtlas {
                    layout: (*layout).clone(),
                    index,
                }),
                ..default()
            },
            Transform::from_xyz(x as f32 * TILE_SIZE, y as f32 * TILE_SIZE, dynamic_z),
        ));
    }
}
