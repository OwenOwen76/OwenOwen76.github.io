use bevy::prelude::*;

use crate::characters::input::Player;
use crate::characters::spawn::PLAYER_SCALE;
use crate::map::generate::{GRID_Y, TILE_SIZE};

const NODE_SIZE_Z: f32 = 1.0;
const PLAYER_BASE_Z: f32 = 4.0;
const PLAYER_Z_OFFSET: f32 = 0.5;

pub fn update_player_depth(
    mut player_query: Query<&mut Transform, (With<Player>, Changed<Transform>)>,
) {
    let map_height = TILE_SIZE * GRID_Y as f32;
    let map_y0 = -TILE_SIZE * GRID_Y as f32 / 2.0;

    let player_sprite_height = 64.0 * PLAYER_SCALE;

    for mut transform in player_query.iter_mut() {
        let player_center_y = transform.translation.y;

        let player_feet_y = player_center_y - (player_sprite_height / 2.0);

        let t = ((player_feet_y - map_y0) / map_height).clamp(0.0, 1.0);

        let player_z = PLAYER_BASE_Z + NODE_SIZE_Z * (1.0 - t) + PLAYER_Z_OFFSET;

        transform.translation.z = player_z;
    }
}
