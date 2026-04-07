use bevy::prelude::*;

use crate::characters::input::Player;
use crate::debug::*;
use crate::map::{assets::*, generate::*};

use crate::map::assets::{DECOR_TILEMAP, MAIN_TILEMAP};
use crate::map::generate::TILE_SIZE;

pub fn debug_spawn_sprites(
    commands: &mut Commands,
    target_name: &str,
    player_query: &Query<&Transform, With<crate::characters::input::Player>>,
    asset_server: &Res<AssetServer>,
    texture_atlas_layouts: &mut ResMut<Assets<TextureAtlasLayout>>,
) {
    let Ok(player_transform) = player_query.single() else {
        return;
    };

    let x = (player_transform.translation.x / TILE_SIZE).floor() as i32;
    let y = (player_transform.translation.y / TILE_SIZE).floor() as i32;

    // 1. Decor TILEMAP
    if DECOR_TILEMAP.sprite_index(target_name).is_some() {
        let tex = asset_server.load("decorations.png");
        let layout = TextureAtlasLayout::from_grid(UVec2::new(16, 16), 17, 15, None, None);
        let layout_handle = texture_atlas_layouts.add(layout);

        spawn_sprite_decor(commands, target_name, x, y, 5.0, &layout_handle, tex);
        info!("Spawned Decor: {}", target_name);
    }
    // 2. Main Tilemap
    else if MAIN_TILEMAP.sprite_index(target_name).is_some() {
        let tex = asset_server.load("tilemap.png");
        let layout = TextureAtlasLayout::from_grid(UVec2::new(16, 16), 8, 15, None, None);
        let layout_handle = texture_atlas_layouts.add(layout);

        spawn_sprite(commands, target_name, x, y, 5.0, &layout_handle, tex, 8);
        info!("Spawned Main Tile: {}", target_name);
    } else {
        error!("Sprite '{}' not found in assets.rs", target_name);
    }
}
