use crate::characters::input::Player;
use crate::map::generate::*;
use crate::map::noise::*;
use bevy::{platform::collections::HashSet, prelude::*};

#[derive(Resource, Default)]
pub struct ChunkManager {
    pub spawned_chunks: HashSet<(i32, i32)>,
}

#[derive(Resource)]
pub struct MapSeed(pub u32);

pub const CHUNK_SIZE: i32 = 32;
pub const RENDER_DISTANCE: i32 = 3;

pub fn update_chunks(
    mut commands: Commands,
    player_query: Query<&Transform, With<Player>>,
    mut chunk_manager: ResMut<ChunkManager>,
    seed: Res<MapSeed>,
    asset_server: Res<AssetServer>,
    mut texture_atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
) {
    let Ok(player_transform) = player_query.single() else {
        return;
    };
    let perlin = Perlin::new(seed.0);

    let chunk_x = (player_transform.translation.x / (CHUNK_SIZE as f32 * TILE_SIZE)).floor() as i32;
    let chunk_y = (player_transform.translation.y / (CHUNK_SIZE as f32 * TILE_SIZE)).floor() as i32;

    // 1. Load Main Tilemap
    let texture: Handle<Image> = asset_server.load("tilemap.png");
    let layout = TextureAtlasLayout::from_grid(UVec2::new(16, 16), 8, 15, None, None);
    let layout_handle = texture_atlas_layouts.add(layout);
    let cols = 8;

    // 2. Load Decorations
    let decor_texture: Handle<Image> = asset_server.load("decorations.png");
    let mut layout = TextureAtlasLayout::new_empty(UVec2::new(256, 256));

    for sprite in crate::map::assets::DECOR_TILEMAP.sprites {
        layout.add_texture(URect::from_corners(
            UVec2::new(sprite.pixel_x, sprite.pixel_y),
            UVec2::new(sprite.pixel_x + 16, sprite.pixel_y + 16),
        ));
    }

    let decor_layout = texture_atlas_layouts.add(layout);

    for cx in (chunk_x - RENDER_DISTANCE)..=(chunk_x + RENDER_DISTANCE) {
        for cy in (chunk_y - RENDER_DISTANCE)..=(chunk_y + RENDER_DISTANCE) {
            if !chunk_manager.spawned_chunks.contains(&(cx, cy)) {
                spawn_chunk(
                    &mut commands,
                    cx,
                    cy,
                    &perlin,
                    &layout_handle,
                    texture.clone(),
                    cols,
                    &decor_layout,
                    decor_texture.clone(),
                );
                chunk_manager.spawned_chunks.insert((cx, cy));
            }
        }
    }
}
