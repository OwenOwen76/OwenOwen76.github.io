use bevy::prelude::*;

use crate::map::chunks::{ChunkManager, MapSeed, update_chunks};
pub mod assets;
pub mod chunks;
pub mod generate;

pub struct MapPlugin;

impl Plugin for MapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkManager>()
            .insert_resource(MapSeed(rand::random()))
            .add_systems(Update, update_chunks);
    }
}
