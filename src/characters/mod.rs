pub mod animation;
pub mod config;
pub mod facing;
pub mod input;
pub mod physics;
pub mod rendering;
pub mod spawn;
pub mod state;

use bevy::prelude::*;
use bevy_common_assets::ron::RonAssetPlugin;
use config::CharactersList;

use crate::state::GameState;

pub struct CharactersPlugin;

impl Plugin for CharactersPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(RonAssetPlugin::<CharactersList>::new(&["characters.ron"]))
            .init_resource::<spawn::CurrentCharacterIndex>()
            .add_systems(Startup, spawn::spawn_player)
            .add_systems(
                Update,
                (
                    input::handle_player_input,
                    rendering::update_player_depth,
                    input::update_jump_state,
                    animation::on_state_change_update_animation,
                    physics::apply_velocity,
                    animation::animations_playback,
                )
                    .chain()
                    .run_if(in_state(GameState::Playing))
                    .run_if(not(in_state(DebugOn::TypingCommand))),
            );
    }
}
