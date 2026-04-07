pub mod game_state;
pub mod loading;
pub mod pause;

use crate::characters::config::CharactersList;
use crate::characters::spawn::CharactersListResource;
use crate::state::pause::toggle_pause;
use bevy::prelude::*;

pub use game_state::GameState;

pub struct StatePlugin;

impl Plugin for StatePlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<GameState>()
            .add_systems(OnEnter(GameState::Loading), loading::spawn_loading_screen)
            .add_systems(
                Update,
                (check_assets_loaded, loading::animate_loading)
                    .run_if(in_state(GameState::Loading)),
            )
            .add_systems(
                OnExit(GameState::Loading),
                (
                    loading::despawn_loading_screen,
                    crate::characters::spawn::initialize_player_character,
                ),
            )
            .add_systems(OnEnter(GameState::Paused), pause::spawn_pause_menu)
            .add_systems(OnExit(GameState::Paused), pause::despawn_pause_menu)
            .add_systems(
                Update,
                toggle_pause.run_if(in_state(GameState::Playing).or(in_state(GameState::Paused))),
            );
    }
}

fn check_assets_loaded(
    characters_list_res: Option<Res<CharactersListResource>>,
    characters_lists: Res<Assets<CharactersList>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    let Some(res) = characters_list_res else {
        return;
    };

    if characters_lists.get(&res.handle).is_some() {
        info!("Assets loaded, transitioning to Playing!");
        next_state.set(GameState::Playing);
    }
}
