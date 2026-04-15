use crate::characters::animation::*;
use crate::characters::config::{CharacterEntry, CharactersList};
use crate::characters::facing::Facing;
use crate::characters::input::Player;
use crate::characters::physics::Velocity;
use crate::characters::state::CharacterState;
use bevy::prelude::*;

pub const PLAYER_SCALE: f32 = 0.5;
pub const PLAYER_Z_POSITION: f32 = 3.0;

#[derive(Resource, Default)]
pub struct CurrentCharacterIndex {
    pub index: usize,
}

#[derive(Resource)]
pub struct CharactersListResource {
    pub handle: Handle<CharactersList>,
}

fn create_character_atlas_layout(
    atlas_layouts: &mut ResMut<Assets<TextureAtlasLayout>>,
    character_entry: &CharacterEntry,
) -> Handle<TextureAtlasLayout> {
    let max_row = character_entry.calculate_max_animation_row();

    atlas_layouts.add(TextureAtlasLayout::from_grid(
        UVec2::splat(character_entry.tile_size),
        character_entry.atlas_columns as u32,
        (max_row + 1) as u32,
        None,
        None,
    ))
}

pub fn spawn_player(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut character_index: ResMut<CurrentCharacterIndex>,
) {
    let characters_list_handle: Handle<CharactersList> = asset_server.load("character.ron");

    commands.insert_resource(CharactersListResource {
        handle: characters_list_handle,
    });

    character_index.index = 0;

    commands.spawn((
        Player,
        Transform::from_translation(Vec3::new(0.0, 0.0, PLAYER_Z_POSITION))
            .with_scale(Vec3::splat(PLAYER_SCALE)),
        Sprite::default(),
    ));
}

pub fn initialize_player_character(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    characters_lists: Res<Assets<CharactersList>>,
    character_index: Res<CurrentCharacterIndex>,
    characters_list_res: Option<Res<CharactersListResource>>,
    mut query: Query<Entity, (With<Player>, Without<AnimationController>)>,
) {
    let Some(characters_list_res) = characters_list_res else {
        return;
    };

    for entity in query.iter_mut() {
        let Some(characters_list) = characters_lists.get(&characters_list_res.handle) else {
            continue;
        };

        if character_index.index >= characters_list.characters.len() {
            continue;
        };

        let character_entry = &characters_list.characters[character_index.index];

        let texture = asset_server.load(&character_entry.texture_path);
        let layout = create_character_atlas_layout(&mut atlas_layouts, character_entry);

        let sprite = Sprite::from_atlas_image(texture, TextureAtlas { layout, index: 0 });

        commands.entity(entity).insert((
            AnimationController::default(),
            CharacterState::default(),
            Velocity::default(),
            Facing::default(),
            AnimationTimer(Timer::from_seconds(
                DEFAULT_ANIMATION_FRAME_TIME,
                TimerMode::Repeating,
            )),
            character_entry.clone(),
            sprite,
        ));
    }
}
