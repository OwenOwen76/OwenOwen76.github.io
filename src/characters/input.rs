use super::{
    animation::{AnimationController, AnimationTimer},
    config::CharacterEntry,
    facing::Facing,
    physics::Velocity,
    state::CharacterState,
};
use bevy::prelude::*;

#[derive(Component)]
pub struct Player;

pub fn read_movement_input(input: &ButtonInput<KeyCode>) -> Vec2 {
    const MOVEMENT_KEYS: [(&[KeyCode], Vec2); 4] = [
        (&[KeyCode::ArrowLeft, KeyCode::KeyA], Vec2::NEG_X),
        (&[KeyCode::ArrowRight, KeyCode::KeyD], Vec2::X),
        (&[KeyCode::ArrowUp, KeyCode::KeyW], Vec2::Y),
        (&[KeyCode::ArrowDown, KeyCode::KeyS], Vec2::NEG_Y),
    ];

    MOVEMENT_KEYS
        .iter()
        .filter(|(keys, _)| input.any_pressed(keys.iter().cloned()))
        .map(|(_, dir)| *dir)
        .sum()
}

fn determine_new_state(
    current: CharacterState,
    direction: Vec2,
    is_running: bool,
    wants_jump: bool,
) -> CharacterState {
    match current {
        CharacterState::Jumping => CharacterState::Jumping,

        _ if wants_jump && current.is_grounded() => CharacterState::Jumping,

        _ if direction != Vec2::ZERO => {
            if is_running {
                CharacterState::Running
            } else {
                CharacterState::Walking
            }
        }

        _ => CharacterState::Idle,
    }
}

pub fn handle_player_input(
    input: Res<ButtonInput<KeyCode>>,
    mut query: Query<
        (
            &mut CharacterState,
            &mut Velocity,
            &mut Facing,
            &CharacterEntry,
        ),
        With<Player>,
    >,
) {
    let Ok((mut state, mut velocity, mut facing, character)) = query.single_mut() else {
        return;
    };

    // Step 1: Read what keys are pressed
    let direction = read_movement_input(&input);
    let is_running = input.pressed(KeyCode::ShiftLeft) || input.pressed(KeyCode::ShiftRight);
    let wants_jump = input.just_pressed(KeyCode::Space);

    // Step 2: Update facing direction (which way the character looks)
    if direction != Vec2::ZERO {
        let new_facing = Facing::from_velocity(direction);
        if *facing != new_facing {
            *facing = new_facing;
        }
    }

    // Step 3: Use our state machine to determine the new state
    let new_state = determine_new_state(*state, direction, is_running, wants_jump);
    if *state != new_state {
        *state = new_state;
    }

    // Step 4: Calculate velocity based on state
    *velocity = super::physics::calculate_velocity(*state, direction, character);
}

pub fn update_jump_state(
    mut query: Query<
        (
            &mut CharacterState,
            &Facing,
            &AnimationController,
            &AnimationTimer,
            &Sprite,
            &CharacterEntry,
        ),
        With<Player>,
    >,
) {
    let Ok((mut state, facing, controller, timer, sprite, config)) = query.single_mut() else {
        return;
    };

    if *state != CharacterState::Jumping {
        return;
    }

    let Some(atlas) = sprite.texture_atlas.as_ref() else {
        return;
    };

    let Some(clip) = controller.get_clip(config, *facing) else {
        return;
    };

    if clip.is_complete(atlas.index, timer.just_finished()) {
        *state = CharacterState::Idle;
    }
}
