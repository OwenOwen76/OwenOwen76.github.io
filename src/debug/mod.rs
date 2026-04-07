use crate::debug::map::*;
use crate::state::GameState;
use bevy::input::keyboard::KeyboardInput;
use bevy::prelude::*;
use std::io::{self, Write};

pub mod map;

pub struct DebugPlugin;

impl Plugin for DebugPlugin {
    fn build(&self, app: &mut App) {
        app.init_state::<DebugOn>()
            .init_resource::<DebugCommand>()
            .add_systems(Update, read_debug_keys.run_if(in_state(GameState::Playing)))
            .add_systems(
                Update,
                listen_for_slash
                    .run_if(in_state(DebugOn::On))
                    .run_if(in_state(GameState::Playing)),
            )
            .add_systems(
                Update,
                read_debug_commands
                    .run_if(in_state(DebugOn::TypingCommand))
                    .run_if(in_state(GameState::Playing)),
            );
    }
}

#[derive(Resource, States, Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DebugOn {
    On,
    #[default]
    Off,
    TypingCommand,
}

pub fn read_debug_keys(
    input: Res<ButtonInput<KeyCode>>,
    current_state: Res<State<DebugOn>>,
    mut next_state: ResMut<NextState<DebugOn>>,
) {
    if input.just_pressed(KeyCode::F3) {
        match current_state.get() {
            DebugOn::Off => {
                next_state.set(DebugOn::On);
                info!("Debug mode: ON");
            }
            DebugOn::On => {
                next_state.set(DebugOn::Off);
                info!("Debug mode: OFF");
            }
            _ => {}
        }
    }
}

#[derive(Resource, Default)]
pub struct DebugCommand {
    pub current_text: String,
}

pub fn listen_for_slash(
    keys: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<DebugOn>>,
    mut debug_cmd: ResMut<DebugCommand>,
) {
    if keys.just_pressed(KeyCode::Slash) {
        next_state.set(DebugOn::TypingCommand);
        debug_cmd.current_text = "/".to_string();

        print!("\r\x1b[K> /");
        let _ = io::stdout().flush();
    }
}

pub fn read_debug_commands(
    mut commands: Commands,
    mut debug_cmd: ResMut<DebugCommand>,
    mut char_evr: MessageReader<KeyboardInput>,
    keys: Res<ButtonInput<KeyCode>>,
    mut player_query: Query<&mut Transform, With<crate::characters::input::Player>>,
    asset_server: Res<AssetServer>,
    mut texture_atlas_layouts: ResMut<Assets<TextureAtlasLayout>>,
    mut next_state: ResMut<NextState<DebugOn>>,
) {
    let mut changed = false;

    // 1. Backspace & Space
    if keys.just_pressed(KeyCode::Backspace) {
        debug_cmd.current_text.pop();
        if debug_cmd.current_text.is_empty() {
            next_state.set(DebugOn::On);
            println!("\r\x1b[K> (closed)");
        }
        changed = true;
    }

    if keys.just_pressed(KeyCode::Space) {
        debug_cmd.current_text.push(' ');
        changed = true;
    }

    // 2. Typing
    for ev in char_evr.read() {
        if !ev.state.is_pressed() {
            continue;
        }

        if let bevy::input::keyboard::Key::Character(c) = &ev.logical_key {
            if c.as_str() == "/" && debug_cmd.current_text == "/" {
                continue;
            }

            debug_cmd.current_text.push_str(c.as_str());
            changed = true;
        }
    }

    if changed {
        print!("\r\x1b[K> {}", debug_cmd.current_text);
        let _ = io::stdout().flush();
    }

    // 3. Enter
    if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        let full_command = std::mem::take(&mut debug_cmd.current_text);
        println!("\nExecuting: {}", full_command);

        next_state.set(DebugOn::On);

        let parts: Vec<&str> = full_command.split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        match parts[0] {
            "/spawn" => {
                // Expected format /spawn {targot type} {target name}
                let option = parts.get(1).unwrap_or(&"_");
                match *option {
                    "tilemap" => {
                        if let Some(target) = parts.get(2) {
                            debug_spawn_sprites(
                                &mut commands,
                                target,
                                &player_query.into_readonly(),
                                &asset_server,
                                &mut texture_atlas_layouts,
                            );
                        }
                    }
                    _ => warn!("Unknown spawn option: {}", option),
                }
            }
            "/tp" => {
                // Expected format /tp @{target} <x> <y>
                let target = parts.get(1).unwrap_or(&"_");

                match *target {
                    "@p" => {
                        let x_arg = parts.get(2);
                        let y_arg = parts.get(3);

                        if let (Some(x_str), Some(y_str)) = (x_arg, y_arg) {
                            if let (Ok(x), Ok(y)) = (x_str.parse::<f32>(), y_str.parse::<f32>()) {
                                if let Ok(mut transform) = player_query.single_mut() {
                                    transform.translation.x = x * 16.0;
                                    transform.translation.y = y * 16.0;
                                    info!("Teleported @p to {}, {}", x, y);
                                }
                            } else {
                                warn!("Invalid coordinates! Use numbers: /tp @p <x> <y>");
                            }
                        } else {
                            warn!("Usage: /tp @p <x> <y>");
                        }
                    }
                    _ => warn!("Unknown target: '{}'. Try @p", target),
                }
            }
            "/help" => info!("Available: /spawn tilemap <name>"),
            _ => warn!("Unknown command: {}", parts[0]),
        }
    }
}
