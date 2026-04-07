use crate::camera::movement::camera_follow_system;
use crate::camera::setup::setup_camera;
use crate::camera::zoom::camera_zoom_system;
use crate::state::GameState;
use bevy::prelude::*;

pub mod config;
pub mod movement;
pub mod setup;
pub mod zoom;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_camera).add_systems(
            Update,
            (camera_follow_system, camera_zoom_system).run_if(in_state(GameState::Playing)),
        );
    }
}
