use crate::camera::config::CAMERA_MOVE_SPEED;
use crate::characters::input::Player;
use bevy::prelude::*;

pub fn camera_follow_system(
    player_query: Query<&Transform, (With<Player>, Without<Camera>)>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
    time: Res<Time>,
) {
    let Ok(player_transform) = player_query.single() else {
        return;
    };
    let Ok(mut camera_transform) = camera_query.single_mut() else {
        return;
    };

    let target = player_transform.translation;
    let speed = CAMERA_MOVE_SPEED;

    camera_transform.translation.x = camera_transform
        .translation
        .x
        .lerp(target.x, time.delta_secs() * speed);
    camera_transform.translation.y = camera_transform
        .translation
        .y
        .lerp(target.y, time.delta_secs() * speed);
}
