use bevy::camera::ScalingMode;
use bevy::prelude::*;

pub fn setup_camera(mut commands: Commands) {
    let zoom = 3.0;

    commands.spawn((
        Camera2d,
        Msaa::Off,
        Projection::Orthographic(OrthographicProjection {
            scale: 1.0 / zoom,
            //scaling_mode: ScalingMode::FixedVertical(720.0),
            ..OrthographicProjection::default_2d()
        }),
    ));
}
