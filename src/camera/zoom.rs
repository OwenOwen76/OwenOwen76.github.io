use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;

pub fn camera_zoom_system(
    mut query: Query<&mut Projection, With<Camera>>,
    mut scroll_evr: MessageReader<MouseWheel>,
) {
    let Ok(mut projection) = query.single_mut() else {
        return;
    };

    if let Projection::Orthographic(ref mut ortho) = *projection {
        for event in scroll_evr.read() {
            let zoom_step = 0.1;
            ortho.scale -= event.y * zoom_step;
            ortho.scale = ortho.scale.clamp(0.2, 0.8);
        }
    }
}
