use bevy::prelude::*;

pub struct TilemapSprite {
    pub name: &'static str,
    pub pixel_x: u32,
    pub pixel_y: u32,
}

pub struct TilemapDefinition {
    pub tile_width: u32,
    pub tile_height: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub sprites: &'static [TilemapSprite],
}

impl TilemapDefinition {
    pub const fn tile_size(&self) -> UVec2 {
        UVec2::new(self.tile_width, self.tile_height)
    }

    pub const fn atlas_size(&self) -> UVec2 {
        UVec2::new(self.atlas_width, self.atlas_height)
    }

    pub fn sprite_index(&self, name: &str) -> Option<usize> {
        self.sprites.iter().position(|sprite| sprite.name == name)
    }

    pub fn sprite_rect(&self, index: usize) -> URect {
        let sprite = &self.sprites[index];
        let min = UVec2::new(sprite.pixel_x, sprite.pixel_y);
        URect::from_corners(min, min + self.tile_size())
    }
}

pub const MAIN_TILEMAP: TilemapDefinition = TilemapDefinition {
    tile_width: 16,
    tile_height: 16,
    atlas_width: 128,
    atlas_height: 240,
    sprites: &[
        // --- DIRT FAMILY (22 Tiles) ---
        // Variety (Top right area)
        TilemapSprite {
            name: "dirt_1",
            pixel_x: 96,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "dirt_2",
            pixel_x: 112,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "dirt_3",
            pixel_x: 96,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "dirt_4",
            pixel_x: 112,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "dirt_5",
            pixel_x: 96,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "dirt_6",
            pixel_x: 112,
            pixel_y: 32,
        },
        // Dirt path/island edges (Middle area)
        TilemapSprite {
            name: "dirt_side_l",
            pixel_x: 48,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "dirt_side_r",
            pixel_x: 80,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "dirt_side_t",
            pixel_x: 64,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "dirt_side_b",
            pixel_x: 64,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "dirt_corner_tl",
            pixel_x: 48,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "dirt_corner_tr",
            pixel_x: 80,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "dirt_corner_bl",
            pixel_x: 48,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "dirt_corner_br",
            pixel_x: 80,
            pixel_y: 80,
        },
        // Dirt Inner Corners (Smallest path turns)
        TilemapSprite {
            name: "dirt_inner_tl",
            pixel_x: 48,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "dirt_inner_tr",
            pixel_x: 64,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "dirt_inner_bl",
            pixel_x: 48,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "dirt_inner_br",
            pixel_x: 64,
            pixel_y: 32,
        },
        // Dirt Cliff/Bottom Edges
        TilemapSprite {
            name: "dirt_edge_l",
            pixel_x: 48,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "dirt_edge_mid",
            pixel_x: 64,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "dirt_edge_r",
            pixel_x: 80,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "dirt_cliff_face",
            pixel_x: 64,
            pixel_y: 64,
        },
        // --- GRASS FAMILY ---
        // Main Grass Island (Top left)
        TilemapSprite {
            name: "grass_tl",
            pixel_x: 0,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "grass_tc",
            pixel_x: 16,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "grass_tr",
            pixel_x: 32,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "grass_ml",
            pixel_x: 0,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "grass_mc",
            pixel_x: 16,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "grass_mr",
            pixel_x: 32,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "grass_bl",
            pixel_x: 0,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "grass_bc",
            pixel_x: 16,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "grass_br",
            pixel_x: 32,
            pixel_y: 32,
        },
        // Grass Inner Corners (Sharp turns)
        TilemapSprite {
            name: "grass_inner_tl",
            pixel_x: 48,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "grass_inner_tr",
            pixel_x: 64,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "grass_inner_bl",
            pixel_x: 48,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "grass_inner_br",
            pixel_x: 64,
            pixel_y: 32,
        },
        // Grass Bottom Cliffs/Edges (The ones you were missing)
        TilemapSprite {
            name: "grass_edge_l",
            pixel_x: 0,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "grass_edge_mid",
            pixel_x: 16,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "grass_edge_r",
            pixel_x: 32,
            pixel_y: 96,
        },
        // Cliff Grass corners (Transition to dirt cliffs)
        TilemapSprite {
            name: "cliff_grass_tl",
            pixel_x: 80,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "cliff_grass_tr",
            pixel_x: 96,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "cliff_grass_bl",
            pixel_x: 80,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "cliff_grass_br",
            pixel_x: 96,
            pixel_y: 80,
        },
        // Grass Variety (Tufts and rocks)
        TilemapSprite {
            name: "grass_1",
            pixel_x: 80,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "grass_2",
            pixel_x: 96,
            pixel_y: 0,
        },
        TilemapSprite {
            name: "grass_3",
            pixel_x: 112,
            pixel_y: 0,
        },
        // --- WATER FAMILY ---
        // --- WATER-GRASS ---
        TilemapSprite {
            name: "water_grass_tl",
            pixel_x: 0,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_grass_tc",
            pixel_x: 16,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_grass_tr",
            pixel_x: 32,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_grass_ml",
            pixel_x: 0,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_mc",
            pixel_x: 16,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_grass_mr",
            pixel_x: 32,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_grass_bl",
            pixel_x: 0,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "water_grass_bc",
            pixel_x: 16,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "water_grass_br",
            pixel_x: 32,
            pixel_y: 192,
        },
        // --- WATER-DIRT ---
        TilemapSprite {
            name: "water_dirt_tl",
            pixel_x: 48,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_dirt_tc",
            pixel_x: 64,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_dirt_tr",
            pixel_x: 80,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_dirt_ml",
            pixel_x: 48,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_dirt_mc",
            pixel_x: 16,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_dirt_mr",
            pixel_x: 80,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_dirt_bl",
            pixel_x: 48,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "water_dirt_bc",
            pixel_x: 64,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "water_dirt_br",
            pixel_x: 80,
            pixel_y: 192,
        },
        // --- INNER-WATER ---
        TilemapSprite {
            name: "water_inner_tl",
            pixel_x: 96,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_inner_tr",
            pixel_x: 112,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "water_inner_bl",
            pixel_x: 96,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "water_inner_br",
            pixel_x: 112,
            pixel_y: 176,
        },
        // --- NARROW 1x1 WATER  ---
        TilemapSprite {
            name: "water_narrow_v",
            pixel_x: 96,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "water_narrow_h",
            pixel_x: 112,
            pixel_y: 192,
        },
        // --- WATER w/ DECOR  ---
        TilemapSprite {
            name: "water_lily_flower",
            pixel_x: 112,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "water_lily_plain",
            pixel_x: 112,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "water_sparkle",
            pixel_x: 112,
            pixel_y: 240,
        },
    ],
};

pub const DECOR_TILEMAP: TilemapDefinition = TilemapDefinition {
    tile_width: 16,
    tile_height: 16,
    atlas_width: 256,
    atlas_height: 256,
    sprites: &[
        // --- 2 LARGE BUSHES ---
        TilemapSprite {
            name: "bush_lg_g_0_0",
            pixel_x: 16,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "bush_lg_g_0_1",
            pixel_x: 32,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "bush_lg_g_1_0",
            pixel_x: 16,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "bush_lg_g_1_1",
            pixel_x: 32,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "bush_lg_d_0_0",
            pixel_x: 16,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "bush_lg_d_0_1",
            pixel_x: 32,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "bush_lg_d_1_0",
            pixel_x: 16,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "bush_lg_d_1_1",
            pixel_x: 32,
            pixel_y: 64,
        },
        // --- 2 LOGS ---
        TilemapSprite {
            name: "log_g_0_0",
            pixel_x: 64,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "log_g_0_1",
            pixel_x: 80,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "log_g_0_2",
            pixel_x: 96,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "log_g_1_0",
            pixel_x: 64,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "log_g_1_1",
            pixel_x: 80,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "log_g_1_2",
            pixel_x: 96,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "log_d_0_0",
            pixel_x: 64,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "log_d_0_1",
            pixel_x: 80,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "log_d_0_2",
            pixel_x: 96,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "log_d_1_0",
            pixel_x: 64,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "log_d_1_1",
            pixel_x: 80,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "log_d_1_2",
            pixel_x: 96,
            pixel_y: 64,
        },
        // --- 2 SMALL BUSHES ---
        TilemapSprite {
            name: "bush_sm_g_0_0",
            pixel_x: 128,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "bush_sm_g_0_1",
            pixel_x: 144,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "bush_sm_g_1_0",
            pixel_x: 128,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "bush_sm_g_1_1",
            pixel_x: 144,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "bush_sm_d_0_0",
            pixel_x: 128,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "bush_sm_d_0_1",
            pixel_x: 144,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "bush_sm_d_1_0",
            pixel_x: 128,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "bush_sm_d_1_1",
            pixel_x: 144,
            pixel_y: 64,
        },
        // --- 9 MUSHROOMS ---
        TilemapSprite {
            name: "mush_g1_0_0",
            pixel_x: 160,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g1_0_1",
            pixel_x: 176,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g1_1_0",
            pixel_x: 160,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_g1_1_1",
            pixel_x: 176,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_d1_0_0",
            pixel_x: 160,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d1_0_1",
            pixel_x: 176,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d1_1_0",
            pixel_x: 160,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_d1_1_1",
            pixel_x: 176,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_g2_0_0",
            pixel_x: 192,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g2_0_1",
            pixel_x: 208,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g2_1_0",
            pixel_x: 192,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_g2_1_1",
            pixel_x: 208,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_d2_0_0",
            pixel_x: 192,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d2_0_1",
            pixel_x: 208,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d2_1_0",
            pixel_x: 192,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_d2_1_1",
            pixel_x: 208,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_g3_0_0",
            pixel_x: 224,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g3_0_1",
            pixel_x: 240,
            pixel_y: 16,
        },
        TilemapSprite {
            name: "mush_g3_1_0",
            pixel_x: 224,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_g3_1_1",
            pixel_x: 240,
            pixel_y: 32,
        },
        TilemapSprite {
            name: "mush_d3_0_0",
            pixel_x: 224,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d3_0_1",
            pixel_x: 240,
            pixel_y: 48,
        },
        TilemapSprite {
            name: "mush_d3_1_0",
            pixel_x: 224,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_d3_1_1",
            pixel_x: 240,
            pixel_y: 64,
        },
        TilemapSprite {
            name: "mush_g4_0_0",
            pixel_x: 160,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g4_0_1",
            pixel_x: 176,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g4_1_0",
            pixel_x: 160,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "mush_g4_1_1",
            pixel_x: 176,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "mush_g5_0_0",
            pixel_x: 192,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g5_0_1",
            pixel_x: 208,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g5_1_0",
            pixel_x: 192,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "mush_g5_1_1",
            pixel_x: 208,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "mush_g6_0_0",
            pixel_x: 224,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g6_0_1",
            pixel_x: 240,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "mush_g6_1_0",
            pixel_x: 224,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "mush_g6_1_1",
            pixel_x: 240,
            pixel_y: 96,
        },
        // --- 2 WEEDS & 2 REEDS ---
        TilemapSprite {
            name: "weed_g_0_0",
            pixel_x: 16,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "weed_g_0_1",
            pixel_x: 32,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "weed_g_1_0",
            pixel_x: 16,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "weed_g_1_1",
            pixel_x: 32,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "weed_d_0_0",
            pixel_x: 48,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "weed_d_0_1",
            pixel_x: 64,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "weed_d_1_0",
            pixel_x: 48,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "weed_d_1_1",
            pixel_x: 64,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "reed_g_0_0",
            pixel_x: 80,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "reed_g_0_1",
            pixel_x: 96,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "reed_g_1_0",
            pixel_x: 80,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "reed_g_1_1",
            pixel_x: 96,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "reed_d_0_0",
            pixel_x: 112,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "reed_d_0_1",
            pixel_x: 128,
            pixel_y: 80,
        },
        TilemapSprite {
            name: "reed_d_1_0",
            pixel_x: 112,
            pixel_y: 96,
        },
        TilemapSprite {
            name: "reed_d_1_1",
            pixel_x: 128,
            pixel_y: 96,
        },
        // --- PEBBLE ---
        TilemapSprite {
            name: "pebble_g_0_0",
            pixel_x: 128,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "pebble_g_0_1",
            pixel_x: 144,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "pebble_g_1_0",
            pixel_x: 128,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "pebble_g_1_1",
            pixel_x: 144,
            pixel_y: 128,
        },
        // --- 3 ROCKS ---
        TilemapSprite {
            name: "rock_g1_0_0",
            pixel_x: 160,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g1_0_1",
            pixel_x: 176,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g1_1_0",
            pixel_x: 160,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "rock_g1_1_1",
            pixel_x: 176,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "rock_g2_0_0",
            pixel_x: 192,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g2_0_1",
            pixel_x: 208,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g2_1_0",
            pixel_x: 192,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "rock_g2_1_1",
            pixel_x: 208,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "rock_g3_0_0",
            pixel_x: 224,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g3_0_1",
            pixel_x: 240,
            pixel_y: 112,
        },
        TilemapSprite {
            name: "rock_g3_1_0",
            pixel_x: 224,
            pixel_y: 128,
        },
        TilemapSprite {
            name: "rock_g3_1_1",
            pixel_x: 240,
            pixel_y: 128,
        },
        // --- 3 BOULDERS ---
        TilemapSprite {
            name: "boulder_g1_0_0",
            pixel_x: 160,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g1_0_1",
            pixel_x: 176,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g1_1_0",
            pixel_x: 160,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "boulder_g1_1_1",
            pixel_x: 176,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "boulder_g2_0_0",
            pixel_x: 192,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g2_0_1",
            pixel_x: 208,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g2_1_0",
            pixel_x: 192,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "boulder_g2_1_1",
            pixel_x: 208,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "boulder_g3_0_0",
            pixel_x: 224,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g3_0_1",
            pixel_x: 240,
            pixel_y: 144,
        },
        TilemapSprite {
            name: "boulder_g3_1_0",
            pixel_x: 224,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "boulder_g3_1_1",
            pixel_x: 240,
            pixel_y: 160,
        },
        // --- 2 OAK TREES ---
        // Grass Base
        TilemapSprite {
            name: "oak_g_0_0",
            pixel_x: 0,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_g_0_1",
            pixel_x: 16,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_g_0_2",
            pixel_x: 32,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_g_0_3",
            pixel_x: 48,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_g_0_4",
            pixel_x: 64,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_g_1_0",
            pixel_x: 0,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_g_1_1",
            pixel_x: 16,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_g_1_2",
            pixel_x: 32,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_g_1_3",
            pixel_x: 48,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_g_1_4",
            pixel_x: 64,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_g_2_0",
            pixel_x: 0,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_g_2_1",
            pixel_x: 16,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_g_2_2",
            pixel_x: 32,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_g_2_3",
            pixel_x: 48,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_g_2_4",
            pixel_x: 64,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_g_3_0",
            pixel_x: 0,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_g_3_1",
            pixel_x: 16,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_g_3_2",
            pixel_x: 32,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_g_3_3",
            pixel_x: 48,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_g_3_4",
            pixel_x: 64,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_g_4_0",
            pixel_x: 0,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_g_4_1",
            pixel_x: 16,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_g_4_2",
            pixel_x: 32,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_g_4_3",
            pixel_x: 48,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_g_4_4",
            pixel_x: 64,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_g_5_0",
            pixel_x: 0,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_g_5_1",
            pixel_x: 16,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_g_5_2",
            pixel_x: 32,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_g_5_3",
            pixel_x: 48,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_g_5_4",
            pixel_x: 64,
            pixel_y: 240,
        },
        // Dirt Base
        TilemapSprite {
            name: "oak_d_0_0",
            pixel_x: 80,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_d_0_1",
            pixel_x: 96,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_d_0_2",
            pixel_x: 112,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_d_0_3",
            pixel_x: 128,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_d_0_4",
            pixel_x: 144,
            pixel_y: 160,
        },
        TilemapSprite {
            name: "oak_d_1_0",
            pixel_x: 80,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_d_1_1",
            pixel_x: 96,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_d_1_2",
            pixel_x: 112,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_d_1_3",
            pixel_x: 128,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_d_1_4",
            pixel_x: 144,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "oak_d_2_0",
            pixel_x: 80,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_d_2_1",
            pixel_x: 96,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_d_2_2",
            pixel_x: 112,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_d_2_3",
            pixel_x: 128,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_d_2_4",
            pixel_x: 144,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "oak_d_3_0",
            pixel_x: 80,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_d_3_1",
            pixel_x: 96,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_d_3_2",
            pixel_x: 112,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_d_3_3",
            pixel_x: 128,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_d_3_4",
            pixel_x: 144,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "oak_d_4_0",
            pixel_x: 80,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_d_4_1",
            pixel_x: 96,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_d_4_2",
            pixel_x: 112,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_d_4_3",
            pixel_x: 128,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_d_4_4",
            pixel_x: 144,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "oak_d_5_0",
            pixel_x: 80,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_d_5_1",
            pixel_x: 96,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_d_5_2",
            pixel_x: 112,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_d_5_3",
            pixel_x: 128,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "oak_d_5_4",
            pixel_x: 144,
            pixel_y: 240,
        },
        // --- 2 PINE TREES ---
        // Grass Base
        TilemapSprite {
            name: "pine_g_0_0",
            pixel_x: 176,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_g_0_1",
            pixel_x: 192,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_g_0_2",
            pixel_x: 208,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_g_1_0",
            pixel_x: 176,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_g_1_1",
            pixel_x: 192,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_g_1_2",
            pixel_x: 208,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_g_2_0",
            pixel_x: 176,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_g_2_1",
            pixel_x: 192,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_g_2_2",
            pixel_x: 208,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_g_3_0",
            pixel_x: 176,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_g_3_1",
            pixel_x: 192,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_g_3_2",
            pixel_x: 208,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_g_4_0",
            pixel_x: 176,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "pine_g_4_1",
            pixel_x: 192,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "pine_g_4_2",
            pixel_x: 208,
            pixel_y: 240,
        },
        // Bare Base
        TilemapSprite {
            name: "pine_b_0_0",
            pixel_x: 224,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_b_0_1",
            pixel_x: 240,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_b_0_2",
            pixel_x: 256,
            pixel_y: 176,
        },
        TilemapSprite {
            name: "pine_b_1_0",
            pixel_x: 224,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_b_1_1",
            pixel_x: 240,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_b_1_2",
            pixel_x: 256,
            pixel_y: 192,
        },
        TilemapSprite {
            name: "pine_b_2_0",
            pixel_x: 224,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_b_2_1",
            pixel_x: 240,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_b_2_2",
            pixel_x: 256,
            pixel_y: 208,
        },
        TilemapSprite {
            name: "pine_b_3_0",
            pixel_x: 224,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_b_3_1",
            pixel_x: 240,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_b_3_2",
            pixel_x: 256,
            pixel_y: 224,
        },
        TilemapSprite {
            name: "pine_b_4_0",
            pixel_x: 224,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "pine_b_4_1",
            pixel_x: 240,
            pixel_y: 240,
        },
        TilemapSprite {
            name: "pine_b_4_2",
            pixel_x: 256,
            pixel_y: 240,
        },
    ],
};
