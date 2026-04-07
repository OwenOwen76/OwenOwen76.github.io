use crate::map::math::{NoiseHasher, PermutationTable, Quintic, Vector2, Vector3, Vector4, linear};
use core::f64;

pub trait NoiseFn<T, const DIM: usize> {
    fn get(&self, point: [T; DIM]) -> f64;
}

impl<T, M, const DIM: usize> NoiseFn<T, DIM> for &M
where
    M: NoiseFn<T, DIM> + ?Sized,
{
    #[inline]
    fn get(&self, point: [T; DIM]) -> f64 {
        M::get(*self, point)
    }
}

impl<T, M, const DIM: usize> NoiseFn<T, DIM> for Box<M>
where
    M: NoiseFn<T, DIM> + ?Sized,
{
    #[inline]
    fn get(&self, point: [T; DIM]) -> f64 {
        M::get(self, point)
    }
}

pub trait Seedable {
    fn set_seed(self, seed: u32) -> Self;

    fn seed(&self) -> u32;
}

#[inline(always)]
pub fn perlin_1d<NH>(point: f64, hasher: &NH) -> f64
where
    NH: NoiseHasher + ?Sized,
{
    const SCALE_FACTOR: f64 = 2.0;

    let corner = point.floor() as isize;
    let distance = point - corner as f64;

    macro_rules! call_gradient(
        ($x_offset:expr) => {
            {
                let offset = distance - $x_offset as f64;
                match hasher.hash(&[corner + $x_offset]) & 0b1 {
                    0 =>  offset,
                    1 => -offset,
                    _ => unreachable!(),
                }
            }
        }
    );

    let g0 = call_gradient!(0);
    let g1 = call_gradient!(1);

    let curve = distance.map_quintic();

    let result = linear(g0, g1, curve) * SCALE_FACTOR;

    result.clamp(-1.0, 1.0)
}

#[inline(always)]
pub fn perlin_2d<NH>(point: Vector2<f64>, hasher: &NH) -> f64
where
    NH: NoiseHasher + ?Sized,
{
    const SCALE_FACTOR: f64 = 2.0 / f64::consts::SQRT_2;

    let corner = point.floor_to_isize();
    let distance = point - corner.numcast().unwrap();

    macro_rules! call_gradient(
        ($x:expr, $y:expr) => {
            {
                let offset = Vector2::new($x, $y);
                let point = distance - offset.numcast().unwrap();

                match hasher.hash(&(corner + offset).into_array()) & 0b11 {
                    0 =>  point.x + point.y,
                    1 => -point.x + point.y,
                    2 =>  point.x - point.y,
                    3 => -point.x - point.y,
                    _ => unreachable!(),
                }
            }
        }
    );

    let g00 = call_gradient!(0, 0);
    let g10 = call_gradient!(1, 0);
    let g01 = call_gradient!(0, 1);
    let g11 = call_gradient!(1, 1);

    let curve = distance.map_quintic();

    let result = linear(
        linear(g00, g01, curve.y),
        linear(g10, g11, curve.y),
        curve.x,
    ) * SCALE_FACTOR;

    result.clamp(-1.0, 1.0)
}

#[inline(always)]
pub fn perlin_3d<NH>(point: Vector3<f64>, hasher: &NH) -> f64
where
    NH: NoiseHasher + ?Sized,
{
    const SCALE_FACTOR: f64 = 1.154_700_538_379_251_5;

    let corner = point.floor_to_isize();
    let distance = point - corner.numcast().unwrap();

    macro_rules! call_gradient(
        ($x:expr, $y:expr, $z:expr) => {
            {
                let offset = Vector3::new($x, $y, $z);
                let point = distance - offset.numcast().unwrap();

                match hasher.hash(&(corner + offset).into_array()) & 0b1111 {
                    0  | 12 =>  point.x + point.y,
                    1  | 13 => -point.x + point.y,
                    2       =>  point.x - point.y,
                    3       => -point.x - point.y,
                    4       =>  point.x + point.z,
                    5       => -point.x + point.z,
                    6       =>  point.x - point.z,
                    7       => -point.x - point.z,
                    8       =>  point.y + point.z,
                    9  | 14 => -point.y + point.z,
                    10      =>  point.y - point.z,
                    11 | 15 => -point.y - point.z,
                    _ => unreachable!(),
                }
            }
        }
    );

    let g000 = call_gradient!(0, 0, 0);
    let g100 = call_gradient!(1, 0, 0);
    let g010 = call_gradient!(0, 1, 0);
    let g110 = call_gradient!(1, 1, 0);
    let g001 = call_gradient!(0, 0, 1);
    let g101 = call_gradient!(1, 0, 1);
    let g011 = call_gradient!(0, 1, 1);
    let g111 = call_gradient!(1, 1, 1);

    let curve = distance.map_quintic();

    let result = linear(
        linear(
            linear(g000, g001, curve.z),
            linear(g010, g011, curve.z),
            curve.y,
        ),
        linear(
            linear(g100, g101, curve.z),
            linear(g110, g111, curve.z),
            curve.y,
        ),
        curve.x,
    ) * SCALE_FACTOR;

    result.clamp(-1.0, 1.0)
}

#[inline(always)]
pub fn perlin_4d<NH>(point: Vector4<f64>, hasher: &NH) -> f64
where
    NH: NoiseHasher + ?Sized,
{
    const SCALE_FACTOR: f64 = 1.0;

    let corner = point.floor_to_isize();
    let distance = point - corner.numcast().unwrap();

    macro_rules! call_gradient(
        ($x:expr, $y:expr, $z:expr, $w:expr) => {
            {
                let offset = Vector4::new($x, $y, $z, $w);
                let point = distance - offset.numcast().unwrap();

                match hasher.hash(&(corner + offset).into_array()) & 0b11111 {
                    0  | 28 =>  point.x + point.y + point.z,
                    1       => -point.x + point.y + point.z,
                    2       =>  point.x - point.y + point.z,
                    3       =>  point.x + point.y - point.z,
                    4       => -point.x + point.y - point.z,
                    5       =>  point.x - point.y - point.z,
                    6       =>  point.x - point.y - point.z,
                    7  | 29 =>  point.x + point.y + point.w,
                    8       => -point.x + point.y + point.w,
                    9       =>  point.x - point.y + point.w,
                    10      =>  point.x + point.y - point.w,
                    11      =>  point.x + point.y - point.w,
                    12      =>  point.x + point.y - point.w,
                    13      => -point.x - point.y - point.w,
                    14 | 30 =>  point.x + point.z + point.w,
                    15      => -point.x + point.z + point.w,
                    16      =>  point.x - point.z + point.w,
                    17      =>  point.x + point.z - point.w,
                    18      =>  point.x + point.z - point.w,
                    19      =>  point.x + point.z - point.w,
                    20      => -point.x - point.z - point.w,
                    21 | 31 =>  point.y + point.z + point.w,
                    22      => -point.y + point.z + point.w,
                    23      =>  point.y - point.z + point.w,
                    24      =>  point.y - point.z - point.w,
                    25      => -point.y - point.z - point.w,
                    26      =>  point.y - point.z - point.w,
                    27      => -point.y - point.z - point.w,
                    _ => unreachable!(),
                }
            }
        }
    );

    let g0000 = call_gradient!(0, 0, 0, 0);
    let g1000 = call_gradient!(1, 0, 0, 0);
    let g0100 = call_gradient!(0, 1, 0, 0);
    let g1100 = call_gradient!(1, 1, 0, 0);
    let g0010 = call_gradient!(0, 0, 1, 0);
    let g1010 = call_gradient!(1, 0, 1, 0);
    let g0110 = call_gradient!(0, 1, 1, 0);
    let g1110 = call_gradient!(1, 1, 1, 0);
    let g0001 = call_gradient!(0, 0, 0, 1);
    let g1001 = call_gradient!(1, 0, 0, 1);
    let g0101 = call_gradient!(0, 1, 0, 1);
    let g1101 = call_gradient!(1, 1, 0, 1);
    let g0011 = call_gradient!(0, 0, 1, 1);
    let g1011 = call_gradient!(1, 0, 1, 1);
    let g0111 = call_gradient!(0, 1, 1, 1);
    let g1111 = call_gradient!(1, 1, 1, 1);

    let curve = distance.map_quintic();

    let result = linear(
        linear(
            linear(
                linear(g0000, g0001, curve.w),
                linear(g0010, g0011, curve.w),
                curve.z,
            ),
            linear(
                linear(g0100, g0101, curve.w),
                linear(g0110, g0111, curve.w),
                curve.z,
            ),
            curve.y,
        ),
        linear(
            linear(
                linear(g1000, g1001, curve.w),
                linear(g1010, g1011, curve.w),
                curve.z,
            ),
            linear(
                linear(g1100, g1101, curve.w),
                linear(g1110, g1111, curve.w),
                curve.z,
            ),
            curve.y,
        ),
        curve.x,
    ) * SCALE_FACTOR;

    result.clamp(-1.0, 1.0)
}

#[derive(Clone, Copy, Debug)]
pub struct Perlin {
    seed: u32,
    perm_table: PermutationTable,
}

impl Perlin {
    pub const DEFAULT_SEED: u32 = 0;

    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            perm_table: PermutationTable::new(seed),
        }
    }
}

impl Default for Perlin {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}

impl Seedable for Perlin {
    fn set_seed(self, seed: u32) -> Self {
        if self.seed == seed {
            return self;
        }

        Self {
            seed,
            perm_table: PermutationTable::new(seed),
        }
    }

    fn seed(&self) -> u32 {
        self.seed
    }
}

impl NoiseFn<f64, 1> for Perlin {
    fn get(&self, point: [f64; 1]) -> f64 {
        perlin_1d(point[0], &self.perm_table)
    }
}

impl NoiseFn<f64, 2> for Perlin {
    fn get(&self, point: [f64; 2]) -> f64 {
        perlin_2d(point.into(), &self.perm_table)
    }
}

impl NoiseFn<f64, 3> for Perlin {
    fn get(&self, point: [f64; 3]) -> f64 {
        perlin_3d(point.into(), &self.perm_table)
    }
}

impl NoiseFn<f64, 4> for Perlin {
    fn get(&self, point: [f64; 4]) -> f64 {
        perlin_4d(point.into(), &self.perm_table)
    }
}
