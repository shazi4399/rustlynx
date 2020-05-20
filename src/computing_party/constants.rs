pub const SIZEOF_MEGABYTE: usize = 1 << 20;
pub const SIZEOF_U128: usize = 1 << 4;
pub const SIZEOF_U64: usize = 1 << 3;
pub const SIZEOF_U32: usize = 1 << 2;
pub const SIZEOF_U16: usize = 1 << 1;
pub const SIZEOF_U8: usize = 1;
pub const BITLENGTH: usize = 64;

pub const ML_MODELS: [&str ; 1] = [
	
	"logisticregression",
	
];


pub const ML_PHASES: [&str ; 2] = [

	"learning",
	"inference",
];

pub const TEST_CR_WRAPPING_U8_0: (u128, u128, u128) = (
		0x03,
		0x2a,
		0xb2,
	);

pub const TEST_CR_WRAPPING_U8_1: (u128, u128, u128) = (
		0x3a,
		0x8c,
		0xac,
	);

pub const TEST_CR_WRAPPING_U16_0: (u128, u128, u128) = (
		0xc2c7,
		0xf8f2,
		0x5cc4,
	);

pub const TEST_CR_WRAPPING_U16_1: (u128, u128, u128) = (
		0x1969,
		0xf191,
		0x2fcc,
	);

pub const TEST_CR_WRAPPING_U32_0: (u128, u128, u128) = (
		0x565e6676,
		0xfa40944f,
		0x8f7f03f2,
	);

pub const TEST_CR_WRAPPING_U32_1: (u128, u128, u128) = (
		0xb5941981,
		0x8d8ea360,
		0x6e7d86e7,
	);

pub const TEST_CR_WRAPPING_U64_0: (u64, u64, u64) = (
		0x91726f49cade6a1a,
		0xe5b959fd785813fe,
		0xffdbcf4a9eb6e119,
	);

pub const TEST_CR_WRAPPING_U64_1: (u64, u64, u64) = (
		0xa57a5db4030e7993,
		0x7c5d53166af86ce0,
		0x4e351284f39d0eed,
	);

pub const TEST_CR_WRAPPING_U128_0: (u128, u128, u128) = (
		0x9cda50f0e54429304f72856fc0fdb923,
		0xbcd229d3361b9eb42559b0e4e268ee00,
		0x4dad9e6214340bc9e0d48b35f686386f,
	);

pub const TEST_CR_WRAPPING_U128_1: (u128, u128, u128) = (
		0xbc7a1d9961e4d5941955488e4a256d91,
		0xa17dced6edc410016c07bc8ad4f94af1,
		0x0f4a19f55913052c5776c38db2f09705,
	);

pub const TEST_CR_XOR_U128_0: (u128, u128, u128) = (
		0x602113ab721dfbb0a41aec62e0aa9d16,
		0x3347a4e096013fd85938a2f1cdf35047,
		0x4e8d7fb29621dcbb373d03cd71d77483,
	);

pub const TEST_CR_XOR_U128_1: (u128, u128, u128) = (
		0xb5f0592ca4dc5b3e3628159382e3a1c7,
		0x5d63a3d79b3dd43c48e20ce7a80e14ef,
		0x0a8d7db592217c3f272fabdd119e7003,
	);





